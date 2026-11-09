"""Motor y ejecutor autónomo de auditoría diaria de telemetría y transcripciones de AiHoreca.

Flujo de ejecución:
1. Sincronización descendente (pull) desde el repositorio privado de Hugging Face Dataset.
2. Archivo local de sesiones completas en `data/logs/sesiones/<fecha>_<session_id>.txt`.
3. Detección incremental basada en cursor (`data/logs/.sync_cursor.json`) de turnos nuevos
   y de votos recibidos a posteriori sobre turnos ya auditados.
4. Informe diario en `data/logs/auditorias/<fecha>_informe_auditoria.md` (fecha Europe/Madrid;
   varias ejecuciones el mismo día se añaden al mismo informe).
5. Histórico de métricas agregadas sin texto libre en `data/logs/historico_metricas.csv`.
6. Cola de revisión de candidatos al golden set en `data/logs/candidatos_golden.jsonl`.
7. Retención local (SESSION_RETENTION_DAYS) de transcripciones, informes y candidatos.
8. Compactación mensual programada (squash) del historial Git en Hugging Face Dataset.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

from src.app.telemetry_sync import HuggingFaceTelemetrySync, get_hf_telemetry_config
from src.config import (
    AUDIT_CURSOR_PATH,
    AUDIT_REPORTS_DIR,
    HF_SQUASH_INTERVAL_DAYS,
    LOGS_DIR,
    SESSION_ARCHIVE_DIR,
    SESSION_RETENTION_DAYS,
    SESSIONS_DB_PATH,
)

try:
    from zoneinfo import ZoneInfo

    LOCAL_TZ = ZoneInfo("Europe/Madrid")
except Exception:  # pragma: no cover - fallback si no hay base de datos de zonas horarias
    LOCAL_TZ = timezone.utc

METRICS_HISTORY_PATH = LOGS_DIR / "historico_metricas.csv"
GOLDEN_CANDIDATES_PATH = LOGS_DIR / "candidatos_golden.jsonl"

# Valores reales de feedback persistidos por la app (ver SessionMemoryStore.update_turn_feedback y ui.py)
FEEDBACK_POSITIVE = "OK"
FEEDBACK_NEGATIVE = "KO"
FEEDBACK_NONE = "SIN_VOTO"

_REPORT_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_informe_auditoria\.md$")
_ARCHIVE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|sin_fecha)_(.+)\.txt$")


def _now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_cursor(cursor_path: Path) -> dict[str, Any]:
    default = {
        "last_audited_turn_id": 0,
        "last_audited_timestamp": "",
        "last_feedback_at": "",
        "last_hf_squash_date": "",
        "updated_at": "",
    }
    if cursor_path.exists():
        try:
            default.update(json.loads(cursor_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return default


def _save_cursor(cursor_path: Path, data: dict[str, Any]) -> None:
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cursor_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _turn_metrics(t: sqlite3.Row) -> dict[str, Any]:
    audit = json.loads(t["audit_json"] or "{}")
    timings = json.loads(t["timings_json"] or "{}")
    return {
        "audit": audit,
        "latency_ms": round(sum(float(v) for v in timings.values()), 1),
        "unverified": audit.get("unverified_numbers") or [],
        "passed": bool(audit.get("passed", True)),
        "feedback": t["user_feedback"] or FEEDBACK_NONE,
    }


def _feedback_icon(fb: str) -> str:
    return {"OK": "👍 OK", "KO": "👎 KO"}.get(fb, "Sin voto")


# ---------------------------------------------------------------------------
# 1. Archivo local de transcripciones
# ---------------------------------------------------------------------------


def archive_session_transcripts(db_path: Path, archive_dir: Path) -> int:
    """Exporta y actualiza la transcripción completa y legible de cada sesión en disco."""
    if not db_path.exists():
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_count = 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute("SELECT * FROM sessions ORDER BY created_at ASC").fetchall()

        for s in sessions:
            sid = s["session_id"]
            created = s["created_at"] or ""
            date_prefix = created[:10] if len(created) >= 10 else "sin_fecha"
            safe_sid = sid.replace("/", "_").replace("\\", "_")
            out_file = archive_dir / f"{date_prefix}_{safe_sid}.txt"

            turns = conn.execute(
                "SELECT * FROM session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                (sid,),
            ).fetchall()

            n_ok = sum(1 for t in turns if t["user_feedback"] == FEEDBACK_POSITIVE)
            n_ko = sum(1 for t in turns if t["user_feedback"] == FEEDBACK_NEGATIVE)

            lines: list[str] = [
                "=" * 80,
                f"SESIÓN: {sid}",
                f"TÍTULO: {s['title']}",
                f"INICIO: {s['created_at']} | ÚLTIMA ACTIVIDAD: {s['updated_at']}",
                f"TERRITORIO: {s['active_territorio'] or 'No fijado'} | "
                f"OFICIO: {s['active_oficio_coloquial'] or 'No fijado'} | "
                f"CATEGORÍA ALEH: {s['active_categoria_aleh'] or 'No fijado'}",
                f"TOTAL TURNOS: {len(turns)} | FEEDBACK ACUMULADO: 👍 {n_ok} / 👎 {n_ko}",
                f"CADUCIDAD: se elimina {SESSION_RETENTION_DAYS} días después de la última actividad.",
                "=" * 80,
                "",
            ]

            for t in turns:
                m = _turn_metrics(t)
                models = json.loads(t["models_json"] or "{}")
                v_count = len(m["audit"].get("verified_numbers") or []) + len(
                    m["audit"].get("derived_numbers") or []
                )
                feedback_str = _feedback_icon(m["feedback"])
                if t["feedback_at"]:
                    feedback_str += f" ({t['feedback_at']})"

                lines.extend([
                    f"[Turno {t['turn_index']} - {t['timestamp']}]",
                    f"Usuario: {t['user_query']}",
                    f"(Efectiva: {t['effective_query']} | Intención: {t['intent']} | Ruta: {t['route_decision']})",
                    "",
                    f"Asistente (Modelo síntesis: {models.get('legal_synthesis_and_audit', 'n/d')}):",
                    t["final_response"],
                    "",
                    f"Feedback usuario: {feedback_str}",
                    f"Métricas turno: Latencia {m['latency_ms']} ms | Auditoría numérica: "
                    f"{'APROBADO' if m['passed'] else 'CON INCIDENCIAS'} "
                    f"({v_count} verificados, {len(m['unverified'])} no verificados)",
                    "-" * 80,
                    "",
                ])

            out_file.write_text("\n".join(lines), encoding="utf-8")
            archived_count += 1
    finally:
        conn.close()

    return archived_count


# ---------------------------------------------------------------------------
# 2. Auditoría incremental (turnos nuevos + votos tardíos)
# ---------------------------------------------------------------------------


def _append_metrics_history(row: dict[str, Any]) -> None:
    METRICS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not METRICS_HISTORY_PATH.exists()
    with METRICS_HISTORY_PATH.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _export_golden_candidates(candidates: list[dict[str, Any]]) -> int:
    """Añade turnos marcados a la cola de revisión, sin duplicar por turn_id."""
    if not candidates:
        return 0
    GOLDEN_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[int] = set()
    if GOLDEN_CANDIDATES_PATH.exists():
        for line in GOLDEN_CANDIDATES_PATH.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(int(json.loads(line)["turn_id"]))
            except Exception:
                continue
    added = 0
    with GOLDEN_CANDIDATES_PATH.open("a", encoding="utf-8") as f:
        for c in candidates:
            if c["turn_id"] in existing_ids:
                continue
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            existing_ids.add(c["turn_id"])
            added += 1
    return added


def extract_and_audit_deltas(
    db_path: Path,
    cursor_path: Path,
    reports_dir: Path,
    force_all: bool = False,
    export_candidates: bool = False,
) -> dict[str, Any]:
    """Detecta turnos nuevos y votos tardíos desde la última auditoría y genera el informe."""
    if not db_path.exists():
        return {"new_turns": 0, "late_votes": 0, "status": "db_not_found"}

    reports_dir.mkdir(parents=True, exist_ok=True)
    cursor = _load_cursor(cursor_path)
    last_turn_id = 0 if force_all else int(cursor.get("last_audited_turn_id", 0) or 0)
    last_feedback_at = "" if force_all else str(cursor.get("last_feedback_at", "") or "")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        new_turns = conn.execute(
            """
            SELECT t.*, s.title AS session_title
            FROM session_turns t JOIN sessions s ON t.session_id = s.session_id
            WHERE t.id > ?
            ORDER BY t.id ASC
            """,
            (last_turn_id,),
        ).fetchall()

        # Votos emitidos después de la última auditoría sobre turnos que ya estaban auditados
        late_votes = []
        if not force_all:
            late_votes = conn.execute(
                """
                SELECT t.*, s.title AS session_title
                FROM session_turns t JOIN sessions s ON t.session_id = s.session_id
                WHERE t.id <= ? AND t.feedback_at IS NOT NULL AND t.feedback_at > ?
                ORDER BY t.feedback_at ASC
                """,
                (last_turn_id, last_feedback_at),
            ).fetchall()

        max_feedback_row = conn.execute(
            "SELECT MAX(feedback_at) AS m FROM session_turns WHERE feedback_at IS NOT NULL"
        ).fetchone()
        max_feedback_at = (max_feedback_row["m"] if max_feedback_row else "") or last_feedback_at
    finally:
        conn.close()

    if not new_turns and not late_votes:
        return {
            "new_turns": 0,
            "late_votes": 0,
            "status": "up_to_date",
            "last_audited_turn_id": last_turn_id,
        }

    now_local = _now_local()
    feedback_count = {FEEDBACK_POSITIVE: 0, FEEDBACK_NEGATIVE: 0, FEEDBACK_NONE: 0}
    intents_count: dict[str, int] = {}
    routes_count: dict[str, int] = {}
    territorios_count: dict[str, int] = {}
    latencies_ms: list[float] = []
    unverified_total = 0
    flagged: list[dict[str, Any]] = []

    def _flag_entry(t: sqlite3.Row, m: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
        return {
            "turn_id": t["id"],
            "session_id": t["session_id"],
            "turn_index": t["turn_index"],
            "timestamp": t["timestamp"],
            "query": t["user_query"],
            "effective_query": t["effective_query"],
            "territorio": t["territorio"],
            "oficio": t["oficio_detectado"],
            "intent": t["intent"],
            "route": t["route_decision"],
            "response": t["final_response"],
            "feedback": m["feedback"],
            "feedback_at": t["feedback_at"],
            "unverified": m["unverified"],
            "latency_ms": m["latency_ms"],
            "reasons": reasons,
        }

    def _reasons(m: dict[str, Any]) -> list[str]:
        r = []
        if m["feedback"] == FEEDBACK_NEGATIVE:
            r.append("voto_negativo")
        if m["unverified"]:
            r.append("cifras_no_verificadas")
        if not m["passed"]:
            r.append("auditoria_numerica_fallida")
        return r

    for t in new_turns:
        m = _turn_metrics(t)
        intents_count[t["intent"]] = intents_count.get(t["intent"], 0) + 1
        routes_count[t["route_decision"]] = routes_count.get(t["route_decision"], 0) + 1
        terr = t["territorio"] or "No especificado"
        territorios_count[terr] = territorios_count.get(terr, 0) + 1
        feedback_count[m["feedback"]] = feedback_count.get(m["feedback"], 0) + 1
        latencies_ms.append(m["latency_ms"])
        unverified_total += len(m["unverified"])
        reasons = _reasons(m)
        if reasons:
            flagged.append(_flag_entry(t, m, reasons))

    late_entries: list[dict[str, Any]] = []
    for t in late_votes:
        m = _turn_metrics(t)
        entry = _flag_entry(t, m, _reasons(m))
        late_entries.append(entry)
        if m["feedback"] == FEEDBACK_NEGATIVE:
            flagged.append(entry)

    n_new = len(new_turns)
    sessions_involved = {t["session_id"] for t in new_turns} | {t["session_id"] for t in late_votes}
    avg_lat = round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else 0.0
    lat_sorted = sorted(latencies_ms)
    p95_lat = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else 0.0
    voted = feedback_count[FEEDBACK_POSITIVE] + feedback_count[FEEDBACK_NEGATIVE]
    satisfaction = f"{round(100 * feedback_count[FEEDBACK_POSITIVE] / voted)}%" if voted else "n/d"
    max_id = max((t["id"] for t in new_turns), default=last_turn_id)
    max_ts = max((t["timestamp"] for t in new_turns), default=cursor.get("last_audited_timestamp", ""))

    # ---- Informe Markdown (sección de esta ejecución) ----
    md: list[str] = [
        f"## Ejecución {now_local.strftime('%H:%M')} (Europe/Madrid)",
        "",
        f"- **Turnos nuevos auditados**: {n_new}"
        + (f" (IDs `{last_turn_id + 1}`–`{max_id}`)" if n_new else ""),
        f"- **Votos tardíos sobre turnos ya auditados**: {len(late_entries)}",
        f"- **Sesiones involucradas**: {len(sessions_involved)}",
        f"- **Feedback (turnos nuevos)**: 👍 {feedback_count[FEEDBACK_POSITIVE]} | "
        f"👎 {feedback_count[FEEDBACK_NEGATIVE]} | Sin voto: {feedback_count[FEEDBACK_NONE]} "
        f"| Satisfacción sobre votados: {satisfaction}",
        f"- **Latencia**: media {avg_lat} ms | P95 {p95_lat} ms",
        f"- **Cifras no verificadas por el auditor numérico**: {unverified_total}",
        "",
    ]

    if n_new:
        md.extend(["### Distribución", "", "| Dimensión | Valor | Turnos |", "|---|---|---|"])
        for k, v in sorted(intents_count.items(), key=lambda x: -x[1]):
            md.append(f"| Intención | `{k}` | {v} |")
        for k, v in sorted(routes_count.items(), key=lambda x: -x[1]):
            md.append(f"| Ruta | `{k}` | {v} |")
        for k, v in sorted(territorios_count.items(), key=lambda x: -x[1]):
            md.append(f"| Territorio | {k} | {v} |")
        md.append("")

    if late_entries:
        md.extend(["### Votos tardíos", "", "| Sesión | Turno | Voto | Fecha voto | Consulta |", "|---|---|---|---|---|"])
        for e in late_entries:
            q = e["query"].replace("|", "/").replace("\n", " ")[:90]
            md.append(
                f"| `{e['session_id']}` | {e['turn_index']} | {_feedback_icon(e['feedback'])} | {e['feedback_at']} | {q} |"
            )
        md.append("")

    md.extend(["### Turnos que requieren revisión", ""])
    if not flagged:
        md.append("✅ Ningún turno con voto negativo ni incidencias numéricas.")
    else:
        for e in flagged:
            resp = e["response"].replace("\n", " ")
            md.extend([
                f"#### Sesión `{e['session_id']}` · turno {e['turn_index']} · turn_id {e['turn_id']}",
                f"- **Motivos**: {', '.join(e['reasons'])}",
                f"- **Consulta**: \"{e['query']}\"",
                f"- **Territorio / oficio**: {e['territorio'] or '—'} / {e['oficio'] or '—'} | "
                f"**Ruta**: `{e['route']}` | **Latencia**: {e['latency_ms']} ms",
                f"- **Feedback**: {_feedback_icon(e['feedback'])}",
                f"- **Cifras no verificadas**: {e['unverified'] or 'Ninguna'}",
                f"- **Extracto respuesta**: {resp[:300]}{'…' if len(resp) > 300 else ''}",
                f"- **Transcripción completa**: `data/logs/sesiones/*_{e['session_id']}.txt`",
                "",
            ])

    report_path = reports_dir / f"{now_local.strftime('%Y-%m-%d')}_informe_auditoria.md"
    header = [
        f"# Informe diario de auditoría de telemetría · AiHoreca · {now_local.strftime('%d/%m/%Y')}",
        "",
        f"> Conservación: este informe contiene extractos de consultas y se elimina a los {SESSION_RETENTION_DAYS} días. "
        "Las métricas agregadas se conservan en `historico_metricas.csv`.",
        "",
    ]
    if force_all or not report_path.exists():
        report_path.write_text("\n".join(header + md), encoding="utf-8")
    else:
        with report_path.open("a", encoding="utf-8") as f:
            f.write("\n\n" + "\n".join(md))

    # ---- Histórico de métricas (sin texto libre, conservación indefinida) ----
    _append_metrics_history({
        "fecha": now_local.strftime("%Y-%m-%d"),
        "hora": now_local.strftime("%H:%M"),
        "turn_id_desde": last_turn_id + 1 if n_new else "",
        "turn_id_hasta": max_id if n_new else "",
        "turnos_nuevos": n_new,
        "sesiones": len(sessions_involved),
        "votos_ok": feedback_count[FEEDBACK_POSITIVE],
        "votos_ko": feedback_count[FEEDBACK_NEGATIVE],
        "sin_voto": feedback_count[FEEDBACK_NONE],
        "votos_tardios": len(late_entries),
        "votos_tardios_ko": sum(1 for e in late_entries if e["feedback"] == FEEDBACK_NEGATIVE),
        "latencia_media_ms": avg_lat,
        "latencia_p95_ms": p95_lat,
        "cifras_no_verificadas": unverified_total,
        "turnos_marcados": len(flagged),
        "reauditoria_completa": int(force_all),
    })

    candidates_added = 0
    if export_candidates:
        candidates_added = _export_golden_candidates(
            [{**e, "exported_at": now_local.isoformat(timespec="seconds"), "estado": "pendiente_revision"} for e in flagged]
        )

    cursor["last_audited_turn_id"] = max_id
    cursor["last_audited_timestamp"] = max_ts
    cursor["last_feedback_at"] = max_feedback_at
    _save_cursor(cursor_path, cursor)

    return {
        "new_turns": n_new,
        "late_votes": len(late_entries),
        "status": "audited",
        "report_path": str(report_path),
        "flagged_turns": len(flagged),
        "feedback": feedback_count,
        "satisfaction": satisfaction,
        "avg_latency_ms": avg_lat,
        "golden_candidates_added": candidates_added,
    }


# ---------------------------------------------------------------------------
# 3. Retención local
# ---------------------------------------------------------------------------


def apply_local_retention(db_path: Path, retention_days: int = SESSION_RETENTION_DAYS) -> dict[str, int]:
    """Aplica en local el mismo plazo de conservación que en origen.

    - Transcripciones: se borran si la sesión ya no existe en la BD (purgada en producción)
      o si su última actividad supera `retention_days`.
    - Informes: se borran los de fecha (nombre de fichero) anterior a `retention_days`.
      Sus métricas ya están en `historico_metricas.csv` desde el día en que se generaron.
    - Candidatos al golden set: se eliminan los pendientes cuyo turno supera `retention_days`.
    """
    result = {"transcripts_deleted": 0, "reports_deleted": 0, "candidates_deleted": 0}
    if retention_days <= 0:
        return result

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_date = (_now_local() - timedelta(days=retention_days)).date()

    # Transcripciones
    session_activity: dict[str, datetime | None] = {}
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            for sid, updated in conn.execute("SELECT session_id, updated_at FROM sessions"):
                session_activity[sid.replace("/", "_").replace("\\", "_")] = _parse_iso(updated)
        finally:
            conn.close()

        if SESSION_ARCHIVE_DIR.exists():
            for f in SESSION_ARCHIVE_DIR.glob("*.txt"):
                m = _ARCHIVE_NAME_RE.match(f.name)
                if not m:
                    continue
                sid = m.group(2)
                updated = session_activity.get(sid)
                if sid not in session_activity or (updated is not None and updated < cutoff_dt):
                    f.unlink(missing_ok=True)
                    result["transcripts_deleted"] += 1

    # Informes
    if AUDIT_REPORTS_DIR.exists():
        for f in AUDIT_REPORTS_DIR.glob("*_informe_auditoria.md"):
            m = _REPORT_NAME_RE.match(f.name)
            if not m:
                continue
            try:
                report_date = date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if report_date < cutoff_date:
                f.unlink(missing_ok=True)
                result["reports_deleted"] += 1

    # Candidatos al golden set (cola de revisión con texto literal)
    if GOLDEN_CANDIDATES_PATH.exists():
        kept: list[str] = []
        for line in GOLDEN_CANDIDATES_PATH.read_text(encoding="utf-8").splitlines():
            try:
                ts = _parse_iso(json.loads(line).get("timestamp"))
            except Exception:
                ts = None
            if ts is not None and ts < cutoff_dt:
                result["candidates_deleted"] += 1
                continue
            kept.append(line)
        if result["candidates_deleted"]:
            GOLDEN_CANDIDATES_PATH.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# 4. Compactación mensual del historial en Hugging Face
# ---------------------------------------------------------------------------


def check_and_squash_hf_history(cursor_path: Path) -> dict[str, Any]:
    """Compacta el historial Git del dataset remoto si han pasado HF_SQUASH_INTERVAL_DAYS."""
    cursor = _load_cursor(cursor_path)
    last_squash = cursor.get("last_hf_squash_date", "")
    today = _now_local().date()

    needs_squash = True
    if last_squash:
        try:
            needs_squash = (today - date.fromisoformat(last_squash)).days >= HF_SQUASH_INTERVAL_DAYS
        except ValueError:
            needs_squash = True

    if not needs_squash:
        return {"squashed": False, "reason": f"Compactación al día (última: {last_squash})"}

    token, repo_id = get_hf_telemetry_config()
    if not token or not repo_id:
        return {"squashed": False, "reason": "Credenciales HF no configuradas"}

    from huggingface_hub import HfApi

    try:
        HfApi(token=token).super_squash_history(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Monthly telemetry compaction ({today.isoformat()}): squash commit history",
        )
        cursor["last_hf_squash_date"] = today.isoformat()
        _save_cursor(cursor_path, cursor)
        return {"squashed": True, "repo_id": repo_id, "date": today.isoformat()}
    except Exception as exc:
        return {"squashed": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Orquestación y CLI
# ---------------------------------------------------------------------------


def run_daily_audit_pipeline(
    force_pull: bool = False,
    force_all_audit: bool = False,
    export_candidates: bool = True,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de sincronización, archivo, auditoría y retención."""
    pull_res = HuggingFaceTelemetrySync().pull_latest_from_hf(force=force_pull)

    retention_res = apply_local_retention(db_path=SESSIONS_DB_PATH)

    archived_sessions = archive_session_transcripts(
        db_path=SESSIONS_DB_PATH,
        archive_dir=SESSION_ARCHIVE_DIR,
    )

    audit_res = extract_and_audit_deltas(
        db_path=SESSIONS_DB_PATH,
        cursor_path=AUDIT_CURSOR_PATH,
        reports_dir=AUDIT_REPORTS_DIR,
        force_all=force_all_audit,
        export_candidates=export_candidates,
    )

    squash_res = check_and_squash_hf_history(cursor_path=AUDIT_CURSOR_PATH)

    return {
        "timestamp": _now_local().isoformat(timespec="seconds"),
        "hf_sync": pull_res,
        "local_retention": retention_res,
        "sessions_archived": archived_sessions,
        "audit": audit_res,
        "hf_squash": squash_res,
    }


def _cli() -> None:
    # Salida UTF-8 en consolas Windows (cp1252) para que tildes y emojis no se corrompan
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Ejecutor de auditoría diaria de AiHoreca.")
    parser.add_argument("--force-pull", action="store_true", help="Fuerza descarga desde HF.")
    parser.add_argument(
        "--force-audit-all",
        action="store_true",
        help="Reaudita todos los turnos y sobrescribe el informe del día (regenera línea base).",
    )
    parser.add_argument(
        "--no-export-candidates",
        action="store_true",
        help="No añade los turnos marcados a data/logs/candidatos_golden.jsonl.",
    )
    parser.add_argument("--json", action="store_true", help="Imprime resultado en JSON.")
    args = parser.parse_args()

    result = run_daily_audit_pipeline(
        force_pull=args.force_pull,
        force_all_audit=args.force_audit_all,
        export_candidates=not args.no_export_candidates,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    hf = result["hf_sync"]
    audit = result["audit"]
    ret = result["local_retention"]
    sq = result["hf_squash"]
    print("[AiHoreca Audit Runner]")
    print(f"- Sincronización HF: {hf.get('reason') or ('Descargado SHA ' + str(hf.get('sha', ''))[:8])}")
    print(
        f"- Retención local: {ret['transcripts_deleted']} transcripciones, "
        f"{ret['reports_deleted']} informes, {ret['candidates_deleted']} candidatos eliminados"
    )
    print(f"- Sesiones archivadas: {result['sessions_archived']}")
    print(f"- Turnos nuevos: {audit.get('new_turns', 0)} | Votos tardíos: {audit.get('late_votes', 0)}")
    if audit.get("report_path"):
        print(f"- Informe: {audit['report_path']}")
        print(f"- Candidatos golden añadidos: {audit.get('golden_candidates_added', 0)}")
    print(f"- Compactación HF: {sq.get('reason') or ('Realizada' if sq.get('squashed') else sq.get('error', ''))}")


if __name__ == "__main__":
    _cli()
