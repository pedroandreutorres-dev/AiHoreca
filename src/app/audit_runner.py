"""Motor y ejecutor autónomo de auditoría diaria de telemetría y transcripciones de AiHoreca.

Flujo de ejecución:
1. Sincronización descendente (pull) desde el repositorio privado de Hugging Face Dataset.
2. Archivo permanente local de sesiones completas en `data/logs/sesiones/<fecha>_<session_id>.txt`.
3. Detección incremental basada en cursor (`data/logs/.sync_cursor.json`) de nuevos turnos.
4. Generación de informe diario de auditoría en `data/logs/auditorias/<fecha>_informe_auditoria.md`.
5. Compactación mensual programada (squash) del historial Git en Hugging Face Dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional

from src.app.telemetry_sync import HuggingFaceTelemetrySync, get_hf_telemetry_config
from src.config import (
    AUDIT_CURSOR_PATH,
    AUDIT_REPORTS_DIR,
    HF_SQUASH_INTERVAL_DAYS,
    SESSION_ARCHIVE_DIR,
    SESSIONS_DB_PATH,
)


def _load_cursor(cursor_path: Path) -> dict[str, Any]:
    if cursor_path.exists():
        try:
            return json.loads(cursor_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "last_audited_turn_id": 0,
        "last_audited_timestamp": "",
        "last_hf_squash_date": "",
        "updated_at": "",
    }


def _save_cursor(cursor_path: Path, data: dict[str, Any]) -> None:
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cursor_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def archive_session_transcripts(db_path: Path, archive_dir: Path) -> int:
    """Exporta y actualiza la transcripción completa y legible de cada sesión en disco."""
    if not db_path.exists():
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_count = 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at ASC"
        ).fetchall()

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

            thumbs_up = sum(1 for t in turns if t["user_feedback"] == "ME_GUSTA")
            thumbs_down = sum(1 for t in turns if t["user_feedback"] == "NO_ME_GUSTA")

            lines: list[str] = [
                "=" * 80,
                f"SESIÓN: {sid}",
                f"TÍTULO: {s['title']}",
                f"INICIO: {s['created_at']} | ÚLTIMA ACTIVIDAD: {s['updated_at']}",
                f"TERRITORIO: {s['active_territorio'] or 'No fijado'} | "
                f"OFICIO: {s['active_oficio_coloquial'] or 'No fijado'} | "
                f"CATEGORÍA ALEH: {s['active_categoria_aleh'] or 'No fijado'}",
                f"TOTAL TURNOS: {len(turns)} | FEEDBACK ACUMULADO: 👍 {thumbs_up} / 👎 {thumbs_down}",
                "=" * 80,
                "",
            ]

            for t in turns:
                audit = json.loads(t["audit_json"] or "{}")
                timings = json.loads(t["timings_json"] or "{}")
                models = json.loads(t["models_json"] or "{}")
                total_latency = round(sum(float(v) for v in timings.values()), 1)

                v_passed = "APROBADO" if audit.get("passed", True) else "CON INCIDENCIAS"
                v_count = len(audit.get("verified_numbers") or []) + len(audit.get("derived_numbers") or [])
                u_count = len(audit.get("unverified_numbers") or [])

                feedback_str = t["user_feedback"] or "SIN_VOTO"
                if t["feedback_at"]:
                    feedback_str += f" ({t['feedback_at']})"

                lines.extend([
                    f"[Turno {t['turn_index']} - {t['timestamp']}]",
                    f"Usuario: {t['user_query']}",
                    f"(Efectiva: {t['effective_query']} | Intención: {t['intent']} | Ruta: {t['route_decision']})",
                    "",
                    f"Asistente (Modelos: {models.get('legal_synthesis_and_audit', 'n/d')}):",
                    t["final_response"],
                    "",
                    f"Feedback Usuario: {feedback_str}",
                    f"Métricas Turno: Latencia {total_latency}ms | Auditoría Numérica: {v_passed} "
                    f"({v_count} verificados, {u_count} no verificados)",
                    "-" * 80,
                    "",
                ])

            out_file.write_text("\n".join(lines), encoding="utf-8")
            archived_count += 1
    finally:
        conn.close()

    return archived_count


def extract_and_audit_deltas(
    db_path: Path,
    cursor_path: Path,
    reports_dir: Path,
    force_all: bool = False,
) -> dict[str, Any]:
    """Detecta turnos nuevos desde la última auditoría y genera el informe estructurado."""
    if not db_path.exists():
        return {"new_turns": 0, "status": "db_not_found"}

    reports_dir.mkdir(parents=True, exist_ok=True)
    cursor = _load_cursor(cursor_path)
    last_turn_id = 0 if force_all else int(cursor.get("last_audited_turn_id", 0))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        new_turns = conn.execute(
            """
            SELECT t.*, s.title as session_title
            FROM session_turns t
            JOIN sessions s ON t.session_id = s.session_id
            WHERE t.id > ?
            ORDER BY t.id ASC
            """,
            (last_turn_id,),
        ).fetchall()

        if not new_turns:
            return {
                "new_turns": 0,
                "status": "up_to_date",
                "last_audited_turn_id": last_turn_id,
            }

        # Cálculo de métricas agregadas del lote auditado
        total_turns = len(new_turns)
        session_ids = {t["session_id"] for t in new_turns}
        max_id = max(t["id"] for t in new_turns)
        max_ts = max(t["timestamp"] for t in new_turns)

        intents_count: dict[str, int] = {}
        routes_count: dict[str, int] = {}
        territorios_count: dict[str, int] = {}
        feedback_count = {"ME_GUSTA": 0, "NO_ME_GUSTA": 0, "SIN_VOTO": 0}
        unverified_total = 0
        latencies_ms: list[float] = []

        flagged_turns: list[dict[str, Any]] = []

        for t in new_turns:
            intents_count[t["intent"]] = intents_count.get(t["intent"], 0) + 1
            routes_count[t["route_decision"]] = routes_count.get(t["route_decision"], 0) + 1
            terr = t["territorio"] or "No especificado"
            territorios_count[terr] = territorios_count.get(terr, 0) + 1

            fb = t["user_feedback"] or "SIN_VOTO"
            feedback_count[fb] = feedback_count.get(fb, 0) + 1

            audit = json.loads(t["audit_json"] or "{}")
            timings = json.loads(t["timings_json"] or "{}")
            tot_lat = round(sum(float(v) for v in timings.values()), 1)
            latencies_ms.append(tot_lat)

            unver = audit.get("unverified_numbers") or []
            unverified_total += len(unver)

            # Destacar turnos críticos para revisión humana: voto negativo o discrepancias numéricas
            if fb == "NO_ME_GUSTA" or unver or not audit.get("passed", True):
                flagged_turns.append({
                    "id": t["id"],
                    "session_id": t["session_id"],
                    "turn_index": t["turn_index"],
                    "query": t["user_query"],
                    "response": t["final_response"][:250] + ("..." if len(t["final_response"]) > 250 else ""),
                    "feedback": fb,
                    "unverified": unver,
                    "latency_ms": tot_lat,
                })

        avg_lat = round(sum(latencies_ms) / max(1, len(latencies_ms)), 1)
        latencies_sorted = sorted(latencies_ms)
        p95_lat = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0.0

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report_path = reports_dir / f"{today_str}_informe_auditoria.md"

        # Generar contenido Markdown del informe de auditoría
        md_lines = [
            f"# Informe Diario de Auditoría de Telemetría - AiHoreca",
            f"**Fecha de Auditoría**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Rango de Turnos Auditados**: IDs `{last_turn_id + 1}` a `{max_id}` ({total_turns} turnos nuevos)",
            f"**Sesiones Involucradas**: {len(session_ids)} sesiones distintas",
            "",
            "## 1. Resumen Ejecutivo y Satisfacción",
            f"- **Turnos procesados**: {total_turns}",
            f"- **Satisfacción de usuarios**: 👍 {feedback_count['ME_GUSTA']} | 👎 {feedback_count['NO_ME_GUSTA']} | Sin votar: {feedback_count['SIN_VOTO']}",
            f"- **Latencia media**: {avg_lat} ms (Percentil 95: {p95_lat} ms)",
            f"- **Discrepancias numéricas (Alucinaciones prevenidas)**: {unverified_total} números no verificados",
            "",
            "## 2. Distribución de Intenciones y Rutas",
            "| Dimensión | Conteo |",
            "|---|---|",
        ]
        for intent_name, count in sorted(intents_count.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"| **Intención**: `{intent_name}` | {count} |")
        for route_name, count in sorted(routes_count.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"| **Ruta**: `{route_name}` | {count} |")
        for terr_name, count in sorted(territorios_count.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"| **Territorio**: {terr_name} | {count} |")

        md_lines.extend([
            "",
            "## 3. Turnos Clave con Incidencias o Feedback Negativo",
        ])

        if not flagged_turns:
            md_lines.append("✅ **Ningún turno requirió revisión crítica**: 0 votos negativos y 100% de coherencia numérica.")
        else:
            for ft in flagged_turns:
                md_lines.extend([
                    f"### Sesión `{ft['session_id']}` (Turno {ft['turn_index']} - Turn ID {ft['id']})",
                    f"- **Consulta usuario**: \"{ft['query']}\"",
                    f"- **Feedback**: `{ft['feedback']}` | **Latencia**: {ft['latency_ms']} ms",
                    f"- **Números no verificados**: {ft['unverified'] if ft['unverified'] else 'Ninguno'}",
                    f"- **Extracto respuesta**: {ft['response']}",
                    "",
                ])

        report_path.write_text("\n".join(md_lines), encoding="utf-8")

        # Actualizar cursor
        cursor["last_audited_turn_id"] = max_id
        cursor["last_audited_timestamp"] = max_ts
        _save_cursor(cursor_path, cursor)

        return {
            "new_turns": total_turns,
            "status": "audited",
            "report_path": str(report_path),
            "flagged_turns": len(flagged_turns),
            "feedback": feedback_count,
            "avg_latency_ms": avg_lat,
        }
    finally:
        conn.close()


def check_and_squash_hf_history(cursor_path: Path) -> dict[str, Any]:
    """Comprueba si corresponde compactar el historial Git del repositorio remoto en Hugging Face."""
    cursor = _load_cursor(cursor_path)
    last_squash = cursor.get("last_hf_squash_date", "")

    today = datetime.now(timezone.utc).date()
    needs_squash = False

    if not last_squash:
        needs_squash = True
    else:
        try:
            last_date = datetime.strptime(last_squash, "%Y-%m-%d").date()
            if (today - last_date).days >= HF_SQUASH_INTERVAL_DAYS:
                needs_squash = True
        except Exception:
            needs_squash = True

    if not needs_squash:
        return {"squashed": False, "reason": f"Compactación al día (última: {last_squash})"}

    token, repo_id = get_hf_telemetry_config()
    if not token or not repo_id:
        return {"squashed": False, "reason": "Credenciales HF no configuradas"}

    from huggingface_hub import HfApi

    try:
        api = HfApi(token=token)
        api.super_squash_history(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Monthly telemetry compaction ({today.isoformat()}): squash commit history",
        )
        cursor["last_hf_squash_date"] = today.isoformat()
        _save_cursor(cursor_path, cursor)
        return {"squashed": True, "repo_id": repo_id, "date": today.isoformat()}
    except Exception as exc:
        return {"squashed": False, "error": str(exc)}


def run_daily_audit_pipeline(
    force_pull: bool = False,
    force_all_audit: bool = False,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de sincronización, archivo y auditoría."""
    syncer = HuggingFaceTelemetrySync()
    pull_res = syncer.pull_latest_from_hf(force=force_pull)

    archived_sessions = archive_session_transcripts(
        db_path=SESSIONS_DB_PATH,
        archive_dir=SESSION_ARCHIVE_DIR,
    )

    audit_res = extract_and_audit_deltas(
        db_path=SESSIONS_DB_PATH,
        cursor_path=AUDIT_CURSOR_PATH,
        reports_dir=AUDIT_REPORTS_DIR,
        force_all=force_all_audit,
    )

    squash_res = check_and_squash_hf_history(cursor_path=AUDIT_CURSOR_PATH)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hf_sync": pull_res,
        "sessions_archived": archived_sessions,
        "audit": audit_res,
        "hf_squash": squash_res,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ejecutor de auditoría diaria de AiHoreca.")
    parser.add_argument("--force-pull", action="store_true", help="Fuerza descarga desde HF.")
    parser.add_argument("--force-audit-all", action="store_true", help="Reaudita todos los turnos.")
    parser.add_argument("--json", action="store_true", help="Imprime resultado en JSON.")
    args = parser.parse_args()

    result = run_daily_audit_pipeline(
        force_pull=args.force_pull,
        force_all_audit=args.force_audit_all,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[AiHoreca Audit Runner]")
        print(f"- Sincronización HF: {result['hf_sync'].get('reason') or ('Descargado SHA ' + str(result['hf_sync'].get('sha', ''))[:8])}")
        print(f"- Sesiones archivadas en local: {result['sessions_archived']}")
        print(f"- Nuevos turnos auditados: {result['audit'].get('new_turns', 0)}")
        if result['audit'].get("report_path"):
            print(f"- Informe generado en: {result['audit']['report_path']}")
        print(f"- Compactación mensual HF: {result['hf_squash'].get('reason') or ('Realizada con éxito' if result['hf_squash'].get('squashed') else result['hf_squash'].get('error', ''))}")


if __name__ == "__main__":
    _cli()
