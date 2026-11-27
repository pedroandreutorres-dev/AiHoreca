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

from src.app.cost_dashboard import generate_cost_dashboard_html
from src.app.pricing import (
    PricingCatalog,
    TurnCostBreakdown,
    compute_turn_cost,
    load_pricing_catalog,
    resolve_model_rate,
)
from src.app.telemetry_sync import HuggingFaceTelemetrySync, get_hf_telemetry_config
from src.graph.memory import SessionMemoryStore
from src.graph.state import TurnUsageSummary
from src.config import (
    AUDIT_CURSOR_PATH,
    AUDIT_REPORTS_DIR,
    COST_DASHBOARD_PATH,
    HF_SQUASH_INTERVAL_DAYS,
    LOGS_DIR,
    PAID_TIER_FLASH_SYNTHESIS_MODEL,
    PAID_TIER_LITE_ROUTER_MODEL,
    PAID_TIER_LITE_SYNTHESIS_MODEL,
    SESSION_ARCHIVE_DIR,
    SESSION_COSTS_HISTORY_PATH,
    SESSION_RETENTION_DAYS,
    SESSIONS_DB_PATH,
    TURN_COSTS_HISTORY_PATH,
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
_UNSUPPORTED_PROV_SUMMARY_RE = re.compile(r"Rechazó consulta fuera de cobertura \(([^)]+)\)")


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
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
    row_keys = set(t.keys())
    audit = _parse_json_dict(t["audit_json"] if "audit_json" in row_keys else "{}")
    timings = _parse_json_dict(t["timings_json"] if "timings_json" in row_keys else "{}")
    models = _parse_json_dict(t["models_json"] if "models_json" in row_keys else "{}")
    usage_raw = _parse_json_dict(t["usage_json"] if "usage_json" in row_keys else "{}")
    try:
        usage = TurnUsageSummary.model_validate(usage_raw) if usage_raw else TurnUsageSummary()
    except Exception:
        usage = TurnUsageSummary()

    territorios_norm = usage_raw.get("territorios_normalizados") or []
    if not isinstance(territorios_norm, list):
        territorios_norm = []
    if not territorios_norm and t["territorio"]:
        territorios_norm = [p.strip() for p in str(t["territorio"]).split(",") if p.strip()]

    return {
        "audit": audit,
        "timings": timings,
        "models": models,
        "usage": usage,
        "has_usage_telemetry": bool(usage.steps or usage.total_tokens or usage.embedding_calls or usage.cohere_calls),
        "retrieval_hop": int(usage_raw.get("retrieval_hop") or 0),
        "audit_retry_count": int(usage_raw.get("audit_retry_count") or 0),
        "territorios_norm": territorios_norm,
        "provincia_detectada": usage_raw.get("provincia_detectada"),
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


def archive_session_transcripts(
    db_path: Path,
    archive_dir: Path,
    pricing_catalog: Optional[PricingCatalog] = None,
) -> int:
    """Exporta y actualiza la transcripción completa y legible de cada sesión en disco."""
    if not db_path.exists():
        return 0

    catalog = pricing_catalog or load_pricing_catalog(fetch_live=False)
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

            turn_costs: list[tuple[sqlite3.Row, dict[str, Any], TurnCostBreakdown]] = []
            sess_real_usd = 0.0
            sess_lite_usd = 0.0
            sess_flash_usd = 0.0
            for t in turns:
                m = _turn_metrics(t)
                tc = compute_turn_cost(m["usage"], catalog) if m["has_usage_telemetry"] else TurnCostBreakdown()
                sess_real_usd += tc.coste_total_real_usd
                sess_lite_usd += tc.coste_total_tier_lite_usd
                sess_flash_usd += tc.coste_total_tier_flash_usd
                turn_costs.append((t, m, tc))

            lines: list[str] = [
                "=" * 80,
                f"SESIÓN: {sid}",
                f"TÍTULO: {s['title']}",
                f"INICIO: {s['created_at']} | ÚLTIMA ACTIVIDAD: {s['updated_at']}",
                f"TERRITORIO: {s['active_territorio'] or 'No fijado'} | "
                f"OFICIO: {s['active_oficio_coloquial'] or 'No fijado'} | "
                f"CATEGORÍA ALEH: {s['active_categoria_aleh'] or 'No fijado'}",
                f"TOTAL TURNOS: {len(turns)} | FEEDBACK ACUMULADO: 👍 {n_ok} / 👎 {n_ko}",
                f"COSTE ESTIMADO API PAGO (fuente={catalog.source}): "
                f"Real ${sess_real_usd:.5f} | Tier Lite ${sess_lite_usd:.5f} | Tier Flash ${sess_flash_usd:.5f}",
                f"CADUCIDAD: se elimina {SESSION_RETENTION_DAYS} días después de la última actividad.",
                "=" * 80,
                "",
            ]

            for t, m, tc in turn_costs:
                models = m["models"]
                u: TurnUsageSummary = m["usage"]
                v_count = len(m["audit"].get("verified_numbers") or []) + len(
                    m["audit"].get("derived_numbers") or []
                )
                feedback_str = _feedback_icon(m["feedback"])
                if t["feedback_at"]:
                    feedback_str += f" ({t['feedback_at']})"

                turn_lines = [
                    f"[Turno {t['turn_index']} - {t['timestamp']}]",
                    f"Usuario: {t['user_query']}",
                    f"(Efectiva: {t['effective_query']} | Intención: {t['intent']} | Ruta: {t['route_decision']})",
                    "",
                    f"Asistente (Modelo síntesis: {models.get('synthesize_response') or models.get('legal_synthesis_and_audit', 'n/d')}):",
                    t["final_response"],
                    "",
                    f"Feedback usuario: {feedback_str}",
                    f"Métricas turno: Latencia {m['latency_ms']} ms | Auditoría numérica: "
                    f"{'APROBADO' if m['passed'] else 'CON INCIDENCIAS'} "
                    f"({v_count} verificados, {len(m['unverified'])} no verificados)",
                ]
                if m["has_usage_telemetry"]:
                    turn_lines.append(
                        f"Tokens turno: Input {u.input_tokens:,} | Output {u.output_tokens:,} | "
                        f"Thoughts {u.thoughts_tokens:,} | Cache {u.cached_tokens:,} | Total {u.total_tokens:,}"
                    )
                    turn_lines.append(
                        f"APIs turno: LLM {u.llm_calls} calls (failovers {u.llm_failovers}) | "
                        f"Embeddings {u.embedding_calls} calls ({u.embedding_cache_hits} cache_hit, {u.embedding_chars:,} chars) | "
                        f"Cohere {u.cohere_calls} calls ({u.cohere_search_units} search_units, {u.cohere_docs_reranked} docs) | "
                        f"B2 hops {m['retrieval_hop']} | A1 retries {m['audit_retry_count']}"
                    )
                    turn_lines.append(
                        f"Coste estimado turno (API pago): Real ${tc.coste_total_real_usd:.5f} | "
                        f"Tier Lite ${tc.coste_total_tier_lite_usd:.5f} | Tier Flash ${tc.coste_total_tier_flash_usd:.5f} "
                        f"(Gemini ${tc.coste_gemini_real_usd:.5f}, Emb ${tc.coste_embeddings_usd:.5f}, Cohere ${tc.coste_cohere_usd:.5f})"
                    )
                    if u.steps:
                        turn_lines.append("Detalle paso a paso:")
                        for s_idx, s in enumerate(u.steps, start=1):
                            err_part = f" | error={s.error_detail}" if s.error_detail else ""
                            turn_lines.append(
                                f"  {s_idx}. [{s.step}] {s.provider}/{s.model or '-'} ({s.status}): "
                                f"in={s.input_tokens:,}, out={s.output_tokens:,}, thoughts={s.thoughts_tokens:,}, "
                                f"cache={s.cached_tokens:,}, total={s.total_tokens:,}, api_calls={s.api_calls}, "
                                f"cache_hit={s.cache_hit}, chars={s.chars}, search_units={s.search_units}, "
                                f"docs={s.docs_reranked}, latency={s.latency_ms}ms{err_part}"
                            )
                else:
                    turn_lines.append("Tokens turno: sin telemetría detallada (turno previo a Bloque D)")

                turn_lines.extend(["-" * 80, ""])
                lines.extend(turn_lines)

            out_file.write_text("\n".join(lines), encoding="utf-8")
            archived_count += 1
    finally:
        conn.close()

    return archived_count


# ---------------------------------------------------------------------------
# 2. Auditoría incremental (turnos nuevos + votos tardíos)
# ---------------------------------------------------------------------------


def _append_metrics_history(row: dict[str, Any]) -> None:
    """Añade una fila a `historico_metricas.csv`, migrando la cabecera de forma atómica si hay columnas nuevas."""
    METRICS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_fields = list(row.keys())

    if not METRICS_HISTORY_PATH.exists():
        tmp_path = METRICS_HISTORY_PATH.with_suffix(".csv.tmp")
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=target_fields)
            writer.writeheader()
            writer.writerow(row)
        tmp_path.replace(METRICS_HISTORY_PATH)
        return

    with METRICS_HISTORY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        existing_rows = list(reader) if existing_fields != target_fields else []

    if existing_fields == target_fields:
        with METRICS_HISTORY_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=target_fields)
            writer.writerow(row)
        return

    unified_fields = list(target_fields)
    for col in existing_fields:
        if col not in unified_fields:
            unified_fields.append(col)

    tmp_path = METRICS_HISTORY_PATH.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unified_fields)
        writer.writeheader()
        for old_row in existing_rows:
            writer.writerow({k: old_row.get(k, "") for k in unified_fields})
        writer.writerow({k: row.get(k, "") for k in unified_fields})
    tmp_path.replace(METRICS_HISTORY_PATH)


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


def _upsert_csv_rows(
    csv_path: Path,
    key_col: str,
    new_rows: list[dict[str, Any]],
    sort_key_fn: Any = None,
) -> int:
    """Realiza upsert atómico por `key_col`, preservando registros históricos ya purgados de SQLite."""
    if not new_rows:
        return 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(new_rows[0].keys())
    existing_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for col in (reader.fieldnames or []):
                if col not in fieldnames:
                    fieldnames.append(col)
            for old_row in reader:
                k = str(old_row.get(key_col, ""))
                if k:
                    existing_by_key[k] = dict(old_row)
                    ordered_keys.append(k)

    for row in new_rows:
        k = str(row.get(key_col, ""))
        if not k:
            continue
        if k not in existing_by_key:
            ordered_keys.append(k)
        existing_by_key[k] = row

    merged = [existing_by_key[k] for k in ordered_keys if k in existing_by_key]
    if sort_key_fn is not None:
        merged.sort(key=sort_key_fn)

    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow({col: row.get(col, "") for col in fieldnames})
    tmp_path.replace(csv_path)
    return len(merged)


def export_cost_analytics_csvs(
    db_path: Path,
    pricing_catalog: Optional[PricingCatalog] = None,
    sessions_csv_path: Path = SESSION_COSTS_HISTORY_PATH,
    turns_csv_path: Path = TURN_COSTS_HISTORY_PATH,
) -> dict[str, Any]:
    """Exporta datasets permanentes sin PII (sesión a sesión y turno a turno) para analítica y monetización."""
    if not db_path.exists():
        return {"sessions_exported": 0, "turns_exported": 0, "pricing_source": "n/a"}

    catalog = pricing_catalog or load_pricing_catalog(fetch_live=False)
    r_lite_router = resolve_model_rate(catalog, PAID_TIER_LITE_ROUTER_MODEL)
    r_lite_synth = resolve_model_rate(catalog, PAID_TIER_LITE_SYNTHESIS_MODEL)
    r_flash_synth = resolve_model_rate(catalog, PAID_TIER_FLASH_SYNTHESIS_MODEL)
    r_emb = resolve_model_rate(catalog, "gemini-embedding-001")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    session_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    try:
        sessions = conn.execute("SELECT * FROM sessions ORDER BY created_at ASC").fetchall()
        turns_by_sid: dict[str, list[sqlite3.Row]] = {}
        for t in conn.execute("SELECT * FROM session_turns ORDER BY id ASC").fetchall():
            turns_by_sid.setdefault(t["session_id"], []).append(t)

        for s in sessions:
            sid = s["session_id"]
            s_turns = turns_by_sid.get(sid, [])
            if not s_turns:
                continue

            created = s["created_at"] or ""
            dt_created = _parse_iso(created)
            weekday_names = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
            dia_semana = weekday_names[dt_created.weekday()] if dt_created else ""

            tot_rag = 0
            tot_sc = 0
            tot_telem = 0
            ok_cnt = ko_cnt = none_cnt = unv_cnt = 0
            lat_sum = 0.0
            in_tok = out_tok = th_tok = cache_tok = all_tok = 0
            llm_calls = llm_fails = emb_calls = emb_hits = emb_chars = emb_tok_est = cohere_calls = cohere_units = 0
            b2_sum = a1_sum = 0
            c_gem_real = c_emb = c_coh = c_real_usd = c_lite_usd = c_flash_usd = 0.0

            for t in s_turns:
                m = _turn_metrics(t)
                u: TurnUsageSummary = m["usage"]
                tc = compute_turn_cost(u, catalog) if m["has_usage_telemetry"] else TurnCostBreakdown()

                route = t["route_decision"] or ""
                if route == "parallel_retrieve":
                    tot_rag += 1
                elif route in ("block_out_of_scope", "ask_clarification"):
                    tot_sc += 1
                if m["has_usage_telemetry"]:
                    tot_telem += 1

                fb = m["feedback"]
                if fb == FEEDBACK_POSITIVE:
                    ok_cnt += 1
                elif fb == FEEDBACK_NEGATIVE:
                    ko_cnt += 1
                else:
                    none_cnt += 1
                unv_cnt += len(m["unverified"])
                lat_sum += m["latency_ms"]

                in_tok += u.input_tokens
                out_tok += u.output_tokens
                th_tok += u.thoughts_tokens
                cache_tok += u.cached_tokens
                all_tok += u.total_tokens
                llm_calls += u.llm_calls
                llm_fails += u.llm_failovers
                emb_calls += u.embedding_calls
                emb_hits += u.embedding_cache_hits
                emb_chars += u.embedding_chars
                emb_tok_est += tc.embedding_tokens_est
                cohere_calls += u.cohere_calls
                cohere_units += u.cohere_search_units
                b2_sum += m["retrieval_hop"]
                a1_sum += m["audit_retry_count"]

                c_gem_real += tc.coste_gemini_real_usd
                c_emb += tc.coste_embeddings_usd
                c_coh += tc.coste_cohere_usd
                c_real_usd += tc.coste_total_real_usd
                c_lite_usd += tc.coste_total_tier_lite_usd
                c_flash_usd += tc.coste_total_tier_flash_usd

                ts_turn = str(t["timestamp"] or "")
                turn_rows.append({
                    "turn_id": int(t["id"]),
                    "session_id": sid,
                    "turn_index": int(t["turn_index"]),
                    "fecha": ts_turn[:10],
                    "hora": ts_turn[11:19] if len(ts_turn) >= 19 else "",
                    "intent": t["intent"] or "",
                    "route_decision": route,
                    "territorio": t["territorio"] or "",
                    "oficio_detectado": t["oficio_detectado"] or "",
                    "modelo_router": m["models"].get("analyze_and_route", ""),
                    "modelo_sintesis": m["models"].get("synthesize_response") or m["models"].get("legal_synthesis_and_audit", ""),
                    "audit_passed": int(m["passed"]),
                    "num_unverified": len(m["unverified"]),
                    "user_feedback": fb,
                    "latencia_ms": m["latency_ms"],
                    "has_telemetry": int(m["has_usage_telemetry"]),
                    "b2_hops": m["retrieval_hop"],
                    "a1_retries": m["audit_retry_count"],
                    "tokens_entrada": u.input_tokens,
                    "tokens_salida": u.output_tokens,
                    "tokens_thoughts": u.thoughts_tokens,
                    "tokens_cached": u.cached_tokens,
                    "tokens_total": u.total_tokens,
                    "llm_llamadas": u.llm_calls,
                    "llm_failovers": u.llm_failovers,
                    "embeddings_llamadas": u.embedding_calls,
                    "embeddings_cache_hits": u.embedding_cache_hits,
                    "embeddings_chars": u.embedding_chars,
                    "embeddings_tokens_est": tc.embedding_tokens_est,
                    "cohere_llamadas": u.cohere_calls,
                    "cohere_search_units": u.cohere_search_units,
                    "coste_gemini_real_usd": tc.coste_gemini_real_usd,
                    "coste_embeddings_usd": tc.coste_embeddings_usd,
                    "coste_cohere_usd": tc.coste_cohere_usd,
                    "coste_total_real_usd": tc.coste_total_real_usd,
                    "coste_total_real_eur": tc.coste_total_real_eur,
                    "coste_total_tier_lite_usd": tc.coste_total_tier_lite_usd,
                    "coste_total_tier_lite_eur": tc.coste_total_tier_lite_eur,
                    "coste_total_tier_flash_usd": tc.coste_total_tier_flash_usd,
                    "coste_total_tier_flash_eur": tc.coste_total_tier_flash_eur,
                    "pricing_source": catalog.source,
                })

            voted = ok_cnt + ko_cnt
            sat_pct = round(100.0 * ok_cnt / voted, 1) if voted else ""
            n_t = len(s_turns)
            session_rows.append({
                "session_id": sid,
                "fecha_inicio": created[:10],
                "hora_inicio": created[11:19] if len(created) >= 19 else "",
                "dia_semana": dia_semana,
                "ultima_actividad": s["updated_at"] or "",
                "territorio": s["active_territorio"] or "",
                "oficio": s["active_oficio_coloquial"] or "",
                "categoria_aleh": s["active_categoria_aleh"] or "",
                "total_turnos": n_t,
                "turnos_rag": tot_rag,
                "turnos_cortocircuito": tot_sc,
                "turnos_con_telemetria": tot_telem,
                "votos_ok": ok_cnt,
                "votos_ko": ko_cnt,
                "sin_voto": none_cnt,
                "satisfaccion_pct": sat_pct,
                "cifras_no_verificadas_total": unv_cnt,
                "latencia_total_ms": round(lat_sum, 1),
                "latencia_media_ms": round(lat_sum / n_t, 1) if n_t else 0.0,
                "tokens_entrada_total": in_tok,
                "tokens_salida_total": out_tok,
                "tokens_thoughts_total": th_tok,
                "tokens_cached_total": cache_tok,
                "tokens_total": all_tok,
                "llm_llamadas": llm_calls,
                "llm_failovers": llm_fails,
                "embeddings_llamadas": emb_calls,
                "embeddings_cache_hits": emb_hits,
                "embeddings_chars": emb_chars,
                "embeddings_tokens_est": emb_tok_est,
                "cohere_llamadas": cohere_calls,
                "cohere_search_units": cohere_units,
                "b2_saltos": b2_sum,
                "a1_reintentos": a1_sum,
                "coste_gemini_real_usd": round(c_gem_real, 6),
                "coste_embeddings_usd": round(c_emb, 6),
                "coste_cohere_usd": round(c_coh, 6),
                "coste_total_real_usd": round(c_real_usd, 6),
                "coste_total_real_eur": round(c_real_usd * catalog.usd_to_eur, 6),
                "coste_total_tier_lite_usd": round(c_lite_usd, 6),
                "coste_total_tier_lite_eur": round(c_lite_usd * catalog.usd_to_eur, 6),
                "coste_total_tier_flash_usd": round(c_flash_usd, 6),
                "coste_total_tier_flash_eur": round(c_flash_usd * catalog.usd_to_eur, 6),
                "coste_medio_turno_lite_usd": round(c_lite_usd / tot_telem, 6) if tot_telem else 0.0,
                "coste_medio_turno_flash_usd": round(c_flash_usd / tot_telem, 6) if tot_telem else 0.0,
                "tarifa_lite_router_in_1m": r_lite_router.input_per_1m if r_lite_router else "",
                "tarifa_lite_router_out_1m": r_lite_router.output_per_1m if r_lite_router else "",
                "tarifa_lite_synth_in_1m": r_lite_synth.input_per_1m if r_lite_synth else "",
                "tarifa_lite_synth_out_1m": r_lite_synth.output_per_1m if r_lite_synth else "",
                "tarifa_flash_synth_in_1m": r_flash_synth.input_per_1m if r_flash_synth else "",
                "tarifa_flash_synth_out_1m": r_flash_synth.output_per_1m if r_flash_synth else "",
                "tarifa_embedding_1m": r_emb.input_per_1m if r_emb else "",
                "tarifa_cohere_1k_units": catalog.cohere_usd_per_1k_units,
                "tipo_cambio_usd_eur": catalog.usd_to_eur,
                "pricing_source": catalog.source,
                "pricing_checked_at": catalog.checked_at,
            })
    finally:
        conn.close()

    total_sess = _upsert_csv_rows(
        sessions_csv_path,
        "session_id",
        session_rows,
        sort_key_fn=lambda r: (str(r.get("fecha_inicio", "")), str(r.get("hora_inicio", "")), str(r.get("session_id", ""))),
    )
    total_turns = _upsert_csv_rows(
        turns_csv_path,
        "turn_id",
        turn_rows,
        sort_key_fn=lambda r: int(r.get("turn_id") or 0),
    )
    return {
        "sessions_exported": len(session_rows),
        "sessions_total_csv": total_sess,
        "turns_exported": len(turn_rows),
        "turns_total_csv": total_turns,
        "sessions_csv_path": str(sessions_csv_path),
        "turns_csv_path": str(turns_csv_path),
        "pricing_source": catalog.source,
    }


def extract_and_audit_deltas(
    db_path: Path,
    cursor_path: Path,
    reports_dir: Path,
    force_all: bool = False,
    export_candidates: bool = False,
    pricing_catalog: Optional[PricingCatalog] = None,
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
    catalog = pricing_catalog or load_pricing_catalog(fetch_live=True)
    feedback_count = {FEEDBACK_POSITIVE: 0, FEEDBACK_NEGATIVE: 0, FEEDBACK_NONE: 0}
    intents_count: dict[str, int] = {}
    routes_count: dict[str, int] = {}
    territorios_count: dict[str, int] = {}
    unsupported_prov_counts: dict[str, int] = {}
    latencies_ms: list[float] = []
    unverified_total = 0
    flagged: list[dict[str, Any]] = []

    # Acumuladores de tokens, llamadas API, failovers, eficiencia del grafo y costes estimados
    total_input_tokens = 0
    total_output_tokens = 0
    total_thoughts_tokens = 0
    total_cached_tokens = 0
    total_tokens = 0
    total_llm_calls = 0
    total_llm_failovers = 0
    total_429 = 0
    total_503_504 = 0
    total_truncated = 0
    total_other_failovers = 0
    total_embedding_calls = 0
    total_embedding_cache_hits = 0
    total_embedding_chars = 0
    total_cohere_calls = 0
    total_cohere_search_units = 0
    total_cohere_docs_reranked = 0

    total_cost_gemini_real_usd = 0.0
    total_cost_embeddings_usd = 0.0
    total_cost_cohere_usd = 0.0
    total_cost_real_usd = 0.0
    total_cost_tier_lite_usd = 0.0
    total_cost_tier_flash_usd = 0.0
    session_batch_costs: dict[str, dict[str, Any]] = {}

    turn_tokens_with_usage: list[int] = []
    rag_tokens_with_usage: list[int] = []
    model_stats: dict[tuple[str, str], dict[str, Any]] = {}
    stage_stats: dict[str, dict[str, Any]] = {}
    per_turn_usage_rows: list[tuple[sqlite3.Row, dict[str, Any]]] = []

    short_circuits_count = 0
    rag_turns_count = 0
    a1_retry_turns = 0
    a1_recovered_turns = 0
    b2_hop_turns = 0
    multi_territory_turns = 0

    def _flag_entry(t: sqlite3.Row, m: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
        u_obj: TurnUsageSummary = m["usage"]
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
            "total_tokens": u_obj.total_tokens,
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
        u: TurnUsageSummary = m["usage"]
        intents_count[t["intent"]] = intents_count.get(t["intent"], 0) + 1
        routes_count[t["route_decision"]] = routes_count.get(t["route_decision"], 0) + 1
        terr = t["territorio"] or "No especificado"
        territorios_count[terr] = territorios_count.get(terr, 0) + 1
        feedback_count[m["feedback"]] = feedback_count.get(m["feedback"], 0) + 1
        latencies_ms.append(m["latency_ms"])
        unverified_total += len(m["unverified"])

        if t["intent"] == "UNSUPPORTED_PROVINCE":
            prov_req = m.get("provincia_detectada")
            if not prov_req and t["assistant_summary"]:
                pm = _UNSUPPORTED_PROV_SUMMARY_RE.search(t["assistant_summary"])
                if pm:
                    prov_req = pm.group(1)
            prov_label = (prov_req or "Desconocida").strip()
            unsupported_prov_counts[prov_label] = unsupported_prov_counts.get(prov_label, 0) + 1

        if t["route_decision"] in ("block_out_of_scope", "ask_clarification"):
            short_circuits_count += 1
        elif t["route_decision"] == "parallel_retrieve":
            rag_turns_count += 1
        if m["audit_retry_count"] > 0:
            a1_retry_turns += 1
            if m["passed"]:
                a1_recovered_turns += 1
        if m["retrieval_hop"] > 0:
            b2_hop_turns += 1
        if len(m["territorios_norm"]) > 1:
            multi_territory_turns += 1

        total_input_tokens += u.input_tokens
        total_output_tokens += u.output_tokens
        total_thoughts_tokens += u.thoughts_tokens
        total_cached_tokens += u.cached_tokens
        total_tokens += u.total_tokens
        total_llm_calls += u.llm_calls
        total_llm_failovers += u.llm_failovers
        total_429 += u.rate_limit_429_count
        total_503_504 += u.server_503_504_count
        total_truncated += u.truncated_retries
        total_other_failovers += max(
            0,
            u.llm_failovers - u.rate_limit_429_count - u.server_503_504_count - u.truncated_retries,
        )
        total_embedding_calls += u.embedding_calls
        total_embedding_cache_hits += u.embedding_cache_hits
        total_embedding_chars += u.embedding_chars
        total_cohere_calls += u.cohere_calls
        total_cohere_search_units += u.cohere_search_units
        total_cohere_docs_reranked += u.cohere_docs_reranked

        if m["has_usage_telemetry"]:
            turn_tokens_with_usage.append(u.total_tokens)
            if t["route_decision"] == "parallel_retrieve":
                rag_tokens_with_usage.append(u.total_tokens)

        tc = compute_turn_cost(u, catalog) if m["has_usage_telemetry"] else TurnCostBreakdown()
        total_cost_gemini_real_usd += tc.coste_gemini_real_usd
        total_cost_embeddings_usd += tc.coste_embeddings_usd
        total_cost_cohere_usd += tc.coste_cohere_usd
        total_cost_real_usd += tc.coste_total_real_usd
        total_cost_tier_lite_usd += tc.coste_total_tier_lite_usd
        total_cost_tier_flash_usd += tc.coste_total_tier_flash_usd

        sb = session_batch_costs.setdefault(
            t["session_id"],
            {
                "turnos": 0,
                "tokens": 0,
                "cohere_units": 0,
                "real_usd": 0.0,
                "lite_usd": 0.0,
                "flash_usd": 0.0,
            },
        )
        sb["turnos"] += 1
        sb["tokens"] += u.total_tokens
        sb["cohere_units"] += u.cohere_search_units
        sb["real_usd"] += tc.coste_total_real_usd
        sb["lite_usd"] += tc.coste_total_tier_lite_usd
        sb["flash_usd"] += tc.coste_total_tier_flash_usd

        for step in u.steps:
            mkey = (step.provider or "-", step.model or "-")
            ms = model_stats.setdefault(
                mkey,
                {
                    "api_calls": 0,
                    "ok": 0,
                    "cache_hits": 0,
                    "fail_429": 0,
                    "fail_503_504": 0,
                    "fail_trunc": 0,
                    "fail_other": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thoughts_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "chars": 0,
                    "search_units": 0,
                    "docs_reranked": 0,
                },
            )
            ms["api_calls"] += step.api_calls
            if step.status == "ok":
                ms["ok"] += 1
            elif step.status == "cache_hit" or step.cache_hit:
                ms["cache_hits"] += 1
            elif step.status == "failed_429":
                ms["fail_429"] += 1
            elif step.status == "failed_503_504":
                ms["fail_503_504"] += 1
            elif step.status == "failed_truncated":
                ms["fail_trunc"] += 1
            else:
                ms["fail_other"] += 1
            ms["input_tokens"] += step.input_tokens
            ms["output_tokens"] += step.output_tokens
            ms["thoughts_tokens"] += step.thoughts_tokens
            ms["cached_tokens"] += step.cached_tokens
            ms["total_tokens"] += step.total_tokens
            ms["chars"] += step.chars
            ms["search_units"] += step.search_units
            ms["docs_reranked"] += step.docs_reranked

            st_stat = stage_stats.setdefault(
                step.step or "desconocido",
                {
                    "events": 0,
                    "api_calls": 0,
                    "cache_hits": 0,
                    "errors": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_tokens": 0,
                    "latencies": [],
                },
            )
            st_stat["events"] += 1
            st_stat["api_calls"] += step.api_calls
            if step.cache_hit or step.status == "cache_hit":
                st_stat["cache_hits"] += 1
            if step.status.startswith("failed_"):
                st_stat["errors"] += 1
            st_stat["input_tokens"] += step.input_tokens
            st_stat["output_tokens"] += step.output_tokens
            st_stat["thoughts_tokens"] += step.thoughts_tokens
            st_stat["total_tokens"] += step.total_tokens
            if step.latency_ms > 0:
                st_stat["latencies"].append(step.latency_ms)

        per_turn_usage_rows.append((t, m))

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

    avg_tokens_per_turn = (
        round(sum(turn_tokens_with_usage) / len(turn_tokens_with_usage), 1)
        if turn_tokens_with_usage
        else 0.0
    )
    avg_tokens_rag = (
        round(sum(rag_tokens_with_usage) / len(rag_tokens_with_usage), 1)
        if rag_tokens_with_usage
        else 0.0
    )
    tok_sorted = sorted(turn_tokens_with_usage)
    p95_tokens_per_turn = (
        tok_sorted[min(len(tok_sorted) - 1, int(len(tok_sorted) * 0.95))]
        if tok_sorted
        else 0
    )

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
        f"- **Consumo de tokens (server `usage_metadata`)**: Entrada `{total_input_tokens:,}` | "
        f"Salida `{total_output_tokens:,}` | Thoughts `{total_thoughts_tokens:,}` | "
        f"Cache `{total_cached_tokens:,}` | **Total `{total_tokens:,}`**",
        f"- **Promedio / P95 por turno con telemetría**: media `{avg_tokens_per_turn:,.1f}` tok/turno | "
        f"media turnos RAG `{avg_tokens_rag:,.1f}` tok/turno | P95 `{p95_tokens_per_turn:,}` tok/turno",
        f"- **Llamadas API reales**: LLM `{total_llm_calls}` (failovers `{total_llm_failovers}`: "
        f"429=`{total_429}`, 503/504=`{total_503_504}`, truncados=`{total_truncated}`, otros=`{total_other_failovers}`) | "
        f"Embeddings `{total_embedding_calls}` calls (`{total_embedding_cache_hits}` cache_hit, `{total_embedding_chars:,}` chars) | "
        f"Cohere `{total_cohere_calls}` calls (`{total_cohere_search_units}` search_units, `{total_cohere_docs_reranked}` docs)",
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

        embed_total_reqs = total_embedding_calls + total_embedding_cache_hits
        embed_hit_pct = round(100 * total_embedding_cache_hits / embed_total_reqs, 1) if embed_total_reqs else 0.0
        short_pct = round(100 * short_circuits_count / n_new, 1)
        rag_pct = round(100 * rag_turns_count / n_new, 1)
        md.extend([
            "### Eficiencia del grafo y fiabilidad API",
            "",
            f"- **Cortocircuitos deterministas** (`block_out_of_scope` + `ask_clarification`): {short_circuits_count}/{n_new} ({short_pct}%) | **RAG completo**: {rag_turns_count}/{n_new} ({rag_pct}%)",
            f"- **Guardrail A1 (autocorrección)**: {a1_retry_turns} turnos con reintento ({a1_recovered_turns} recuperados con auditoría aprobada)",
            f"- **Expansión B2 (2º salto por remisiones)**: {b2_hop_turns} turnos con salto dirigido | **Consultas multi-territorio**: {multi_territory_turns}",
            f"- **Caché single-flight de embeddings**: {total_embedding_cache_hits}/{embed_total_reqs} hits ({embed_hit_pct}% ahorro de llamadas `gemini-embedding-001`)",
            "",
        ])

        if model_stats:
            md.extend([
                "### Consumo por modelo y proveedor",
                "",
                "| Proveedor | Modelo | API Calls | OK | Cache Hit | Fallos (429 / 503-504 / Trunc / Otros) | Entrada (tok) | Salida (tok) | Thoughts (tok) | Cache (tok) | Total (tok) | Extra (Chars / Search Units) |",
                "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
            ])
            for (prov, mod_name), st_m in sorted(
                model_stats.items(),
                key=lambda x: (-x[1]["total_tokens"], -x[1]["api_calls"], x[0][0], x[0][1]),
            ):
                fails_str = f"{st_m['fail_429']} / {st_m['fail_503_504']} / {st_m['fail_trunc']} / {st_m['fail_other']}"
                extra_parts: list[str] = []
                if st_m["chars"]:
                    extra_parts.append(f"{st_m['chars']:,} chars")
                if st_m["search_units"]:
                    extra_parts.append(f"{st_m['search_units']} units ({st_m['docs_reranked']} docs)")
                extra_str = ", ".join(extra_parts) if extra_parts else "—"
                md.append(
                    f"| `{prov}` | `{mod_name}` | {st_m['api_calls']} | {st_m['ok']} | {st_m['cache_hits']} | "
                    f"{fails_str} | {st_m['input_tokens']:,} | {st_m['output_tokens']:,} | "
                    f"{st_m['thoughts_tokens']:,} | {st_m['cached_tokens']:,} | {st_m['total_tokens']:,} | {extra_str} |"
                )
            md.append("")

        if stage_stats:
            md.extend([
                "### Consumo por etapa / paso del grafo",
                "",
                "| Etapa (`step`) | Eventos | API Calls | Cache Hits | Errores | Entrada (tok) | Salida (tok) | Thoughts (tok) | Total (tok) | Latencia media (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for step_key, st_s in sorted(
                stage_stats.items(),
                key=lambda x: (-x[1]["total_tokens"], -x[1]["api_calls"], x[0]),
            ):
                avg_step_ms = (
                    round(sum(st_s["latencies"]) / len(st_s["latencies"]), 1)
                    if st_s["latencies"]
                    else 0.0
                )
                md.append(
                    f"| `{step_key}` | {st_s['events']} | {st_s['api_calls']} | {st_s['cache_hits']} | "
                    f"{st_s['errors']} | {st_s['input_tokens']:,} | {st_s['output_tokens']:,} | "
                    f"{st_s['thoughts_tokens']:,} | {st_s['total_tokens']:,} | {avg_step_ms} |"
                )
            md.append("")

        r_lr = resolve_model_rate(catalog, PAID_TIER_LITE_ROUTER_MODEL)
        r_ls = resolve_model_rate(catalog, PAID_TIER_LITE_SYNTHESIS_MODEL)
        r_fs = resolve_model_rate(catalog, PAID_TIER_FLASH_SYNTHESIS_MODEL)
        r_em = resolve_model_rate(catalog, "gemini-embedding-001")
        n_telem = len(turn_tokens_with_usage)
        avg_lite_turn = (total_cost_tier_lite_usd / n_telem) if n_telem else 0.0
        avg_flash_turn = (total_cost_tier_flash_usd / n_telem) if n_telem else 0.0
        n_sess_batch = max(1, len(session_batch_costs))
        avg_lite_sess = total_cost_tier_lite_usd / n_sess_batch
        avg_flash_sess = total_cost_tier_flash_usd / n_sess_batch
        md.extend([
            "### Estimación de costes en API de pago (Unit Economics)",
            "",
            f"- **Fuente tarifas**: `{catalog.source}` (vigencia `{catalog.as_of_date}`, cambio `1 USD = {catalog.usd_to_eur} EUR`)",
            f"- **Tarifas comprobadas (1M tok)**: Router `{PAID_TIER_LITE_ROUTER_MODEL}` (`${r_lr.input_per_1m if r_lr else 0:.2f}` in / `${r_lr.output_per_1m if r_lr else 0:.2f}` out) · "
            f"Síntesis Lite `{PAID_TIER_LITE_SYNTHESIS_MODEL}` (`${r_ls.input_per_1m if r_ls else 0:.2f}` in / `${r_ls.output_per_1m if r_ls else 0:.2f}` out) · "
            f"Síntesis Flash `{PAID_TIER_FLASH_SYNTHESIS_MODEL}` (`${r_fs.input_per_1m if r_fs else 0:.2f}` in / `${r_fs.output_per_1m if r_fs else 0:.2f}` out) · "
            f"Embeddings (`${r_em.input_per_1m if r_em else 0.15:.2f}`) · Cohere Rerank (`${catalog.cohere_usd_per_1k_units:.2f}` / 1k units)",
            f"- **Coste total del lote (real modelos ejecutados)**: **`${total_cost_real_usd:.5f} USD`** (`{total_cost_real_usd * catalog.usd_to_eur:.5f} €`) "
            f"— Gemini LLM `${total_cost_gemini_real_usd:.5f}` | Embeddings `${total_cost_embeddings_usd:.5f}` | Cohere `${total_cost_cohere_usd:.5f}`",
            f"- **Simulación Stack Tier Lite (`3.1-flash-lite` + `3.5-flash-lite`)**: **`${total_cost_tier_lite_usd:.5f} USD`** (`{total_cost_tier_lite_usd * catalog.usd_to_eur:.5f} €`) "
            f"· media `{avg_lite_turn:.5f} USD/turno` (`${avg_lite_sess * 1000:.2f} / 1.000 conv.` similares)",
            f"- **Simulación Stack Tier Flash (`3.1-flash-lite` + `3.8-flash`)**: **`${total_cost_tier_flash_usd:.5f} USD`** (`{total_cost_tier_flash_usd * catalog.usd_to_eur:.5f} €`) "
            f"· media `{avg_flash_turn:.5f} USD/turno` (`${avg_flash_sess * 1000:.2f} / 1.000 conv.` similares)",
            "",
            "| Sesión | Turnos | Tokens | Cohere Units | Real (USD) | Tier Lite (USD / EUR) | Tier Flash (USD / EUR) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for s_id, sb in sorted(session_batch_costs.items(), key=lambda x: -x[1]["lite_usd"]):
            md.append(
                f"| `{s_id}` | {sb['turnos']} | {sb['tokens']:,} | {sb['cohere_units']} | "
                f"${sb['real_usd']:.5f} | ${sb['lite_usd']:.5f} / €{sb['lite_usd'] * catalog.usd_to_eur:.5f} | "
                f"${sb['flash_usd']:.5f} / €{sb['flash_usd'] * catalog.usd_to_eur:.5f} |"
            )
        md.append("")
        radar = catalog.radar
        if radar.needs_attention:
            md.append("**Alertas del radar oficial de modelos y tarifas:**")
            if radar.price_changes:
                md.append(f"- Cambios de tarifa detectados: `{', '.join(radar.price_changes)}`")
            if radar.deprecated_configured_models:
                md.append(f"- Modelos en `config.py` deprecados o ausentes: `{', '.join(radar.deprecated_configured_models)}`")
            if radar.upcoming_price_changes:
                md.append(f"- Subidas programadas próximas: `{'; '.join(radar.upcoming_price_changes)}`")
            if radar.new_models:
                md.append(f"- Modelos nuevos en doc oficial: `{', '.join(radar.new_models)}`")
            if radar.removed_models:
                md.append(f"- Modelos retirados del catálogo oficial: `{', '.join(radar.removed_models)}`")
            md.append("")

        if turn_tokens_with_usage:
            top_turns = sorted(
                ((t, m) for t, m in per_turn_usage_rows if m["has_usage_telemetry"]),
                key=lambda pair: -pair[1]["usage"].total_tokens,
            )[:5]
            md.extend([
                "### Top 5 consultas por consumo total de tokens",
                "",
                "| Turn ID | Sesión | Turno | Ruta | Entrada (tok) | Salida (tok) | Total (tok) | LLM Calls | Consulta |",
                "|---:|---|---:|---|---:|---:|---:|---:|---|",
            ])
            for top_t, top_m in top_turns:
                tu = top_m["usage"]
                q_prev = top_t["user_query"].replace("|", "/").replace("\n", " ")[:75]
                md.append(
                    f"| {top_t['id']} | `{top_t['session_id']}` | {top_t['turn_index']} | `{top_t['route_decision']}` | "
                    f"{tu.input_tokens:,} | {tu.output_tokens:,} | {tu.total_tokens:,} | {tu.llm_calls} | {q_prev} |"
                )
            md.append("")

        if unsupported_prov_counts:
            md.extend([
                "### Demanda no cubierta (`UNSUPPORTED_PROVINCE`)",
                "",
                "| Provincia / territorio solicitado | Consultas |",
                "|---|---:|",
            ])
            for prov_name, cnt in sorted(unsupported_prov_counts.items(), key=lambda x: (-x[1], x[0])):
                md.append(f"| {prov_name} | {cnt} |")
            md.append("")

        md.extend(["### Desglose detallado de tokens paso a paso por consulta", ""])
        for row_t, row_m in per_turn_usage_rows:
            u_t: TurnUsageSummary = row_m["usage"]
            q_prev = row_t["user_query"].replace("\n", " ").strip()
            if len(q_prev) > 140:
                q_prev = q_prev[:137] + "…"
            md.append(
                f"#### Turno {row_t['turn_index']} · `turn_id={row_t['id']}` · sesión `{row_t['session_id']}` "
                f"(`{row_t['intent']}` → `{row_t['route_decision']}`)"
            )
            md.append(f"- **Pregunta**: \"{q_prev}\"")
            if row_m["has_usage_telemetry"]:
                md.append(
                    f"- **Totales del turno**: Entrada `{u_t.input_tokens:,}` | Salida `{u_t.output_tokens:,}` | "
                    f"Thoughts `{u_t.thoughts_tokens:,}` | Cache `{u_t.cached_tokens:,}` | "
                    f"**Total `{u_t.total_tokens:,}`** | LLM `{u_t.llm_calls}` calls (failovers `{u_t.llm_failovers}`) | "
                    f"Embeddings `{u_t.embedding_calls}` (`{u_t.embedding_cache_hits}` cache_hit, `{u_t.embedding_chars:,}` chars) | "
                    f"Cohere `{u_t.cohere_calls}` (`{u_t.cohere_search_units}` search_units, `{u_t.cohere_docs_reranked}` docs) | "
                    f"B2 hops `{row_m['retrieval_hop']}` | A1 retries `{row_m['audit_retry_count']}` | Latencia `{row_m['latency_ms']} ms`"
                )
                if u_t.steps:
                    md.extend([
                        "",
                        "| # | Paso | Proveedor | Modelo | Estado | Entrada (tok) | Salida (tok) | Thoughts (tok) | Cache (tok) | Total (tok) | Calls / Hit / Extra | Latencia (ms) |",
                        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---:|",
                    ])
                    for st_idx, st_ev in enumerate(u_t.steps, start=1):
                        extra_items = [f"calls={st_ev.api_calls}"]
                        if st_ev.cache_hit:
                            extra_items.append("cache_hit=True")
                        if st_ev.chars:
                            extra_items.append(f"{st_ev.chars:,} chars")
                        if st_ev.search_units:
                            extra_items.append(f"{st_ev.search_units} units/{st_ev.docs_reranked} docs")
                        if st_ev.attempt > 1:
                            extra_items.append(f"intento={st_ev.attempt}")
                        if st_ev.error_detail:
                            err_clean = st_ev.error_detail.replace("|", "/").replace("\n", " ")[:45]
                            extra_items.append(f"err={err_clean}")
                        md.append(
                            f"| {st_idx} | `{st_ev.step}` | `{st_ev.provider}` | `{st_ev.model or '-'}` | `{st_ev.status}` | "
                            f"{st_ev.input_tokens:,} | {st_ev.output_tokens:,} | {st_ev.thoughts_tokens:,} | "
                            f"{st_ev.cached_tokens:,} | {st_ev.total_tokens:,} | {'; '.join(extra_items)} | {st_ev.latency_ms} |"
                        )
            else:
                md.append("- **Totales del turno**: sin telemetría de tokens registrada (turno previo a Bloque D).")
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
                f"**Ruta**: `{e['route']}` | **Latencia**: {e['latency_ms']} ms | **Tokens**: {e['total_tokens']:,}",
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
        "tokens_entrada_total": total_input_tokens,
        "tokens_salida_total": total_output_tokens,
        "tokens_thoughts_total": total_thoughts_tokens,
        "tokens_cached_total": total_cached_tokens,
        "tokens_total": total_tokens,
        "tokens_media_turno": avg_tokens_per_turn,
        "tokens_media_rag": avg_tokens_rag,
        "tokens_p95_turno": p95_tokens_per_turn,
        "llm_llamadas": total_llm_calls,
        "llm_failovers": total_llm_failovers,
        "rate_limit_429": total_429,
        "server_503_504": total_503_504,
        "reintentos_truncamiento": total_truncated,
        "embeddings_llamadas": total_embedding_calls,
        "embeddings_cache_hits": total_embedding_cache_hits,
        "cohere_llamadas": total_cohere_calls,
        "cohere_search_units": total_cohere_search_units,
        "cortocircuitos": short_circuits_count,
        "a1_reintentos": a1_retry_turns,
        "b2_saltos": b2_hop_turns,
        "multiterritorio_turnos": multi_territory_turns,
        "coste_real_usd": round(total_cost_real_usd, 6),
        "coste_tier_lite_usd": round(total_cost_tier_lite_usd, 6),
        "coste_tier_lite_eur": round(total_cost_tier_lite_usd * catalog.usd_to_eur, 6),
        "coste_tier_flash_usd": round(total_cost_tier_flash_usd, 6),
        "coste_tier_flash_eur": round(total_cost_tier_flash_usd * catalog.usd_to_eur, 6),
        "coste_cohere_usd": round(total_cost_cohere_usd, 6),
        "coste_embeddings_usd": round(total_cost_embeddings_usd, 6),
        "pricing_source": catalog.source,
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
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "thoughts": total_thoughts_tokens,
            "cached": total_cached_tokens,
            "total": total_tokens,
            "avg_per_turn": avg_tokens_per_turn,
            "avg_per_rag_turn": avg_tokens_rag,
            "p95_per_turn": p95_tokens_per_turn,
        },
        "api_calls": {
            "llm": total_llm_calls,
            "llm_failovers": total_llm_failovers,
            "embedding": total_embedding_calls,
            "embedding_cache_hits": total_embedding_cache_hits,
            "cohere": total_cohere_calls,
            "cohere_search_units": total_cohere_search_units,
        },
        "costs": {
            "pricing_source": catalog.source,
            "as_of_date": catalog.as_of_date,
            "real_usd": round(total_cost_real_usd, 6),
            "tier_lite_usd": round(total_cost_tier_lite_usd, 6),
            "tier_lite_eur": round(total_cost_tier_lite_usd * catalog.usd_to_eur, 6),
            "tier_flash_usd": round(total_cost_tier_flash_usd, 6),
            "tier_flash_eur": round(total_cost_tier_flash_usd * catalog.usd_to_eur, 6),
            "cohere_usd": round(total_cost_cohere_usd, 6),
            "embeddings_usd": round(total_cost_embeddings_usd, 6),
        },
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
    fetch_live_pricing: bool = True,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de sincronización, archivo, auditoría, analítica de costes y retención."""
    pull_res = HuggingFaceTelemetrySync().pull_latest_from_hf(force=force_pull)

    retention_res = apply_local_retention(db_path=SESSIONS_DB_PATH)

    if SESSIONS_DB_PATH.exists():
        SessionMemoryStore(db_path=SESSIONS_DB_PATH).export_flat_logs()

    pricing_catalog = load_pricing_catalog(fetch_live=fetch_live_pricing)

    archived_sessions = archive_session_transcripts(
        db_path=SESSIONS_DB_PATH,
        archive_dir=SESSION_ARCHIVE_DIR,
        pricing_catalog=pricing_catalog,
    )

    cost_analytics = export_cost_analytics_csvs(
        db_path=SESSIONS_DB_PATH,
        pricing_catalog=pricing_catalog,
    )

    dashboard_res = generate_cost_dashboard_html(
        pricing_catalog=pricing_catalog,
        output_path=COST_DASHBOARD_PATH,
    )

    audit_res = extract_and_audit_deltas(
        db_path=SESSIONS_DB_PATH,
        cursor_path=AUDIT_CURSOR_PATH,
        reports_dir=AUDIT_REPORTS_DIR,
        force_all=force_all_audit,
        export_candidates=export_candidates,
        pricing_catalog=pricing_catalog,
    )

    squash_res = check_and_squash_hf_history(cursor_path=AUDIT_CURSOR_PATH)

    return {
        "timestamp": _now_local().isoformat(timespec="seconds"),
        "hf_sync": pull_res,
        "local_retention": retention_res,
        "sessions_archived": archived_sessions,
        "cost_analytics": cost_analytics,
        "dashboard": dashboard_res,
        "pricing_radar": pricing_catalog.radar.model_dump(),
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
    tok = audit.get("tokens") or {}
    apis = audit.get("api_calls") or {}
    if tok:
        print(
            f"- Tokens: {tok.get('total', 0):,} total "
            f"(in={tok.get('input', 0):,}, out={tok.get('output', 0):,}, "
            f"thoughts={tok.get('thoughts', 0):,}, media={tok.get('avg_per_turn', 0):,.1f}/turno)"
        )
    if apis:
        print(
            f"- APIs: LLM={apis.get('llm', 0)} (failovers={apis.get('llm_failovers', 0)}) | "
            f"Embeddings={apis.get('embedding', 0)} (cache_hit={apis.get('embedding_cache_hits', 0)}) | "
            f"Cohere={apis.get('cohere', 0)} ({apis.get('cohere_search_units', 0)} search_units)"
        )
    costs = audit.get("costs") or {}
    if costs:
        print(
            f"- Coste API pago est. ({costs.get('pricing_source')}): Real ${costs.get('real_usd', 0):.5f} | "
            f"Tier Lite ${costs.get('tier_lite_usd', 0):.5f} | Tier Flash ${costs.get('tier_flash_usd', 0):.5f}"
        )
    ca = result.get("cost_analytics") or {}
    if ca.get("sessions_total_csv"):
        print(
            f"- Analítica costes CSV ({ca.get('pricing_source')}): {ca.get('sessions_total_csv')} sesiones, "
            f"{ca.get('turns_total_csv')} turnos"
        )
    dash = result.get("dashboard") or {}
    if dash.get("dashboard_path"):
        print(f"- Dashboard HTML: {dash['dashboard_path']} ({dash.get('bytes', 0):,} bytes)")
    radar = result.get("pricing_radar") or {}
    if radar.get("needs_attention"):
        print(
            f"- Radar modelos/tarifas: cambios={len(radar.get('price_changes') or [])}, "
            f"deprecados_config={len(radar.get('deprecated_configured_models') or [])}, "
            f"subidas_futuras={len(radar.get('upcoming_price_changes') or [])}, "
            f"nuevos={len(radar.get('new_models') or [])}"
        )
    if audit.get("report_path"):
        print(f"- Informe: {audit['report_path']}")
        print(f"- Candidatos golden añadidos: {audit.get('golden_candidates_added', 0)}")
    print(f"- Compactación HF: {sq.get('reason') or ('Realizada' if sq.get('squashed') else sq.get('error', ''))}")


if __name__ == "__main__":
    _cli()
