"""Gestor de Memoria Conversacional Multi-Sesión persistente en SQLite local (CPU) para AiHoreca.

Principios de ingeniería:
1. Aislamiento estricto por `session_id`: múltiples sesiones concurrentes o históricas conviven sin colisión.
2. Separación entre Working Memory (`SessionContextSlots`) y Episodic Memory (`turns`):
   - `SessionContextSlots` almacena el estado estructurado vigente de la sesión (`active_territorio`,
     `active_oficio_coloquial`, `active_categoria_aleh`, `active_area_funcional_aleh`, `pending_clarification_query`).
   - `get_compact_history()` devuelve únicamente una ventana deslizante ligera (`max_turns=4`) con resúmenes
     estructurales de las respuestas del asistente (<250 tokens), evitando intoxicar el router con tablas extensas.
3. Protección contra envenenamiento de estado (*State Poisoning*):
   - Si un turno intermedio es bloqueado por `OUT_OF_SCOPE` o `UNSUPPORTED_PROVINCE` (ej. consulta puntual sobre Valencia),
     el turno se registra en la auditoría pero NO corrompe el `active_territorio` válido previo de la sesión.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import CONVERSATIONS_TXT_PATH, LOGS_DIR, SESSIONS_DB_PATH, TURNS_CSV_PATH
from src.graph.state import LegalGraphState, SessionContextSlots, SessionTurnRecord


class SessionMemoryStore:
    """Almacén transaccional SQLite para memoria conversacional multi-sesión y slots de contexto."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or SESSIONS_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    active_territorio TEXT,
                    active_oficio_coloquial TEXT,
                    active_categoria_aleh TEXT,
                    active_area_funcional_aleh TEXT,
                    pending_clarification_query TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    effective_query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    route_decision TEXT NOT NULL,
                    territorio TEXT,
                    oficio_detectado TEXT,
                    categoria_aleh TEXT,
                    area_funcional_aleh TEXT,
                    assistant_summary TEXT NOT NULL,
                    final_response TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    timings_json TEXT NOT NULL,
                    models_json TEXT NOT NULL DEFAULT '{}',
                    user_feedback TEXT NOT NULL DEFAULT 'SIN_VOTO',
                    feedback_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_turn
                ON session_turns(session_id, turn_index);
                """
            )
            # Migración automática retrocompatible si la tabla session_turns ya existía en disco
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(session_turns)").fetchall()
            }
            migrations = [
                ("oficio_detectado", "TEXT"),
                ("models_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("user_feedback", "TEXT NOT NULL DEFAULT 'SIN_VOTO'"),
                ("feedback_at", "TEXT"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE session_turns ADD COLUMN {col_name} {col_type}"
                    )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def ensure_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Garantiza que la sesión exista en SQLite; si no se pasa `session_id`, genera uno nuevo."""
        sid = (session_id or f"session-{uuid.uuid4().hex[:8]}").strip()
        now = self._now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, title, active_territorio, active_oficio_coloquial,
                        active_categoria_aleh, active_area_funcional_aleh,
                        pending_clarification_query, created_at, updated_at
                    ) VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)
                    """,
                    (sid, title or f"Sesión {sid}", now, now),
                )
        return sid

    def get_slots(self, session_id: str) -> SessionContextSlots:
        """Recupera los slots de memoria de trabajo (Working Memory) de una sesión."""
        self.ensure_session(session_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT active_territorio, active_oficio_coloquial,
                       active_categoria_aleh, active_area_funcional_aleh,
                       pending_clarification_query
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if not row:
                return SessionContextSlots()
            return SessionContextSlots(
                active_territorio=row["active_territorio"],
                active_oficio_coloquial=row["active_oficio_coloquial"],
                active_categoria_aleh=row["active_categoria_aleh"],
                active_area_funcional_aleh=row["active_area_funcional_aleh"],
                pending_clarification_query=row["pending_clarification_query"],
            )

    def get_compact_history(self, session_id: str, max_turns: int = 4) -> list[dict[str, str]]:
        """Devuelve una ventana deslizante compacta de los últimos `max_turns` turnos para el router."""
        self.ensure_session(session_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, user_query, effective_query, assistant_summary
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (session_id, max_turns),
            ).fetchall()

        history: list[dict[str, str]] = []
        for r in reversed(rows):
            history.append(
                {
                    "role": "user",
                    "content": r["user_query"],
                    "resolved_query": r["effective_query"],
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": r["assistant_summary"],
                }
            )
        return history

    def get_full_turns(self, session_id: str) -> list[SessionTurnRecord]:
        """Recupera todos los turnos completos de una sesión para interfaz de usuario o auditoría."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, timestamp, user_query, effective_query,
                       intent, route_decision, territorio, oficio_detectado,
                       categoria_aleh, area_funcional_aleh, assistant_summary,
                       final_response, audit_json, user_feedback, feedback_at
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_index ASC
                """,
                (session_id,),
            ).fetchall()

        records: list[SessionTurnRecord] = []
        for r in rows:
            audit_data = json.loads(r["audit_json"] or "{}")
            records.append(
                SessionTurnRecord(
                    turn_index=int(r["turn_index"]),
                    timestamp=r["timestamp"],
                    user_query=r["user_query"],
                    effective_query=r["effective_query"],
                    intent=r["intent"],
                    route_decision=r["route_decision"],
                    territorio=r["territorio"],
                    oficio_detectado=r["oficio_detectado"],
                    categoria_aleh=r["categoria_aleh"],
                    area_funcional_aleh=r["area_funcional_aleh"],
                    assistant_summary=r["assistant_summary"],
                    final_response=r["final_response"],
                    audit_passed=bool(audit_data.get("passed", True)),
                    user_feedback=r["user_feedback"] or "SIN_VOTO",
                    feedback_at=r["feedback_at"],
                )
            )
        return records

    def save_turn_and_update_slots(
        self,
        session_id: str,
        state: LegalGraphState,
    ) -> SessionContextSlots:
        """Persiste el turno ejecutado y actualiza de forma segura los slots de la sesión."""
        sid = self.ensure_session(session_id)
        current_slots = self.get_slots(sid)

        intent = state.get("intent", "OUT_OF_SCOPE")
        route = state.get("route_decision", "block_out_of_scope")
        user_query = state.get("query", "")
        effective_query = state.get("effective_query") or user_query
        terr = state.get("territorio_normalizado")
        oficio = state.get("oficio_coloquial_detectado")
        cat_aleh = state.get("categoria_profesional_aleh")
        area_aleh = state.get("area_funcional_aleh")
        final_response = state.get("final_response", "")

        new_territorio = current_slots.active_territorio
        new_oficio = current_slots.active_oficio_coloquial
        new_cat_aleh = current_slots.active_categoria_aleh
        new_area_aleh = current_slots.active_area_funcional_aleh
        new_pending = current_slots.pending_clarification_query

        if route == "ask_clarification":
            # Guardamos la consulta pendiente para fusionarla cuando el usuario indique la provincia
            new_pending = effective_query
            if oficio:
                new_oficio = oficio
            if cat_aleh:
                new_cat_aleh = cat_aleh
            if area_aleh:
                new_area_aleh = area_aleh
        elif route == "parallel_retrieve" and intent == "HOSPITALITY_CORE":
            # Consulta completada con éxito: actualizamos territorio y puesto activos y limpiamos aclaración pendiente
            new_pending = None
            if terr:
                new_territorio = terr
            if oficio:
                new_oficio = oficio
            if cat_aleh:
                new_cat_aleh = cat_aleh
            if area_aleh:
                new_area_aleh = area_aleh
        # Nota crítica: si route == "block_out_of_scope" (ej. pregunta puntual sobre Valencia o accidente),
        # NO sobreescribimos new_territorio ni new_cat_aleh para no envenenar el estado válido de la sesión.

        # Construir resumen semántico compacto de la respuesta del asistente para memoria episódica
        citas = state.get("citas_normativas") or []
        citas_str = ", ".join(
            f"{c.get('norma', '')} ({c.get('articulo_o_tabla', '')})" for c in citas[:3]
        )
        cifras = state.get("cifras_y_datos_criticos") or []
        cifras_str = ", ".join(cifras[:4])

        # Extraer síntesis textual limpia de final_response (sin tablas Markdown pesadas) para que el Router
        # y el Sintetizador sepan exactamente qué se explicó en turnos previos sin saturar tokens.
        clean_lines: list[str] = []
        for line in (final_response or "").splitlines():
            s = line.strip()
            if not s or s.startswith("|") or s.startswith("---"):
                continue
            s = re.sub(r"^[#>*\-\d\.\)\s]+", "", s).replace("**", "").strip()
            if s:
                clean_lines.append(s)
        semantic_digest = " ".join(clean_lines)[:420]

        if route == "ask_clarification":
            assistant_summary = (
                f"[GATEKEEPER_HITL] Solicitó provincia al usuario para resolver: '{effective_query}'. "
                f"Adelantó encuadramiento ALEH: {new_cat_aleh or 'N/A'}."
            )
        elif route == "block_out_of_scope":
            assistant_summary = (
                f"[BLOQUEO_{intent}] Rechazó consulta fuera de cobertura ({state.get('provincia_detectada') or 'no hostelería'})."
            )
        else:
            assistant_summary = (
                f"[DICTAMEN_EMITIDO] Territorio={new_territorio} | Categoría={new_cat_aleh} | "
                f"Citas=[{citas_str}] | Cifras=[{cifras_str}] | Resumen='{semantic_digest}'"
            )

        now = self._now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) AS max_turn FROM session_turns WHERE session_id = ?",
                (sid,),
            ).fetchone()
            next_turn = int(row["max_turn"]) + 1

            # Actualizar título de la sesión en el primer turno
            if next_turn == 1:
                short_title = (user_query[:55] + "...") if len(user_query) > 58 else user_query
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (short_title, sid),
                )

            conn.execute(
                """
                UPDATE sessions
                SET active_territorio = ?,
                    active_oficio_coloquial = ?,
                    active_categoria_aleh = ?,
                    active_area_funcional_aleh = ?,
                    pending_clarification_query = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    new_territorio,
                    new_oficio,
                    new_cat_aleh,
                    new_area_aleh,
                    new_pending,
                    now,
                    sid,
                ),
            )

            conn.execute(
                """
                INSERT INTO session_turns (
                    session_id, turn_index, timestamp, user_query, effective_query,
                    intent, route_decision, territorio, oficio_detectado,
                    categoria_aleh, area_funcional_aleh, assistant_summary,
                    final_response, audit_json, timings_json, models_json,
                    user_feedback, feedback_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SIN_VOTO', NULL)
                """,
                (
                    sid,
                    next_turn,
                    now,
                    user_query,
                    effective_query,
                    intent,
                    route,
                    terr,
                    new_oficio,
                    new_cat_aleh,
                    new_area_aleh,
                    assistant_summary,
                    final_response,
                    json.dumps(state.get("audit_report") or {}, ensure_ascii=False),
                    json.dumps(state.get("node_timings_ms") or {}, ensure_ascii=False),
                    json.dumps(state.get("models_used") or {}, ensure_ascii=False),
                ),
            )

        # Fase 1 (T = 0 ms): Exportación inmediata a archivos planos sin esperar voto del usuario
        self.export_flat_logs()

        return SessionContextSlots(
            active_territorio=new_territorio,
            active_oficio_coloquial=new_oficio,
            active_categoria_aleh=new_cat_aleh,
            active_area_funcional_aleh=new_area_aleh,
            pending_clarification_query=new_pending,
        )

    def update_turn_feedback(self, session_id: str, turn_index: int, feedback: str) -> bool:
        """Fase 2 (T > 0): Actualiza el voto ('OK' / 'KO') de un turno y regenera los archivos de telemetría."""
        normalized = feedback.strip().upper()
        if normalized not in ("OK", "KO", "SIN_VOTO"):
            raise ValueError(f"Feedback inválido: {feedback}")

        now = self._now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE session_turns
                SET user_feedback = ?,
                    feedback_at = ?
                WHERE session_id = ? AND turn_index = ?
                """,
                (normalized, now if normalized != "SIN_VOTO" else None, session_id, int(turn_index)),
            )
            updated = cur.rowcount > 0

        if updated:
            self.export_flat_logs()
        return updated

    def export_flat_logs(
        self,
        txt_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """Genera de forma atómica `conversaciones.txt` (agrupado por session_id) y `turnos_metricas.csv`."""
        target_txt = Path(txt_path or CONVERSATIONS_TXT_PATH)
        target_csv = Path(csv_path or TURNS_CSV_PATH)
        target_txt.parent.mkdir(parents=True, exist_ok=True)
        target_csv.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id,
                       s.created_at AS session_created_at,
                       s.active_territorio AS session_active_territorio,
                       t.turn_index,
                       t.timestamp,
                       t.user_query,
                       t.effective_query,
                       t.intent,
                       t.route_decision,
                       t.territorio,
                       t.oficio_detectado,
                       t.categoria_aleh,
                       t.final_response,
                       t.audit_json,
                       t.timings_json,
                       t.models_json,
                       t.user_feedback,
                       t.feedback_at
                FROM session_turns t
                JOIN sessions s ON s.session_id = t.session_id
                ORDER BY s.created_at ASC, s.session_id ASC, t.turn_index ASC
                """
            ).fetchall()

        # 1. Construir `conversaciones.txt` agrupado por session_id (texto plano limpio para lectura humana)
        txt_blocks: list[str] = []
        current_sid: Optional[str] = None
        for r in rows:
            sid = r["session_id"]
            if sid != current_sid:
                if current_sid is not None:
                    txt_blocks.append("=" * 80 + "\n\n")
                current_sid = sid
                terr_hdr = r["session_active_territorio"] or r["territorio"] or "N/A"
                txt_blocks.append(
                    "=" * 80 + "\n"
                    f"SESSION_ID: {sid} | Inicio: {r['session_created_at']} | Territorio Activo: {terr_hdr}\n"
                    + "=" * 80 + "\n\n"
                )
            else:
                txt_blocks.append("-" * 80 + "\n\n")

            txt_blocks.append(
                f"[Turno {r['turn_index']} | {r['timestamp']}] USUARIO:\n"
                f"{r['user_query'].strip()}\n\n"
                f"[Turno {r['turn_index']} | {r['timestamp']}] AIHORECA:\n"
                f"{r['final_response'].strip()}\n\n"
            )

        if current_sid is not None:
            txt_blocks.append("=" * 80 + "\n")

        tmp_txt = target_txt.with_suffix(".txt.tmp")
        tmp_txt.write_text("".join(txt_blocks), encoding="utf-8")
        os.replace(tmp_txt, target_txt)

        # 2. Construir `turnos_metricas.csv` relacional con clave compuesta (session_id, turn_index)
        csv_headers = [
            "session_id",
            "turn_index",
            "created_at",
            "intent",
            "route_decision",
            "territorio",
            "oficio_detectado",
            "categoria_aleh",
            "audit_passed",
            "verified_numbers_count",
            "unverified_numbers_count",
            "total_latency_ms",
            "model_router",
            "model_synthesis",
            "user_feedback",
            "feedback_at",
            "user_query",
            "effective_query",
        ]

        tmp_csv = target_csv.with_suffix(".csv.tmp")
        with tmp_csv.open("w", encoding="utf-8-sig", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(csv_headers)
            for r in rows:
                audit_dict = json.loads(r["audit_json"] or "{}")
                timings_dict = json.loads(r["timings_json"] or "{}")
                models_dict = json.loads(r["models_json"] or "{}")

                passed_int = 1 if audit_dict.get("passed", True) else 0
                verified_cnt = len(audit_dict.get("verified_numbers") or []) + len(
                    audit_dict.get("derived_numbers") or []
                )
                unverified_cnt = len(audit_dict.get("unverified_numbers") or [])
                total_ms = round(sum(float(v) for v in timings_dict.values()), 1)

                writer.writerow(
                    [
                        r["session_id"],
                        r["turn_index"],
                        r["timestamp"],
                        r["intent"],
                        r["route_decision"],
                        r["territorio"] or "",
                        r["oficio_detectado"] or "",
                        r["categoria_aleh"] or "",
                        passed_int,
                        verified_cnt,
                        unverified_cnt,
                        total_ms,
                        models_dict.get("analyze_and_route", ""),
                        models_dict.get("legal_synthesis_and_audit", ""),
                        r["user_feedback"] or "SIN_VOTO",
                        r["feedback_at"] or "",
                        r["user_query"],
                        r["effective_query"],
                    ]
                )
        os.replace(tmp_csv, target_csv)
        return target_txt, target_csv

    def list_sessions(self) -> list[dict[str, Any]]:
        """Lista todas las sesiones almacenadas con sus slots activos y número de turnos."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.active_territorio,
                       s.active_oficio_coloquial, s.active_categoria_aleh,
                       s.active_area_funcional_aleh, s.pending_clarification_query,
                       s.created_at, s.updated_at,
                       COUNT(t.id) AS num_turns
                FROM sessions s
                LEFT JOIN session_turns t ON s.session_id = t.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        """Elimina una sesión y todos sus turnos asociados."""
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
