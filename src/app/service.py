"""Capa de Servicio Unificada y Thread-Safe para AiHoreca (`src/app/service.py`).

Resuelve el bloqueo de archivo exclusivo (`file lock`) de Qdrant Local mantendo una única instancia
compartida de `AiHorecaVectorStore` + `StateGraph` + `SessionMemoryStore`, protegida mediante
un cerrojo de concurrencia (`threading.Lock`) para entornos web (Streamlit/FastAPI) y CLI.
"""

from __future__ import annotations

import importlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

import src.app.telemetry_sync
import src.config
import src.graph.memory
import src.graph.nodes
import src.graph.state
import src.graph.workflow
import src.retrieval.vector_store
from src.app.telemetry_sync import HuggingFaceTelemetrySync
from src.config import BOUNDARIES_CACHE_PATH, EVALUATION_RESULTS_PATH, PROJECT_ROOT
from src.evaluation.evaluator import audit_corpus_and_index_stats
from src.graph.memory import SessionMemoryStore
from src.graph.nodes import get_indexed_territories
from src.graph.state import AuditReport, LegalCitation, SessionContextSlots
from src.graph.workflow import build_aihoreca_graph, run_legal_query
from src.retrieval.schemas import RetrievedDocument
from src.retrieval.vector_store import AiHorecaVectorStore


class SessionSummary(BaseModel):
    """Resumen estructurado de una sesión almacenada en SQLite."""

    session_id: str
    title: str
    active_territorio: Optional[str] = None
    active_oficio_coloquial: Optional[str] = None
    active_categoria_aleh: Optional[str] = None
    active_area_funcional_aleh: Optional[str] = None
    pending_clarification_query: Optional[str] = None
    created_at: str
    updated_at: str
    num_turns: int = 0


class TurnExecutionPayload(BaseModel):
    """Contrato completo de salida de un turno ejecutado por el servicio de aplicación."""

    session_id: str
    user_query: str
    effective_query: str
    intent: str
    motivo_clasificacion: str = ""
    route_decision: str
    provincia_detectada: Optional[str] = None
    territorio_normalizado: Optional[str] = None
    requiere_convenio_provincial: bool = False
    oficio_coloquial_detectado: Optional[str] = None
    categoria_profesional_aleh: Optional[str] = None
    area_funcional_aleh: Optional[str] = None
    query_busqueda_hibrida: str = ""
    jerarquia_normativa_aplicada: str = ""
    citas_normativas: list[LegalCitation] = Field(default_factory=list)
    cifras_y_datos_criticos: list[str] = Field(default_factory=list)
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    audit_report: AuditReport
    updated_slots: SessionContextSlots
    node_timings_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    models_used: dict[str, str] = Field(default_factory=dict)
    final_response: str


class AiHorecaEngineService:
    """Servicio singleton thread-safe que coordina Qdrant local, LangGraph y SQLite multi-sesión."""

    _instance: Optional[AiHorecaEngineService] = None
    _instance_version: str = ""
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._exec_lock = threading.Lock()
        self.vector_store = AiHorecaVectorStore()
        self.memory_store = SessionMemoryStore()
        self.telemetry_sync = HuggingFaceTelemetrySync()
        self.telemetry_sync.restore_from_hf_if_needed(self.memory_store)
        self.compiled_graph = build_aihoreca_graph(vector_store=self.vector_store)
        self._corpus_stats_cache: Optional[dict[str, Any]] = None

    @staticmethod
    def compute_code_version() -> str:
        """Calcula una huella temporal de los módulos core de `src/` para recarga automática en caliente."""
        watched_files = [
            PROJECT_ROOT / "src" / "config.py",
            PROJECT_ROOT / "src" / "retrieval" / "vector_store.py",
            PROJECT_ROOT / "src" / "graph" / "state.py",
            PROJECT_ROOT / "src" / "graph" / "memory.py",
            PROJECT_ROOT / "src" / "graph" / "nodes.py",
            PROJECT_ROOT / "src" / "graph" / "workflow.py",
            PROJECT_ROOT / "src" / "app" / "telemetry_sync.py",
        ]
        mtimes = [str(int(p.stat().st_mtime)) for p in watched_files if p.exists()]
        return "-".join(mtimes)

    @classmethod
    def get_instance(cls, code_version: Optional[str] = None) -> AiHorecaEngineService:
        """Devuelve la instancia única del motor y recarga en caliente si los módulos de `src/` cambiaron."""
        ver = code_version or cls.compute_code_version()
        if cls._instance is None or cls._instance_version != ver:
            with cls._instance_lock:
                if cls._instance is None or cls._instance_version != ver:
                    if cls._instance is not None:
                        try:
                            cls._instance.close()
                        except Exception:
                            pass
                        importlib.reload(src.config)
                        importlib.reload(src.retrieval.vector_store)
                        importlib.reload(src.graph.state)
                        importlib.reload(src.graph.memory)
                        importlib.reload(src.graph.nodes)
                        importlib.reload(src.graph.workflow)
                        importlib.reload(src.app.telemetry_sync)
                    cls._instance = cls()
                    cls._instance_version = ver
        return cls._instance

    def close(self) -> None:
        """Cierra limpiamente la conexión de Qdrant local."""
        with self._exec_lock:
            self.vector_store.close()

    def create_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Crea o inicializa una sesión en SQLite y devuelve su `session_id`."""
        sid = session_id or f"ses-{uuid.uuid4().hex[:6]}"
        return self.memory_store.ensure_session(session_id=sid, title=title)

    def list_active_sessions(self) -> list[SessionSummary]:
        """Devuelve todas las sesiones persistidas en SQLite ordenadas por actividad reciente."""
        rows = self.memory_store.list_sessions()
        return [SessionSummary.model_validate(r) for r in rows]

    def list_visitor_sessions(self, visitor_session_ids: list[str]) -> list[SessionSummary]:
        """Devuelve únicamente las sesiones pertenecientes al visitante actual (aislamiento web multi-usuario)."""
        allowed = set(visitor_session_ids)
        if not allowed:
            return []
        rows = self.memory_store.list_sessions()
        return [SessionSummary.model_validate(r) for r in rows if r.get("session_id") in allowed]

    def get_session_slots(self, session_id: str) -> SessionContextSlots:
        """Obtiene los slots de memoria de trabajo (Working Memory) vigentes de una sesión."""
        return self.memory_store.get_slots(session_id)

    def get_session_turns(self, session_id: str) -> list[dict[str, Any]]:
        """Recupera todos los turnos de una sesión desde SQLite con sus métricas y auditoría."""
        records = self.memory_store.get_full_turns(session_id)
        return [r.model_dump() for r in records]

    def record_turn_feedback(self, session_id: str, turn_index: int, feedback: str) -> bool:
        """Registra la valoración 👍 ('OK') / 👎 ('KO') de un turno y encola sincronización remota."""
        with self._exec_lock:
            updated = self.memory_store.update_turn_feedback(
                session_id=session_id,
                turn_index=turn_index,
                feedback=feedback,
            )
        if updated:
            self.telemetry_sync.trigger_async_sync()
        return updated

    def reset_or_delete_session(self, session_id: str) -> None:
        """Elimina una sesión y sus turnos asociados en SQLite."""
        with self._exec_lock:
            self.memory_store.delete_session(session_id)
            self.memory_store.export_flat_logs()
        self.telemetry_sync.trigger_async_sync()

    def process_chat_turn(self, session_id: str, query: str) -> TurnExecutionPayload:
        """Ejecuta un turno conversacional de forma thread-safe y devuelve toda la telemetría RAG."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("La consulta del usuario no puede estar vacía.")

        with self._exec_lock:
            state_out = run_legal_query(
                query=clean_query,
                session_id=session_id,
                memory_store=self.memory_store,
                compiled_graph=self.compiled_graph,
            )

        # Disparar sincronización asíncrona con debounce hacia Hugging Face Dataset (sin bloquear UI)
        self.telemetry_sync.trigger_async_sync()

        timings = state_out.get("node_timings_ms") or {}
        total_ms = round(sum(float(v) for v in timings.values()), 1)
        raw_audit = state_out.get("audit_report") or {
            "passed": True,
            "has_explicit_citations": False,
            "verified_numbers": [],
            "unverified_numbers": [],
            "warnings": [],
        }
        raw_citas = state_out.get("citas_normativas") or []
        raw_slots = state_out.get("session_slots") or self.memory_store.get_slots(session_id).model_dump()

        return TurnExecutionPayload(
            session_id=session_id,
            user_query=clean_query,
            effective_query=state_out.get("effective_query") or clean_query,
            intent=state_out.get("intent", "OUT_OF_SCOPE"),
            motivo_clasificacion=state_out.get("motivo_clasificacion", ""),
            route_decision=state_out.get("route_decision", "block_out_of_scope"),
            provincia_detectada=state_out.get("provincia_detectada"),
            territorio_normalizado=state_out.get("territorio_normalizado"),
            requiere_convenio_provincial=bool(state_out.get("requiere_convenio_provincial", False)),
            oficio_coloquial_detectado=state_out.get("oficio_coloquial_detectado"),
            categoria_profesional_aleh=state_out.get("categoria_profesional_aleh"),
            area_funcional_aleh=state_out.get("area_funcional_aleh"),
            query_busqueda_hibrida=state_out.get("query_busqueda_hibrida", ""),
            jerarquia_normativa_aplicada=state_out.get("jerarquia_normativa_aplicada", ""),
            citas_normativas=[LegalCitation.model_validate(c) for c in raw_citas],
            cifras_y_datos_criticos=state_out.get("cifras_y_datos_criticos") or [],
            retrieved_docs=state_out.get("retrieved_docs") or [],
            audit_report=AuditReport.model_validate(raw_audit),
            updated_slots=SessionContextSlots.model_validate(raw_slots),
            node_timings_ms=timings,
            total_latency_ms=total_ms,
            models_used=state_out.get("models_used") or {},
            final_response=state_out.get("final_response", ""),
        )

    def get_system_health_and_metrics(self) -> dict[str, Any]:
        """Recupera estadísticas del índice Qdrant, territorios activos y métricas del benchmark RAGAS."""
        if self._corpus_stats_cache is None:
            self._corpus_stats_cache = audit_corpus_and_index_stats()

        eval_data: dict[str, Any] = {}
        eval_path = Path(EVALUATION_RESULTS_PATH)
        if eval_path.exists():
            try:
                eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception:
                eval_data = {}

        boundaries_data: dict[str, Any] = {}
        b_path = Path(BOUNDARIES_CACHE_PATH)
        if b_path.exists():
            try:
                boundaries_data = json.loads(b_path.read_text(encoding="utf-8"))
            except Exception:
                boundaries_data = {}

        return {
            "indexed_territories": sorted(get_indexed_territories()),
            "corpus_stats": self._corpus_stats_cache,
            "boundaries_cache": boundaries_data,
            "evaluation_benchmark": eval_data,
        }
