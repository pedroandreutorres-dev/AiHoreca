"""Interfaz Interactiva Multi-Sesión y Consola de Auditoría Jurídica en Tiempo Real (`src/app/ui.py`).

Ejecutar con:
    streamlit run src/app/ui.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.service import AiHorecaEngineService, TurnExecutionPayload


@st.cache_resource(show_spinner="Inicializando motor híbrido Qdrant + LangGraph + SQLite...")
def get_engine_service(code_version: str) -> AiHorecaEngineService:
    """Mantiene una única instancia singleton de `AiHorecaEngineService` y se invalida automáticamente si cambia el código en `src/`."""
    return AiHorecaEngineService.get_instance(code_version=code_version)


def _init_ui_state(service: AiHorecaEngineService) -> None:
    """Inicializa el estado de sesión en la UI con aislamiento estricto por visitante (pestaña de navegador)."""
    if "turn_telemetry_cache" not in st.session_state:
        st.session_state.turn_telemetry_cache = {}

    if "visitor_session_ids" not in st.session_state or not st.session_state.visitor_session_ids:
        initial_sid = service.create_session()
        st.session_state.visitor_session_ids = [initial_sid]
        st.session_state.active_session_id = initial_sid
    elif "active_session_id" not in st.session_state:
        st.session_state.active_session_id = st.session_state.visitor_session_ids[0]


def _render_turn_feedback_controls(
    service: AiHorecaEngineService,
    session_id: str,
    turn_index: int,
    current_feedback: str,
) -> None:
    """Renderiza los botones de valoración 👍 / 👎 debajo de una respuesta del asistente."""
    fb = (current_feedback or "SIN_VOTO").upper()
    col_ok, col_ko, col_status = st.columns([1.2, 1.3, 6.5])
    with col_ok:
        if st.button(
            "👍 Útil",
            key=f"fb_ok_{session_id}_{turn_index}",
            type="primary" if fb == "OK" else "secondary",
            use_container_width=True,
        ):
            new_val = "SIN_VOTO" if fb == "OK" else "OK"
            service.record_turn_feedback(session_id=session_id, turn_index=turn_index, feedback=new_val)
            st.rerun()
    with col_ko:
        if st.button(
            "👎 Mejorar",
            key=f"fb_ko_{session_id}_{turn_index}",
            type="primary" if fb == "KO" else "secondary",
            use_container_width=True,
        ):
            new_val = "SIN_VOTO" if fb == "KO" else "KO"
            service.record_turn_feedback(session_id=session_id, turn_index=turn_index, feedback=new_val)
            st.rerun()
    with col_status:
        if fb == "OK":
            st.caption("Valoración registrada: **👍 Respuesta útil**")
        elif fb == "KO":
            st.caption("Valoración registrada: **👎 Marcada para revisión**")


def _render_turn_xray_panel(telemetry: TurnExecutionPayload) -> None:
    """Renderiza el panel desplegable de rayos-X (telemetría RAG + Guardrail Nodo 5) bajo la respuesta."""
    route_badge = {
        "parallel_retrieve": "RAG Híbrido 3 Capas + Síntesis",
        "ask_clarification": "Gatekeeper HITL (Falta Contexto)",
        "block_out_of_scope": "Guardrail de Entrada (Bloqueo 0 Tokens)",
    }.get(telemetry.route_decision, telemetry.route_decision)

    audit_status = "PASSED (0 Alucinaciones)" if telemetry.audit_report.passed else "ALERTA DE AUDITORÍA"
    expander_title = (
        f"Rayos-X del Turno | Ruta: {route_badge} | "
        f"Territorio: {telemetry.territorio_normalizado or telemetry.provincia_detectada or 'Estatal/N/A'} | "
        f"Auditoría Nodo 5: {audit_status} | Latencia: {telemetry.total_latency_ms:.0f} ms"
    )

    with st.expander(expander_title, expanded=False):
        tab_route, tab_audit, tab_docs, tab_perf = st.tabs(
            [
                "1. Enrutamiento y Correferencia",
                "2. Guardrail Numérico (Nodo 5)",
                f"3. Evidencia Normativa ({len(telemetry.retrieved_docs)} Chunks)",
                "4. Latencias y Modelos",
            ]
        )

        with tab_route:
            c1, c2, c3 = st.columns(3)
            c1.metric("Intención Detectada", telemetry.intent)
            c2.metric("Decisión de Ruta", telemetry.route_decision)
            c3.metric("Territorio Activo", telemetry.territorio_normalizado or telemetry.provincia_detectada or "N/A")

            st.markdown(f"**Consulta Original del Turno:** `{telemetry.user_query}`")
            st.markdown(f"**Consulta Autocontenida Resuelta (`effective_query`):** `{telemetry.effective_query}`")
            if telemetry.query_busqueda_hibrida:
                st.markdown(f"**Query Enriquecida para Qdrant:** `{telemetry.query_busqueda_hibrida}`")
            st.markdown(
                f"**Encuadramiento VI ALEH:** Oficio=`{telemetry.oficio_coloquial_detectado or 'N/A'}` | "
                f"Categoría=`{telemetry.categoria_profesional_aleh or 'N/A'}` | "
                f"Área Funcional=`{telemetry.area_funcional_aleh or 'N/A'}`"
            )
            if telemetry.motivo_clasificacion:
                st.caption(f"Motivo del clasificador: {telemetry.motivo_clasificacion}")

        with tab_audit:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Estado Guardrail Salida", "PASSED" if telemetry.audit_report.passed else "WARN")
            a2.metric("Cifras Literales (Corpus)", len(telemetry.audit_report.verified_numbers))
            a3.metric("Cálculos Derivados", len(telemetry.audit_report.derived_numbers))
            a4.metric("Cifras No Verificadas", len(telemetry.audit_report.unverified_numbers))

            if telemetry.audit_report.verified_numbers:
                st.success(
                    "Importes (€) y porcentajes (%) verificados literalmente contra los fragmentos de Qdrant: "
                    + ", ".join(f"`{n}`" for n in telemetry.audit_report.verified_numbers)
                )
            if telemetry.audit_report.derived_numbers:
                st.info(
                    "Importes calculados mediante fórmulas trazables (operandos verificados en corpus): "
                    + ", ".join(f"`{n}`" for n in telemetry.audit_report.derived_numbers)
                )
            if telemetry.audit_report.unverified_numbers:
                st.error(
                    "Cifras no localizadas literalmente en los chunks: "
                    + ", ".join(f"`{n}`" for n in telemetry.audit_report.unverified_numbers)
                )
            if telemetry.audit_report.warnings:
                for w in telemetry.audit_report.warnings:
                    st.info(w)

            if telemetry.citas_normativas:
                st.markdown("#### Citas Normativas Estructuradas")
                citas_rows = [
                    {
                        "Norma": c.norma,
                        "Capa": c.capa_normativa,
                        "Artículo / Tabla": c.articulo_o_tabla,
                        "Extracto Clave": c.extracto_clave,
                    }
                    for c in telemetry.citas_normativas
                ]
                st.dataframe(citas_rows, use_container_width=True, hide_index=True)

        with tab_docs:
            if not telemetry.retrieved_docs:
                st.info("Este turno se resolvió por cortocircuito determinista (0 llamadas a Qdrant/Cohere).")
            else:
                for idx, doc in enumerate(telemetry.retrieved_docs, start=1):
                    score_val = doc.rerank_score if doc.rerank_score is not None else doc.score
                    st.markdown(
                        f"**#{idx} — [{doc.capa_normativa.upper()}] {doc.norma_titulo}**  \n"
                        f"`Territorio: {doc.territorio}` | `Sección: {doc.titulo_o_capitulo}` | "
                        f"`Ref: {doc.articulo}` | `Tipo: {doc.tipo_contenido}` | "
                        f"**Cohere Rerank Score: `{score_val:.4f}`**"
                    )
                    st.code(doc.text, language="markdown")

        with tab_perf:
            st.markdown("#### Desglose de Latencia por Nodo (ms)")
            st.json(telemetry.node_timings_ms)
            st.markdown("#### Modelos Utilizados en cada Etapa")
            st.json(telemetry.models_used)


def _render_sidebar(service: AiHorecaEngineService) -> str:
    """Renderiza el panel lateral de gestión de sesiones SQLite y visualización de Working Memory."""
    with st.sidebar:
        st.title("AiHoreca — Panel de Control")
        st.caption("RAG Agéntico Híbrido (CPU) • Jerarquía Normativa 3 Capas • Memoria SQLite WAL")

        st.subheader("1. Gestión de Tu Sesión")
        visitor_sids: list[str] = st.session_state.visitor_session_ids
        if st.session_state.active_session_id not in visitor_sids:
            visitor_sids.insert(0, st.session_state.active_session_id)

        sessions = service.list_visitor_sessions(visitor_sids)
        session_ids = [s.session_id for s in sessions]
        if st.session_state.active_session_id not in session_ids:
            service.create_session(st.session_state.active_session_id)
            sessions = service.list_visitor_sessions(visitor_sids)
            session_ids = [s.session_id for s in sessions]

        session_labels = {
            s.session_id: f"{s.session_id} ({s.num_turns}t | {s.active_territorio or 'sin prov.'})"
            for s in sessions
        }

        selected_sid = st.selectbox(
            "Sesión Activa",
            options=session_ids,
            index=session_ids.index(st.session_state.active_session_id),
            format_func=lambda x: session_labels.get(x, x),
        )
        if selected_sid != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_sid
            st.rerun()

        col_new, col_del = st.columns(2)
        with col_new:
            if st.button("+ Nueva Sesión", use_container_width=True):
                new_sid = service.create_session()
                st.session_state.visitor_session_ids.insert(0, new_sid)
                st.session_state.active_session_id = new_sid
                st.rerun()
        with col_del:
            if st.button("Limpiar Sesión", use_container_width=True):
                service.reset_or_delete_session(st.session_state.active_session_id)
                service.create_session(st.session_state.active_session_id)
                st.session_state.turn_telemetry_cache.pop(st.session_state.active_session_id, None)
                st.rerun()

        st.divider()
        st.subheader("2. Working Memory (`SessionContextSlots`)")
        slots = service.get_session_slots(st.session_state.active_session_id)
        st.markdown(f"- **Territorio Activo:** `{slots.active_territorio or 'Pendiente'}`")
        st.markdown(f"- **Oficio Detectado:** `{slots.active_oficio_coloquial or 'N/A'}`")
        st.markdown(f"- **Categoría VI ALEH:** `{slots.active_categoria_aleh or 'N/A'}`")
        st.markdown(f"- **Área Funcional:** `{slots.active_area_funcional_aleh or 'N/A'}`")
        if slots.pending_clarification_query:
            st.warning(f"**Gatekeeper HITL Pendiente:**\n{slots.pending_clarification_query}")
        else:
            st.caption("Sin preguntas pendientes de aclaración contextual.")

    return st.session_state.active_session_id


def main() -> None:
    st.set_page_config(
        page_title="AiHoreca — Asistente Jurídico RAG Agéntico",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    service = get_engine_service(AiHorecaEngineService.compute_code_version())
    _init_ui_state(service)
    active_sid = _render_sidebar(service)

    st.title("AiHoreca — Asistente Legal RAG Agéntico para Hostelería")
    st.caption(
        "Estatuto de los Trabajadores (BOE) • VI Acuerdo Laboral Estatal de Hostelería (ALEH VI) • "
        "Convenios Colectivos de Madrid (BOCM), Cataluña/Barcelona/Girona/Tarragona/Lleida (DOGC) y León (BOP)"
    )

    persisted_turns = service.get_session_turns(active_sid)
    session_telemetry: dict[int, TurnExecutionPayload] = st.session_state.turn_telemetry_cache.setdefault(
        active_sid, {}
    )

    if not persisted_turns:
        st.info(
            "Sesión activa sin turnos previos. Escribe abajo tu consulta laboral sobre hostelería "
            "(ej. salario base, categoría profesional, pagas extraordinarias, vacaciones, nocturnidad o régimen disciplinario)."
        )

    for t in persisted_turns:
        t_idx = int(t["turn_index"])
        with st.chat_message("user"):
            st.markdown(t["user_query"])
        with st.chat_message("assistant"):
            st.markdown(t["final_response"])
            if t_idx in session_telemetry:
                _render_turn_xray_panel(session_telemetry[t_idx])
            else:
                st.caption(
                    f"Turno histórico #{t_idx} | Ruta: `{t['route_decision']}` | "
                    f"Territorio: `{t['territorio'] or 'N/A'}` | "
                    f"Consulta resuelta: `{t['effective_query']}` | "
                    f"Guardrail Nodo 5: `{'PASSED' if t['audit_passed'] else 'WARN'}`"
                )
            _render_turn_feedback_controls(
                service=service,
                session_id=active_sid,
                turn_index=t_idx,
                current_feedback=t.get("user_feedback", "SIN_VOTO"),
            )

    user_input = st.chat_input(
        "Escribe tu consulta laboral de hostelería (ej. salario base, vacaciones, nocturnidad, faltas)..."
    )

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Ejecutando Grafo Agéntico (Router -> Qdrant 3 Capas + Cohere Rerank -> Síntesis + Auditoría)..."):
                payload = service.process_chat_turn(session_id=active_sid, query=user_input)
                updated_turns = service.get_session_turns(active_sid)
                latest_turn_idx = int(updated_turns[-1]["turn_index"]) if updated_turns else 1
                session_telemetry[latest_turn_idx] = payload

            st.markdown(payload.final_response)
            _render_turn_xray_panel(payload)
        st.rerun()


if __name__ == "__main__":
    main()
