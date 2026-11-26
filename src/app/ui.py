"""Interfaz Interactiva Multi-Sesión y Consola de Auditoría Jurídica en Tiempo Real (`src/app/ui.py`).

Ejecutar con:
    streamlit run src/app/ui.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import uuid

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.service import AiHorecaEngineService, TurnExecutionPayload
from src.config import CONTACT_EMAIL, SESSION_RETENTION_DAYS


@st.cache_resource(show_spinner="Inicializando motor híbrido Qdrant + LangGraph + SQLite...")
def get_engine_service(code_version: str) -> AiHorecaEngineService:
    """Mantiene una única instancia singleton de `AiHorecaEngineService` y se invalida automáticamente si cambia el código en `src/`."""
    return AiHorecaEngineService.get_instance(code_version=code_version)


def _init_ui_state(service: AiHorecaEngineService) -> None:
    """Inicializa el estado de sesión en la UI con aislamiento estricto por visitante (pestaña de navegador)."""
    if "turn_telemetry_cache" not in st.session_state:
        st.session_state.turn_telemetry_cache = {}

    if "visitor_session_ids" not in st.session_state or not st.session_state.visitor_session_ids:
        initial_sid = f"ses-{uuid.uuid4().hex[:6]}"
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


def _format_stream_node_event(node_name: str, update: dict[str, Any]) -> tuple[str, str]:
    """Traduce una actualización de nodo LangGraph a etiqueta de estado y detalle legible en tiempo real."""
    if node_name == "analyze_and_route":
        route = update.get("route_decision", "")
        terrs = update.get("territorios_normalizados") or []
        terr_lbl = ", ".join(terrs) if terrs else (update.get("territorio_normalizado") or "Estatal (ET/ALEH)")
        if route == "block_out_of_scope":
            return "Guardrail de entrada completado", f"- **Enrutamiento:** Bloqueo determinista (`{update.get('intent', 'OUT_OF_SCOPE')}`)"
        if route == "ask_clarification":
            return "Clarificación contextual requerida", "- **Gatekeeper HITL:** Faltan datos mínimos antes de recuperar tablas"
        return (
            "Enrutamiento completado; recuperando normativa en paralelo...",
            f"- **Análisis y ruta (`analyze_and_route`):** `{update.get('intent', 'HOSPITALITY_CORE')}` | Territorios: `{terr_lbl}`",
        )
    if node_name == "retrieve_state_layers":
        docs = update.get("branch_docs") or []
        return (
            "Capas estatales (ET + ALEH) recuperadas",
            f"- **Rama estatal (`retrieve_state_layers`):** `{len(docs)}` candidatos (Estatuto + VI ALEH)",
        )
    if node_name == "retrieve_territory":
        docs = update.get("branch_docs") or []
        timings = update.get("branch_timings") or []
        terr_name = timings[-1][0].split(":", 1)[-1] if timings else "provincial"
        return (
            f"Convenio de {terr_name} recuperado",
            f"- **Rama territorial (`retrieve_territory:{terr_name}`):** `{len(docs)}` candidatos híbridos",
        )
    if node_name == "verify_retrieval":
        docs = update.get("retrieved_docs") or []
        needs_exp = bool(update.get("needs_retrieval_expansion"))
        if needs_exp:
            refs = [f"{r.get('territorio')}:{r.get('ref_label')}" for r in (update.get("missing_cross_refs") or [])]
            return (
                "Remisiones internas detectadas; ejecutando 2.º salto...",
                f"- **Verificación (`verify_retrieval`):** `{len(docs)}` fragmentos reranqueados; remisiones ausentes: `{', '.join(refs)}`",
            )
        return (
            "Contexto verificado; redactando dictamen jurídico...",
            f"- **Verificación (`verify_retrieval`):** `{len(docs)}` fragmentos reranqueados y validados",
        )
    if node_name == "expand_retrieval":
        docs = update.get("retrieved_docs") or []
        return (
            "2.º salto completado; redactando dictamen jurídico...",
            f"- **Expansión (`expand_retrieval`):** Contexto ampliado a `{len(docs)}` fragmentos (hop 1)",
        )
    if node_name == "synthesize_response":
        return (
            "Dictamen generado; ejecutando auditoría determinista...",
            "- **Síntesis jurídica (`synthesize_response`):** Redacción estructurada y citas generadas",
        )
    if node_name == "audit_output":
        if update.get("audit_needs_retry"):
            return (
                "Auditoría solicitó corrección; reintentando síntesis...",
                "- **Auditoría (`audit_output`):** Detectada cifra/cita revisable; autocorrección (reintento 1/1)",
            )
        audit = update.get("audit_report") or {}
        state_lbl = "PASSED" if audit.get("passed", True) else "CON AVISO"
        return (
            "Auditoría determinista completada",
            f"- **Auditoría (`audit_output`):** Resultado `{state_lbl}`",
        )
    if node_name in ("block_out_of_scope", "ask_clarification"):
        return (
            "Respuesta directa generada (0 tokens extra)",
            f"- **Cortocircuito (`{node_name}`):** Respuesta determinista preparada",
        )
    return (f"Ejecutado nodo `{node_name}`", f"- Nodo `{node_name}` completado")


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
            if len(telemetry.territorios_normalizados) > 1 or telemetry.retrieval_hop > 0:
                st.caption(
                    f"Territorios paralelos (`Send`): `{telemetry.territorios_normalizados}` | "
                    f"Saltos de expansión (`retrieval_hop`): `{telemetry.retrieval_hop}`"
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
            tu = telemetry.token_usage
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tokens Totales", f"{tu.total_tokens:,}")
            m2.metric("Entrada (Input)", f"{tu.input_tokens:,}")
            m3.metric("Salida (Output)", f"{tu.output_tokens:,}")
            m4.metric("Embeddings (Calls / Hit)", f"{tu.embedding_calls} / {tu.embedding_cache_hits}")
            m5.metric("Cohere Search Units", f"{tu.cohere_search_units} ({tu.cohere_calls} calls)")

            if tu.thoughts_tokens or tu.cached_tokens or tu.llm_failovers:
                st.caption(
                    f"Thoughts: `{tu.thoughts_tokens:,}` tok | "
                    f"Context Cache: `{tu.cached_tokens:,}` tok | "
                    f"Llamadas LLM: `{tu.llm_calls}` | "
                    f"Failovers LLM: `{tu.llm_failovers}` "
                    f"(429=`{tu.rate_limit_429_count}`, 503/504=`{tu.server_503_504_count}`, trunc=`{tu.truncated_retries}`)"
                )

            if tu.steps:
                st.markdown("#### Consumo Paso a Paso por Nodo / Proveedor")
                step_rows = []
                for idx, step in enumerate(tu.steps, start=1):
                    step_rows.append(
                        {
                            "#": idx,
                            "Paso": step.step,
                            "Proveedor": step.provider,
                            "Modelo": step.model,
                            "Estado": step.status,
                            "Input Tok": step.input_tokens,
                            "Output Tok": step.output_tokens,
                            "Thoughts Tok": step.thoughts_tokens,
                            "Total Tok": step.total_tokens,
                            "API Calls": step.api_calls,
                            "Cache Hit": step.cache_hit,
                            "Chars": step.chars,
                            "Cohere Units": step.search_units,
                            "Docs": step.docs_reranked,
                            "Latencia (ms)": step.latency_ms,
                        }
                    )
                st.dataframe(step_rows, use_container_width=True, hide_index=True)

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
            session_ids.insert(0, st.session_state.active_session_id)

        session_labels = {
            s.session_id: f"{s.session_id} ({s.num_turns}t | {s.active_territorio or 'sin prov.'})"
            for s in sessions
        }
        if st.session_state.active_session_id not in session_labels:
            session_labels[st.session_state.active_session_id] = f"{st.session_state.active_session_id} (nueva)"

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
                new_sid = f"ses-{uuid.uuid4().hex[:6]}"
                st.session_state.visitor_session_ids.insert(0, new_sid)
                st.session_state.active_session_id = new_sid
                st.rerun()
        with col_del:
            if st.button("Limpiar Sesión", use_container_width=True):
                service.reset_or_delete_session(st.session_state.active_session_id)
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

        st.divider()
        st.subheader("3. Contacto y Privacidad")
        st.markdown(
            "¿Tienes dudas, comentarios, consejos o has detectado algún error en esta demo?"
        )
        st.link_button(
            f"✉️ {CONTACT_EMAIL}",
            f"mailto:{CONTACT_EMAIL}?subject=Comentarios%20sobre%20AiHoreca",
            use_container_width=True,
        )
        st.markdown(
            "- **No introduzcas datos personales** (nombre, DNI/NIE, empresa o teléfono).\n"
            "- **Auditorías de calidad:** las consultas se recogen para mejorar el sistema.\n"
            f"- **Borrado automático:** eliminación tras **{SESSION_RETENTION_DAYS} días** de inactividad (RGPD)."
        )
        st.caption("Demo orientativa • Contrasta siempre la información con un profesional.")

    return st.session_state.active_session_id


def _render_demo_privacy_banner(expanded: bool) -> None:
    """Renderiza el aviso de versión demo, privacidad (90 días) y contacto, plegado tras el primer turno."""
    with st.expander(
        f"⚠️ Aviso Demo, Privacidad (borrado en {SESSION_RETENTION_DAYS} días) y Contacto",
        expanded=expanded,
    ):
        st.warning(
            "- **Versión Demo (carácter orientativo):** esta aplicación es una demostración técnica y la información generada puede contener errores u omisiones. **Contrasta siempre** cualquier dato salarial o criterio jurídico con un profesional cualificado (abogado laboralista, graduado social o asesoría) y con el boletín oficial vigente.\n"
            f"- **Privacidad y auditorías (borrado en {SESSION_RETENTION_DAYS} días):** te aconsejamos **no introducir datos personales** (nombres, DNI/NIE, datos de empresa, teléfonos, etc.). La información introducida se recoge únicamente para mejorar el sistema mediante auditorías de calidad y se **elimina automáticamente transcurridos {SESSION_RETENTION_DAYS} días** de inactividad (art. 5.1.e RGPD).\n"
            f"- **Contacto y sugerencias:** para preguntas, consejos, comentarios o avisos de mejora, puedes escribir a [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}?subject=Comentarios%20sobre%20AiHoreca) *(también disponible en el panel izquierdo)*."
        )


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
    supported_territories_str = service.get_supported_territories_display()
    st.caption(
        "Estatuto de los Trabajadores (BOE) • VI Acuerdo Laboral Estatal de Hostelería (ALEH VI) • "
        f"Convenios Colectivos indexados: {supported_territories_str}"
    )

    persisted_turns = service.get_session_turns(active_sid)
    session_telemetry: dict[int, TurnExecutionPayload] = st.session_state.turn_telemetry_cache.setdefault(
        active_sid, {}
    )

    _render_demo_privacy_banner(expanded=not persisted_turns)

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
            with st.status("Analizando consulta laboral...", expanded=True) as status:
                def _on_node_update(node_name: str, update: dict[str, Any]) -> None:
                    label, detail = _format_stream_node_event(node_name, update)
                    status.update(label=label, state="running")
                    st.markdown(detail)

                payload = service.process_chat_turn(
                    session_id=active_sid,
                    query=user_input,
                    on_node_event=_on_node_update,
                )
                updated_turns = service.get_session_turns(active_sid)
                latest_turn_idx = int(updated_turns[-1]["turn_index"]) if updated_turns else 1
                session_telemetry[latest_turn_idx] = payload
                status.update(
                    label=f"Respuesta lista ({payload.total_latency_ms:.0f} ms)",
                    state="complete",
                    expanded=False,
                )

            st.markdown(payload.final_response)
            _render_turn_xray_panel(payload)
        st.rerun()


if __name__ == "__main__":
    main()
