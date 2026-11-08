"""Compilación del grafo agéntico en LangGraph y ejecución con Memoria Multi-Sesión (SQLite) para AiHoreca."""

from __future__ import annotations

import json
import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph

from src.graph.memory import SessionMemoryStore
from src.graph.nodes import AiHorecaGraphNodes
from src.graph.state import LegalGraphState, RouteDecision
from src.retrieval.vector_store import AiHorecaVectorStore

logger = logging.getLogger(__name__)


def _route_after_analysis(state: LegalGraphState) -> RouteDecision:
    """Arista condicional determinista tras el nodo de Guardrail + Gatekeeper + Traductor ALEH."""
    decision = state.get("route_decision", "block_out_of_scope")
    if decision in ("block_out_of_scope", "ask_clarification", "parallel_retrieve"):
        return decision
    return "block_out_of_scope"


def build_aihoreca_graph(vector_store: Optional[AiHorecaVectorStore] = None):
    """Construye y compila la máquina de estados (StateGraph) de AiHoreca."""
    nodes = AiHorecaGraphNodes(vector_store=vector_store)

    workflow = StateGraph(LegalGraphState)

    workflow.add_node("analyze_and_route", nodes.analyze_and_route_node)
    workflow.add_node("block_out_of_scope", nodes.block_out_of_scope_node)
    workflow.add_node("ask_clarification", nodes.ask_clarification_node)
    workflow.add_node("parallel_retrieve", nodes.parallel_retrieval_node)
    workflow.add_node("legal_synthesis_and_audit", nodes.legal_synthesis_and_audit_node)

    workflow.add_edge(START, "analyze_and_route")

    workflow.add_conditional_edges(
        "analyze_and_route",
        _route_after_analysis,
        {
            "block_out_of_scope": "block_out_of_scope",
            "ask_clarification": "ask_clarification",
            "parallel_retrieve": "parallel_retrieve",
        },
    )

    workflow.add_edge("block_out_of_scope", END)
    workflow.add_edge("ask_clarification", END)

    workflow.add_edge("parallel_retrieve", "legal_synthesis_and_audit")
    workflow.add_edge("legal_synthesis_and_audit", END)

    return workflow.compile()


def run_legal_query(
    query: str,
    session_id: Optional[str] = None,
    memory_store: Optional[SessionMemoryStore] = None,
    conversation_history: Optional[list[dict[str, str]]] = None,
    compiled_graph=None,
) -> LegalGraphState:
    """Ejecuta una consulta laboral a través del grafo agéntico con soporte de memoria persistente por `session_id`."""
    graph = compiled_graph or build_aihoreca_graph()
    store = memory_store or (SessionMemoryStore() if session_id else None)

    if store and session_id:
        sid = store.ensure_session(session_id)
        slots = store.get_slots(sid).model_dump()
        history = (
            conversation_history
            if conversation_history is not None
            else store.get_compact_history(sid, max_turns=4)
        )
    else:
        sid = session_id or "stateless-session"
        slots = {}
        history = conversation_history or []

    initial_state: LegalGraphState = {
        "session_id": sid,
        "query": query,
        "session_slots": slots,
        "conversation_history": history,
        "node_timings_ms": {},
        "models_used": {},
    }

    result_state: LegalGraphState = graph.invoke(initial_state)

    if store and session_id:
        updated_slots = store.save_turn_and_update_slots(sid, result_state)
        result_state["session_slots"] = updated_slots.model_dump()

    return result_state


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    vstore = AiHorecaVectorStore()
    mem = SessionMemoryStore()

    # Limpiamos las dos sesiones de prueba para que el benchmark sea determinista y reproducible
    mem.delete_session("ses-madrid-friegaplatos")
    mem.delete_session("ses-leon-pinche")

    app = build_aihoreca_graph(vector_store=vstore)

    interleaved_turns = [
        {
            "step": "TURNO 1 [Sesión A: ses-madrid-friegaplatos] -> Falta provincia (Dispara Gatekeeper HITL y guarda pending_query)",
            "session_id": "ses-madrid-friegaplatos",
            "query": "Soy friegaplatos a jornada completa, ¿cuánto tengo que cobrar de sueldo base y cuántos días de vacaciones me tocan?",
        },
        {
            "step": "TURNO 2 [Sesión B: ses-leon-pinche] -> Sesión paralela independiente en León (Verifica aislamiento entre sesiones)",
            "session_id": "ses-leon-pinche",
            "query": "Soy ayudante de cocina (pinche) en un restaurante en León. ¿Cuál es mi salario base según las tablas salariales?",
        },
        {
            "step": "TURNO 3 [Sesión A: ses-madrid-friegaplatos] -> Usuario responde solo 'Trabajo en Madrid' (Resuelve HITL pendiente del Turno 1)",
            "session_id": "ses-madrid-friegaplatos",
            "query": "Trabajo en Madrid.",
        },
        {
            "step": "TURNO 4 [Sesión A: ses-madrid-friegaplatos] -> Seguimiento elíptico sin repetir puesto ni provincia ('nocturnidad')",
            "session_id": "ses-madrid-friegaplatos",
            "query": "¿Y cómo se regula o cuánto se paga por nocturnidad en mi caso?",
        },
        {
            "step": "TURNO 5 [Sesión B: ses-leon-pinche] -> Cambio hipotético a provincia no soportada ('Valencia') (Verifica bloqueo sin State Poisoning)",
            "session_id": "ses-leon-pinche",
            "query": "¿Y si me ofrecen irme de cocinero a un hotel en Valencia, cuánto cobraría allí?",
        },
        {
            "step": "TURNO 6 [Sesión B: ses-leon-pinche] -> Retorno a su restaurante actual en León sin nombrar León (Verifica que el bloqueo de Valencia no corrompió el slot 'leon')",
            "session_id": "ses-leon-pinche",
            "query": "Vale, quedándome aquí en mi restaurante actual, ¿cuántos días de vacaciones anuales y permisos tengo?",
        },
    ]

    for item in interleaved_turns:
        print("\n" + "=" * 100)
        print(item["step"])
        print(f"SESSION_ID: {item['session_id']} | RAW QUERY: {item['query']}")
        print("=" * 100)

        out = run_legal_query(
            query=item["query"],
            session_id=item["session_id"],
            memory_store=mem,
            compiled_graph=app,
        )

        print(f"-> EFFECTIVE QUERY RESUELTA : {out.get('effective_query')}")
        print(f"-> INTENT: {out.get('intent')} | ROUTE: {out.get('route_decision')}")
        print(f"-> PROVINCIA: {out.get('provincia_detectada')} | TERRITORIO: {out.get('territorio_normalizado')}")
        print(f"-> CATEGORÍA ALEH VI: {out.get('categoria_profesional_aleh')}")
        print(f"-> SLOTS PERSISTIDOS POST-TURNO: {json.dumps(out.get('session_slots'), ensure_ascii=False)}")
        print(f"-> TIMINGS (ms): {out.get('node_timings_ms')}")
        print(f"-> AUDIT REPORT: {json.dumps(out.get('audit_report'), ensure_ascii=False)}")
        print("\n--- RESPUESTA AL USUARIO ---\n")
        print(out.get("final_response", ""))

    print("\n" + "=" * 100)
    print("RESUMEN DE SESIONES PERSISTIDAS EN SQLITE (data/sessions/aihoreca_sessions.db)")
    print("=" * 100)
    for s_info in mem.list_sessions():
        print(json.dumps(s_info, ensure_ascii=False, indent=2))

    vstore.close()
