"""Tests unitarios deterministas para la remediación de la auditoría (Puntos 1, 2, 3 y 4).

Cubre:
1. Punto 1: Detección de comparativa global / indiferencia provincial y respuesta pedagógica con suelo estatutario (ET).
2. Punto 2: Cuarentena de 24h para modelos con 404 NOT_FOUND en la cadena de failover.
3. Punto 3: Atajo determinista CPU para saludos y despedidas (0 tokens, latencia <1ms) y control de thoughts en enrutamiento.
4. Punto 4: Creación perezosa de sesiones en SQLite (sin sesiones fantasma de 0 turnos en visitas web) y timeout por intento.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from src.app.service import AiHorecaEngineService
from src.graph.memory import SessionMemoryStore
from src.graph.nodes import (
    _MODEL_COOLDOWN_UNTIL,
    _NOT_FOUND_COOLDOWN_SECONDS,
    AiHorecaGraphNodes,
    _call_structured_llm,
)
from src.graph.state import LegalGraphState, QueryAnalysisSchema, SessionContextSlots


class TestAuditRemediation(unittest.TestCase):
    """Batería de tests de remediación de auditoría (Puntos 1 a 4)."""

    def setUp(self) -> None:
        self.nodes = AiHorecaGraphNodes()
        self.mem = SessionMemoryStore()

    def tearDown(self) -> None:
        # Limpiar cooldowns para evitar contaminación entre tests
        _MODEL_COOLDOWN_UNTIL.clear()

    # -------------------------------------------------------------------------
    # Punto 3: Atajo determinista CPU para saludos y despedidas (0 tokens)
    # -------------------------------------------------------------------------
    def test_greeting_deterministic_cpu_shortcut(self) -> None:
        """Verifica que saludos y despedidas aislados sin aclaración pendiente cortocircuitan en CPU."""
        greetings = ["hola", "¡Hola!", "buenos días", "Buenas tardes", "adiós", "muchas gracias", "gracias"]
        for g in greetings:
            state: LegalGraphState = {
                "session_id": "ses-test-greeting",
                "query": g,
                "session_slots": SessionContextSlots().model_dump(),
                "conversation_history": [],
            }
            res = self.nodes.analyze_and_route_node(state)
            self.assertEqual(res["intent"], "HOSPITALITY_CORE")
            self.assertFalse(res["requiere_consulta_normativa_bbdd"])
            self.assertEqual(res["route_decision"], "ask_clarification")
            self.assertEqual(res["models_used"].get("analyze_and_route"), "cpu_deterministic_shortcut")
            self.assertFalse(res.get("usage_events"))  # 0 llamadas API, 0 tokens

            # Verificar que ask_clarification_node devuelve el mensaje de bienvenida y no una petición de provincia
            state_clarif: LegalGraphState = {**state, **res}
            clarif_res = self.nodes.ask_clarification_node(state_clarif)
            body = clarif_res["final_response"]
            self.assertIn("¡Hola! Soy AiHoreca", body)
            self.assertNotIn("necesito saber en qué provincia trabajas", body.lower())

    def test_greeting_with_pending_clarification_does_not_shortcut(self) -> None:
        """Verifica que si había una consulta bloqueada pendiente, el saludo NO activa el atajo determinista."""
        state: LegalGraphState = {
            "session_id": "ses-test-pending",
            "query": "hola",
            "session_slots": SessionContextSlots(
                pending_clarification_query="¿Cuántos días de vacaciones me corresponden?"
            ).model_dump(),
            "conversation_history": [],
        }
        # Mock de _call_structured_llm para no consumir API en test unitario
        mock_analysis = QueryAnalysisSchema(
            intent="HOSPITALITY_CORE",
            motivo_clasificacion="Aclaración pendiente",
            provincias_detectadas_en_turno_actual=[],
            cambio_de_puesto_en_turno_actual=False,
            requiere_consulta_normativa_bbdd=True,
            requiere_convenio_provincial=True,
            requiere_puesto_o_categoria=False,
            consulta_autocontenida_resuelta="¿Cuántos días de vacaciones me corresponden?",
            query_busqueda_hibrida="vacaciones",
        )
        with patch("src.graph.nodes._call_structured_llm", return_value=(mock_analysis, "mock_model")):
            res = self.nodes.analyze_and_route_node(state)
            self.assertNotEqual(res["models_used"].get("analyze_and_route"), "cpu_deterministic_shortcut")

    # -------------------------------------------------------------------------
    # Punto 1: Comparativas globales e indiferencia provincial
    # -------------------------------------------------------------------------
    def test_global_comparison_detection_and_pedagogical_baseline(self) -> None:
        """Verifica que 'de cada provincia' o 'da igual la provincia' ofrece el suelo del ET y pide 2-4 provincias."""
        queries = [
            "Hazme una tabla comparativa con los dias de vacaciones d ecada provincia",
            "Da igual en que provincia trabaje. Te he pedido una tabla comparativa",
            "Quiero comparar todas las provincias de hostelería",
        ]
        for q in queries:
            state: LegalGraphState = {
                "session_id": "ses-test-global-comp",
                "query": q,
                "session_slots": SessionContextSlots().model_dump(),
                "conversation_history": [],
                "es_comparativa_global": True,
                "territorio_normalizado": None,
                "requiere_convenio_provincial": True,
                "requiere_consulta_normativa_bbdd": True,
            }
            clarif_res = self.nodes.ask_clarification_node(state)
            body = clarif_res["final_response"]

            # Debe explicar la estructura de 50+ convenios
            self.assertTrue("50 convenios" in body or "más de 50" in body)
            # Debe citar el suelo estatal del Estatuto de los Trabajadores
            self.assertIn("Art. 38.1 ET", body)
            self.assertIn("30 días naturales", body)
            self.assertIn("Art. 34.1 ET", body)
            self.assertIn("40 horas semanales", body)
            self.assertIn("Art. 31 ET", body)
            self.assertIn("VI ALEH", body)
            # Debe invitar a seleccionar libremente las provincias a comparar
            self.assertIn("qué provincias o territorios deseas comparar", body)
            # NO debe repetir el template rígido de trabajador individual
            self.assertNotIn("necesito saber en qué provincia trabajas", body.lower())

    # -------------------------------------------------------------------------
    # Punto 2: Cuarentena 24h para modelos con 404 NOT_FOUND
    # -------------------------------------------------------------------------
    def test_model_404_quarantined_for_24_hours(self) -> None:
        """Verifica que si un modelo devuelve 404 NOT_FOUND, queda en cuarentena de 24h y no se reintenta."""
        model_bad = "gemini-model-inexistente"
        model_good = "gemini-model-bueno"

        mock_client = MagicMock()
        # Primer modelo lanza 404 NOT_FOUND; segundo modelo responde OK
        mock_response_good = MagicMock()
        mock_response_good.parsed = QueryAnalysisSchema(
            intent="HOSPITALITY_CORE",
            motivo_clasificacion="OK",
            provincias_detectadas_en_turno_actual=["madrid"],
            cambio_de_puesto_en_turno_actual=False,
            requiere_consulta_normativa_bbdd=True,
            requiere_convenio_provincial=True,
            requiere_puesto_o_categoria=False,
            consulta_autocontenida_resuelta="consulta",
            query_busqueda_hibrida="consulta",
        )
        mock_response_good.usage_metadata = None

        def mock_generate_content(model, contents, config):
            if model == model_bad:
                raise RuntimeError("404 NOT_FOUND. models/gemini-model-inexistente is not found for API version v1")
            return mock_response_good

        mock_client.models.generate_content.side_effect = mock_generate_content

        with patch("src.graph.nodes.get_genai_client", return_value=mock_client), patch(
            "src.graph.nodes.GEMINI_FALLBACK_MODELS", (model_good,)
        ):
            usage_events = []
            obj, used = _call_structured_llm(
                prompt="test prompt",
                schema=QueryAnalysisSchema,
                preferred_model=model_bad,
                system_instruction="sys",
                usage_out=usage_events,
            )
            self.assertEqual(used, model_good)
            # El modelo con 404 debe tener cooldown de ~86400s
            cd_until = _MODEL_COOLDOWN_UNTIL.get(model_bad, 0.0)
            self.assertGreater(cd_until, time.time() + 80000.0)

            # Verificar que el status de telemetría reporta failed_404
            fail_event = next(e for e in usage_events if e["model"] == model_bad)
            self.assertEqual(fail_event["status"], "failed_404")

    # -------------------------------------------------------------------------
    # Punto 4: Creación perezosa de sesiones y purga de sesiones de 0 turnos
    # -------------------------------------------------------------------------
    def test_lazy_session_does_not_insert_on_read(self) -> None:
        """Verifica que leer slots o historial de una sesión nueva en memoria NO crea fila en SQLite."""
        new_sid = "ses-test-lazy-ghost"
        # Asegurarse de que no existe previamente
        self.mem.delete_session(new_sid)

        # 1. get_slots no debe insertar
        slots = self.mem.get_slots(new_sid)
        self.assertIsInstance(slots, SessionContextSlots)
        sessions_in_db = [s["session_id"] for s in self.mem.list_sessions()]
        self.assertNotIn(new_sid, sessions_in_db)

        # 2. get_compact_history no debe insertar
        history = self.mem.get_compact_history(new_sid)
        self.assertEqual(history, [])
        sessions_in_db = [s["session_id"] for s in self.mem.list_sessions()]
        self.assertNotIn(new_sid, sessions_in_db)

    def test_purge_inactive_sessions_removes_orphaned_zero_turn_sessions(self) -> None:
        """Verifica que purge_inactive_sessions elimina sesiones fantasma de 0 turnos."""
        ghost_sid = "ses-test-ghost-zero-turns"
        self.mem.ensure_session(ghost_sid, title="Sesión Fantasma")

        # Comprobar que existe con 0 turnos
        all_sessions = {s["session_id"]: s["num_turns"] for s in self.mem.list_sessions()}
        self.assertIn(ghost_sid, all_sessions)
        self.assertEqual(all_sessions[ghost_sid], 0)

        # Ejecutar purga
        purged = self.mem.purge_inactive_sessions(retention_days=90)
        self.assertGreaterEqual(purged, 1)

        # Comprobar que ya no existe en la base de datos
        remaining_sids = [s["session_id"] for s in self.mem.list_sessions()]
        self.assertNotIn(ghost_sid, remaining_sids)

    def test_ui_init_state_generates_lazy_session_id_without_name_error(self) -> None:
        """Verifica que ui._init_ui_state genera ses-<hex> sin NameError ni escritura previa en SQLite."""
        from src.app import ui

        class _FakeSessionState(dict):
            def __getattr__(self, name: str):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name: str, value) -> None:
                self[name] = value

        fake_state = _FakeSessionState()
        mock_service = MagicMock()
        with patch.object(ui.st, "session_state", fake_state):
            ui._init_ui_state(mock_service)

        self.assertEqual(fake_state.turn_telemetry_cache, {})
        self.assertEqual(len(fake_state.visitor_session_ids), 1)
        sid = fake_state.visitor_session_ids[0]
        self.assertTrue(sid.startswith("ses-"))
        self.assertEqual(len(sid), 10)
        self.assertEqual(fake_state.active_session_id, sid)
        mock_service.create_session.assert_not_called()

    def test_ui_sidebar_renders_three_collapsible_expanders(self) -> None:
        """Verifica que _render_sidebar envuelve sus 3 bloques en st.expander desplegables."""
        from contextlib import nullcontext
        from src.app import ui

        class _FakeSessionState(dict):
            def __getattr__(self, name: str):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name: str, value) -> None:
                self[name] = value

        fake_state = _FakeSessionState(
            visitor_session_ids=["ses-demo01"],
            active_session_id="ses-demo01",
            turn_telemetry_cache={},
        )
        expander_titles: list[tuple[str, bool]] = []

        fake_st = MagicMock()
        fake_st.session_state = fake_state
        fake_st.sidebar = nullcontext()
        fake_st.expander.side_effect = lambda title, expanded=True: (
            expander_titles.append((title, expanded)) or nullcontext()
        )
        fake_st.selectbox.return_value = "ses-demo01"
        fake_st.columns.return_value = (nullcontext(), nullcontext())
        fake_st.button.return_value = False

        mock_service = MagicMock()
        mock_service.list_visitor_sessions.return_value = []
        mock_service.get_session_slots.return_value = SessionContextSlots()

        with patch.object(ui, "st", fake_st):
            active_sid = ui._render_sidebar(mock_service)

        self.assertEqual(active_sid, "ses-demo01")
        self.assertEqual(len(expander_titles), 3)
        self.assertTrue(expander_titles[0][0].startswith("1. Gestión de Tu Sesión"))
        self.assertTrue(expander_titles[1][0].startswith("2. Working Memory"))
        self.assertTrue(expander_titles[2][0].startswith("3. Contacto y Privacidad"))
        self.assertTrue(all(exp is True for _, exp in expander_titles))


if __name__ == "__main__":
    unittest.main()
