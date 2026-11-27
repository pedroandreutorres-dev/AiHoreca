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

    # -------------------------------------------------------------------------
    # Mejoras de auditoría 27/11: objeciones conversacionales, topónimos y costes
    # -------------------------------------------------------------------------
    def test_conversational_objection_uses_direct_response_instead_of_greeting(self) -> None:
        """Objeción meta con requiere_consulta_normativa_bbdd=False responde contextualmente sin saludo enlatado."""
        mock_analysis = QueryAnalysisSchema(
            intent="HOSPITALITY_CORE",
            motivo_clasificacion="Objeción conversacional sobre suposición previa sin consulta normativa nueva.",
            provincias_detectadas_en_turno_actual=[],
            cambio_de_puesto_en_turno_actual=False,
            requiere_consulta_normativa_bbdd=False,
            requiere_convenio_provincial=True,
            requiere_puesto_o_categoria=False,
            respuesta_conversacional_directa=(
                "Lo asumí por continuidad con la comarca mencionada previamente; "
                "si te refieres a otra provincia, indícamela y reviso su convenio."
            ),
            consulta_autocontenida_resuelta="Aclaración sobre territorio inferido en el turno anterior.",
            query_busqueda_hibrida="aclaracion territorio",
        )
        state: LegalGraphState = {
            "session_id": "ses-test-objection",
            "query": "¿Por qué asumes esa localidad y no otra con el mismo nombre?",
            "session_slots": SessionContextSlots(active_territorio="leon").model_dump(),
            "conversation_history": [
                {"user_query": "Ponferrada", "assistant_summary": "Convenio de León aplicado."}
            ],
        }
        with patch("src.graph.nodes._call_structured_llm", return_value=(mock_analysis, "mock_model")):
            res = self.nodes.analyze_and_route_node(state)

        self.assertEqual(res["route_decision"], "ask_clarification")
        self.assertFalse(res["requiere_consulta_normativa_bbdd"])
        self.assertTrue(res["respuesta_conversacional_directa"])

        clarif_res = self.nodes.ask_clarification_node({**state, **res})
        body = clarif_res["final_response"]
        self.assertIn("continuidad con la comarca", body)
        self.assertNotIn("¡Hola! Soy AiHoreca", body)

        # Garantizar que NO ensucia pending_clarification_query ni pierde active_territorio
        self.mem.save_turn_and_update_slots(
            session_id="ses-test-objection",
            state={
                **state,
                **res,
                **clarif_res,
            },
        )
        saved_slots = self.mem.get_slots("ses-test-objection")
        self.assertIsNone(saved_slots.pending_clarification_query)
        self.mem.delete_session("ses-test-objection")

    def test_ambiguous_toponym_option_b_with_and_without_context(self) -> None:
        """Opción B: inyecta aviso si encaja con territorio activo; si no encaja, evita heredar y pregunta."""
        from src.graph.state import LegalCitation, LegalSynthesisSchema
        from src.retrieval.schemas import RetrievedDocument

        # Subcaso B.1: hay territorio activo compatible -> mantiene provincia y añade aviso determinista
        aviso_compat = (
            "Asumo la localidad homónima dentro de la provincia activa por el contexto previo; "
            "si te refieres a otra provincia, indícamela para revisar su convenio."
        )
        analysis_compat = QueryAnalysisSchema(
            intent="HOSPITALITY_CORE",
            motivo_clasificacion="Topónimo homónimo compatible con el territorio activo en sesión.",
            provincias_detectadas_en_turno_actual=["leon"],
            cambio_de_puesto_en_turno_actual=False,
            requiere_consulta_normativa_bbdd=True,
            requiere_convenio_provincial=True,
            requiere_puesto_o_categoria=False,
            aviso_ambiguedad_geografica=aviso_compat,
            consulta_autocontenida_resuelta="Descanso mínimo en jornada partida en localidad compatible.",
            query_busqueda_hibrida="jornada partida descanso leon",
        )
        state_b1: LegalGraphState = {
            "session_id": "ses-test-ambig-b1",
            "query": "¿Y en esa localidad homónima?",
            "session_slots": SessionContextSlots(active_territorio="leon").model_dump(),
            "conversation_history": [],
        }
        with patch("src.graph.nodes._call_structured_llm", return_value=(analysis_compat, "mock_model")):
            route_b1 = self.nodes.analyze_and_route_node(state_b1)
        self.assertEqual(route_b1["route_decision"], "parallel_retrieve")
        self.assertEqual(route_b1["territorio_normalizado"], "leon")
        self.assertEqual(route_b1["aviso_ambiguedad_geografica"], aviso_compat)

        mock_synth = LegalSynthesisSchema(
            jerarquia_normativa_aplicada="ET + ALEH + Convenio Provincial",
            citas_normativas=[
                LegalCitation(
                    norma="Convenio Colectivo de León",
                    capa_normativa="convenio_provincial",
                    articulo_o_tabla="Art. 19",
                    extracto_clave="Al menos 2 horas de descanso.",
                )
            ],
            cifras_y_datos_criticos=["2 horas"],
            calculos_derivados=[],
            respuesta_final_markdown="El descanso mínimo en jornada partida es de **2 horas** (Art. 19).",
        )
        fake_doc = RetrievedDocument(
            chunk_id="c1",
            source_file="convenio_leon_texto.md",
            norma_titulo="Convenio León",
            capa_normativa="convenio_provincial",
            territorio="leon",
            titulo_o_capitulo="Jornada",
            articulo="Art. 19",
            tipo_contenido="articulo",
            text="Se entenderá por jornada partida aquella con descanso de al menos 2 horas.",
            score=0.9,
        )
        with patch("src.graph.nodes._call_structured_llm", return_value=(mock_synth, "mock_model")):
            synth_b1 = self.nodes.synthesize_response_node({
                **state_b1,
                **route_b1,
                "retrieved_docs": [fake_doc],
            })
        self.assertTrue(synth_b1["final_response"].startswith("> **Nota de desambiguación territorial:**"))
        self.assertIn("2 horas", synth_b1["final_response"])

        # Subcaso B.2: topónimo ambiguo sin coincidencia en la provincia activa -> no hereda slot previo
        aviso_incompat = (
            "Existen varias localidades homónimas en distintas provincias de España. "
            "¿A qué provincia concreta te refieres?"
        )
        analysis_incompat = QueryAnalysisSchema(
            intent="HOSPITALITY_CORE",
            motivo_clasificacion="Topónimo homónimo sin encaje en la provincia activa.",
            provincias_detectadas_en_turno_actual=[],
            cambio_de_puesto_en_turno_actual=False,
            requiere_consulta_normativa_bbdd=True,
            requiere_convenio_provincial=True,
            requiere_puesto_o_categoria=False,
            aviso_ambiguedad_geografica=aviso_incompat,
            consulta_autocontenida_resuelta="Descanso mínimo en localidad homónima sin provincia.",
            query_busqueda_hibrida="jornada partida descanso",
        )
        state_b2: LegalGraphState = {
            "session_id": "ses-test-ambig-b2",
            "query": "¿Y en esa otra localidad?",
            "session_slots": SessionContextSlots(active_territorio="madrid").model_dump(),
            "conversation_history": [],
        }
        with patch("src.graph.nodes._call_structured_llm", return_value=(analysis_incompat, "mock_model")):
            route_b2 = self.nodes.analyze_and_route_node(state_b2)
        self.assertEqual(route_b2["route_decision"], "ask_clarification")
        self.assertIsNone(route_b2["territorio_normalizado"])
        clarif_b2 = self.nodes.ask_clarification_node({**state_b2, **route_b2})
        self.assertIn("Desambiguación Territorial Requerida", clarif_b2["final_response"])
        self.assertIn("distintas provincias", clarif_b2["final_response"])

    def test_pricing_markdoc_parser_date_tiers_and_turn_cost_calculation(self) -> None:
        """Verifica el parseo Markdoc oficial con cambios por fecha y cálculo real + Tier Lite + Tier Flash."""
        from datetime import date
        from src.app.pricing import PricingCatalog, compute_turn_cost, parse_official_gemini_pricing_markdown
        from src.graph.state import StepUsageRecord, TurnUsageSummary

        sample_md = """
## Gemini 3.1 Flash-Lite  {% id="gemini-3.1-flash-lite" %}
*[`gemini-3.1-flash-lite`](/docs/models/gemini-3.1-flash-lite)*
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.25 (text / image / video)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $1.50

  ---

  * Context caching price
  * Not available
  * $0.025 (text / image / video)
  $1.00 / 1,000,000 tokens per hour (storage price)
  {% /table %}
  {% /tab %}
{% /tabs %}

## Gemini 3.8 Flash  {% id="gemini-3.8-flash" %}
*[`gemini-3.8-flash`](/docs/models/gemini-3.8-flash)*
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.75 through December 31, 2026.
  $1.50 starting January 1, 2027.

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $3.75 through December 31, 2026.
  $7.50 starting January 1, 2027.

  ---

  * Context caching price
  * Free of charge
  * $0.075 through December 31, 2026.
  $0.15 starting January 1, 2027.
  $0.50 / 1,000,000 tokens per hour (storage price) through December 31, 2026.
  {% /table %}
  {% /tab %}
{% /tabs %}

## Gemini 3.5 Flash-Lite  {% id="gemini-3.5-flash-lite" %}
*[`gemini-3.5-flash-lite`](/docs/models/gemini-3.5-flash-lite)*
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.30

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $2.50
  {% /table %}
  {% /tab %}
{% /tabs %}

## Gemini Embedding  {% id="gemini-embedding" %}
*[`gemini-embedding-001`](/docs/models/gemini-embedding-001)*
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.15
  {% /table %}
  {% /tab %}
{% /tabs %}
"""
        rates_2026 = parse_official_gemini_pricing_markdown(sample_md, as_of=date(2026, 11, 27))
        rates_2027 = parse_official_gemini_pricing_markdown(sample_md, as_of=date(2027, 2, 1))
        self.assertAlmostEqual(rates_2026["gemini-3.8-flash"].input_per_1m, 0.75)
        self.assertAlmostEqual(rates_2026["gemini-3.8-flash"].output_per_1m, 3.75)
        self.assertAlmostEqual(rates_2027["gemini-3.8-flash"].input_per_1m, 1.50)
        self.assertAlmostEqual(rates_2027["gemini-3.8-flash"].output_per_1m, 7.50)
        self.assertAlmostEqual(rates_2026["gemini-3.5-flash-lite"].cached_input_per_1m, 0.03)
        self.assertAlmostEqual(rates_2026["gemini-embedding-001"].input_per_1m, 0.15)

        catalog = PricingCatalog(
            checked_at="2026-11-27T18:00:00+00:00",
            as_of_date="2026-11-27",
            source="live_official",
            usd_to_eur=0.92,
            cohere_usd_per_1k_units=2.0,
            models=rates_2026,
        )
        usage = TurnUsageSummary(
            input_tokens=1_000_000,
            output_tokens=100_000,
            thoughts_tokens=50_000,
            cached_tokens=200_000,
            total_tokens=1_150_000,
            embedding_calls=1,
            embedding_chars=400,
            cohere_calls=1,
            cohere_search_units=1,
            steps=[
                StepUsageRecord(
                    step="analyze_and_route",
                    provider="gemini_llm",
                    model="gemini-3.1-flash-lite",
                    input_tokens=200_000,
                    cached_tokens=200_000,
                    output_tokens=50_000,
                    api_calls=1,
                ),
                StepUsageRecord(
                    step="embed_state_layers",
                    provider="gemini_embedding",
                    model="gemini-embedding-001",
                    api_calls=1,
                    cache_hit=False,
                    chars=400,
                ),
                StepUsageRecord(
                    step="embed_territory",
                    provider="gemini_embedding",
                    model="gemini-embedding-001",
                    api_calls=0,
                    cache_hit=True,
                    chars=400,
                ),
                StepUsageRecord(
                    step="synthesize_response",
                    provider="gemini_llm",
                    model="gemini-3.5-flash-lite",
                    input_tokens=800_000,
                    output_tokens=50_000,
                    thoughts_tokens=50_000,
                    api_calls=1,
                ),
            ],
        )
        cost = compute_turn_cost(usage, catalog)
        # Router: 200k cached * 0.025/1M + 50k out * 1.50/1M = 0.005 + 0.075 = 0.080
        # Synth Lite: 800k in * 0.30/1M + (50k+50k) out * 2.50/1M = 0.240 + 0.250 = 0.490
        # Emb: ceil(400/4)=100 tok * 0.15/1M = 0.000015 -> Cohere: 1 * 0.002 = 0.002
        self.assertAlmostEqual(cost.coste_gemini_real_usd, 0.57, places=5)
        self.assertAlmostEqual(cost.coste_gemini_tier_lite_usd, 0.57, places=5)
        # Synth Flash: 800k * 0.75/1M + 100k * 3.75/1M = 0.600 + 0.375 = 0.975 -> + 0.080 = 1.055
        self.assertAlmostEqual(cost.coste_gemini_tier_flash_usd, 1.055, places=5)
        self.assertEqual(cost.embedding_tokens_est, 100)
        self.assertAlmostEqual(cost.coste_cohere_usd, 0.002, places=6)
        self.assertAlmostEqual(cost.coste_total_tier_lite_usd, 0.572015, places=6)
        self.assertAlmostEqual(cost.coste_total_tier_flash_usd, 1.057015, places=6)

    def test_cost_analytics_csv_upsert_preserves_historical_purged_sessions(self) -> None:
        """Verifica exportación idempotente sin PII a historico_sesiones_costes y historico_turnos_costes."""
        import csv
        import tempfile
        from pathlib import Path
        from src.app.audit_runner import export_cost_analytics_csvs
        from src.app.pricing import load_pricing_catalog
        from src.graph.memory import SessionMemoryStore
        from src.graph.state import AuditReport, StepUsageRecord, TurnUsageSummary

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "sessions.db"
            sess_csv = tmp / "historico_sesiones_costes.csv"
            turn_csv = tmp / "historico_turnos_costes.csv"

            # Sembrar una fila histórica de sesión ya purgada hace meses
            sess_csv.write_text(
                "session_id,fecha_inicio,total_turnos,coste_total_tier_lite_usd\n"
                "ses-purgada,2026-01-10,2,0.015000\n",
                encoding="utf-8-sig",
            )

            mem = SessionMemoryStore(db_path=db_path)
            mem.save_turn_and_update_slots(
                session_id="ses-activa01",
                state={
                    "session_id": "ses-activa01",
                    "query": "Consulta secreta con DNI 12345678A",
                    "effective_query": "Consulta efectiva",
                    "intent": "HOSPITALITY_CORE",
                    "route_decision": "parallel_retrieve",
                    "territorio_normalizado": "madrid",
                    "oficio_coloquial_detectado": "camarero",
                    "final_response": "Respuesta completa confidencial.",
                    "audit_report": AuditReport(passed=True, has_explicit_citations=True).model_dump(),
                    "node_timings_ms": {"analyze_and_route": 1200.0, "synthesize_response": 2800.0},
                    "models_used": {
                        "analyze_and_route": "gemini-3.1-flash-lite",
                        "synthesize_response": "gemini-3.5-flash-lite",
                    },
                    "usage_events": [
                        StepUsageRecord(
                            step="analyze_and_route",
                            provider="gemini_llm",
                            model="gemini-3.1-flash-lite",
                            input_tokens=3000,
                            output_tokens=300,
                            api_calls=1,
                        ).model_dump(),
                        StepUsageRecord(
                            step="synthesize_response",
                            provider="gemini_llm",
                            model="gemini-3.5-flash-lite",
                            input_tokens=7000,
                            output_tokens=600,
                            api_calls=1,
                        ).model_dump(),
                    ],
                },
            )

            catalog = load_pricing_catalog(fetch_live=False, snapshot_path=tmp / "no_snap.json")
            res1 = export_cost_analytics_csvs(
                db_path=db_path,
                pricing_catalog=catalog,
                sessions_csv_path=sess_csv,
                turns_csv_path=turn_csv,
            )
            res2 = export_cost_analytics_csvs(
                db_path=db_path,
                pricing_catalog=catalog,
                sessions_csv_path=sess_csv,
                turns_csv_path=turn_csv,
            )

            self.assertEqual(res1["sessions_total_csv"], 2)
            self.assertEqual(res2["sessions_total_csv"], 2)
            self.assertEqual(res2["turns_total_csv"], 1)

            raw_sess_text = sess_csv.read_text(encoding="utf-8-sig")
            raw_turn_text = turn_csv.read_text(encoding="utf-8-sig")
            self.assertNotIn("12345678A", raw_sess_text)
            self.assertNotIn("12345678A", raw_turn_text)
            self.assertNotIn("Respuesta completa", raw_turn_text)

            with sess_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = {r["session_id"]: r for r in csv.DictReader(f)}
            self.assertIn("ses-purgada", rows)
            self.assertIn("ses-activa01", rows)
            self.assertEqual(rows["ses-activa01"]["territorio"], "madrid")
            self.assertGreater(float(rows["ses-activa01"]["coste_total_tier_lite_usd"]), 0.0)
            self.assertGreater(
                float(rows["ses-activa01"]["coste_total_tier_flash_usd"]),
                float(rows["ses-activa01"]["coste_total_tier_lite_usd"]),
            )

    def test_pricing_radar_detects_price_changes_new_models_and_upcoming_step_ups(self) -> None:
        """Verifica que el radar detecta cambios de tarifa, deprecación, modelos nuevos y subidas futuras."""
        from datetime import date
        from src.app.pricing import ModelRate, PricingCatalog, build_pricing_radar, parse_official_gemini_pricing_markdown

        sample_md = """
## Gemini 3.1 Flash-Lite  {% id="gemini-3.1-flash-lite" %}
*[`gemini-3.1-flash-lite`](/docs/models/gemini-3.1-flash-lite)*
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.20

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $1.50
  {% /table %}
  {% /tab %}
{% /tabs %}

## Gemini 3.8 Flash  {% id="gemini-3.8-flash" %}
*[`gemini-3.8-flash`](/docs/models/gemini-3.8-flash)*
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.75 through December 31, 2026.
  $1.50 starting January 1, 2027.

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $3.75 through December 31, 2026.
  $7.50 starting January 1, 2027.
  {% /table %}
  {% /tab %}
{% /tabs %}

## Gemini Old Legacy  {% id="gemini-old-legacy" %}
*[`gemini-old-legacy`](/docs/models/gemini-old-legacy)*
Warning: Gemini Old Legacy (`gemini-old-legacy`) is deprecated and will be shut down.
{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.10

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $0.40
  {% /table %}
  {% /tab %}
{% /tabs %}
"""
        parsed = parse_official_gemini_pricing_markdown(sample_md, as_of=date(2026, 11, 27))
        self.assertIn("Desde 2027-01-01", parsed["gemini-3.8-flash"].upcoming_change)
        self.assertTrue(parsed["gemini-old-legacy"].deprecated)
        self.assertFalse(parsed["gemini-3.1-flash-lite"].deprecated)

        prev_catalog = PricingCatalog(
            checked_at="2026-11-26T08:00:00+00:00",
            as_of_date="2026-11-26",
            source="live_official",
            models={
                "gemini-3.1-flash-lite": ModelRate(model_id="gemini-3.1-flash-lite", input_per_1m=0.25, output_per_1m=1.50, cached_input_per_1m=0.025),
                "gemini-retired": ModelRate(model_id="gemini-retired", input_per_1m=0.50, output_per_1m=2.00, cached_input_per_1m=0.05),
            },
        )
        radar = build_pricing_radar(
            prev_catalog=prev_catalog,
            new_models=parsed,
            live_parsed_ids=set(parsed.keys()),
            configured_models=["gemini-3.1-flash-lite", "gemini-old-legacy"],
        )
        self.assertTrue(radar.needs_attention)
        self.assertIn("gemini-3.8-flash", radar.new_models)
        self.assertIn("gemini-retired", radar.removed_models)
        self.assertTrue(any("gemini-3.1-flash-lite" in c for c in radar.price_changes))
        self.assertTrue(any("gemini-old-legacy" in d for d in radar.deprecated_configured_models))
        self.assertTrue(any("gemini-3.8-flash" in u for u in radar.upcoming_price_changes))

    def test_cost_dashboard_html_generation_is_self_contained_and_pii_free(self) -> None:
        """Verifica que generate_cost_dashboard_html crea un HTML sin CDNs externos ni texto libre/PII."""
        import tempfile
        from pathlib import Path
        from src.app.audit_runner import export_cost_analytics_csvs
        from src.app.cost_dashboard import generate_cost_dashboard_html
        from src.app.pricing import load_pricing_catalog
        from src.graph.memory import SessionMemoryStore
        from src.graph.state import AuditReport, StepUsageRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "sessions.db"
            sess_csv = tmp / "historico_sesiones_costes.csv"
            turn_csv = tmp / "historico_turnos_costes.csv"
            out_html = tmp / "dashboard_costes.html"

            mem = SessionMemoryStore(db_path=db_path)
            mem.save_turn_and_update_slots(
                session_id="ses-dash01",
                state={
                    "session_id": "ses-dash01",
                    "query": "Consulta confidencial con DNI 99988877Z",
                    "effective_query": "Consulta efectiva",
                    "intent": "HOSPITALITY_CORE",
                    "route_decision": "parallel_retrieve",
                    "territorio_normalizado": "zaragoza",
                    "oficio_coloquial_detectado": "cocinero",
                    "final_response": "Texto confidencial.",
                    "audit_report": AuditReport(passed=True, has_explicit_citations=True).model_dump(),
                    "node_timings_ms": {"analyze_and_route": 1000.0, "synthesize_response": 2000.0},
                    "models_used": {"analyze_and_route": "gemini-3.1-flash-lite", "synthesize_response": "gemini-3.5-flash-lite"},
                    "usage_events": [
                        StepUsageRecord(step="analyze_and_route", provider="gemini_llm", model="gemini-3.1-flash-lite", input_tokens=2500, output_tokens=250, api_calls=1).model_dump(),
                        StepUsageRecord(step="synthesize_response", provider="gemini_llm", model="gemini-3.5-flash-lite", input_tokens=7500, output_tokens=700, api_calls=1).model_dump(),
                    ],
                },
            )
            catalog = load_pricing_catalog(fetch_live=False, snapshot_path=tmp / "no_snap.json")
            export_cost_analytics_csvs(db_path=db_path, pricing_catalog=catalog, sessions_csv_path=sess_csv, turns_csv_path=turn_csv)
            dash = generate_cost_dashboard_html(
                sessions_csv_path=sess_csv,
                turns_csv_path=turn_csv,
                pricing_catalog=catalog,
                output_path=out_html,
            )
            self.assertTrue(out_html.exists())
            self.assertEqual(dash["sessions_in_dashboard"], 1)
            html_txt = out_html.read_text(encoding="utf-8")
            self.assertIn("ses-dash01", html_txt)
            self.assertNotIn("99988877Z", html_txt)
            self.assertNotIn("<script src=\"http", html_txt)


if __name__ == "__main__":
    unittest.main()
