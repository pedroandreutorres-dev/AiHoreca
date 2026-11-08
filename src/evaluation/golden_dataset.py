"""Golden Dataset jurídico-laboral verificado para la evaluación empírica de AiHoreca (Fase 4).

Cubre 12 escenarios representativos divididos en 3 bloques:
1. Guardarraíles de Entrada y Enrutamiento Territorial (`guardrail_routing`).
2. Recuperación Híbrida 3-Capas, Síntesis Jurídica y Tablas Salariales (`provincial_rag`).
3. Memoria Conversacional Multi-Turno, Resolución HITL y Blindaje Anti-State-Poisoning (`multi_turn_memory`).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.graph.state import IntentCategory, RouteDecision

EvalCategory = Literal[
    "guardrail_routing",
    "provincial_rag",
    "multi_turn_memory",
]


class GoldenTestCase(BaseModel):
    """Caso de prueba auditado del Golden Dataset de AiHoreca."""

    case_id: str = Field(description="Identificador único del caso de prueba.")
    category: EvalCategory = Field(description="Categoría técnica evaluada.")
    session_id: str = Field(description="ID de sesión en SQLite para probar aislamiento o continuidad multi-turno.")
    reset_session_before: bool = Field(
        default=True,
        description="True si se debe limpiar la sesión antes de ejecutar el caso; False si continúa el turno previo.",
    )
    query: str = Field(description="Mensaje de entrada del usuario.")
    expected_intent: IntentCategory = Field(description="Clasificación de dominio esperada.")
    expected_route: RouteDecision = Field(description="Decisión de enrutamiento esperada en LangGraph.")
    expected_territorio: Optional[str] = Field(
        default=None,
        description="Territorio normalizado esperado ('madrid', 'barcelona', 'girona', 'leon', 'valencia', 'sevilla' o None).",
    )
    expected_layers: list[str] = Field(
        default_factory=list,
        description="Capas normativas que deben estar representadas en los chunks recuperados (vacío si es bloqueo/HITL).",
    )
    expected_citation_keywords: list[str] = Field(
        default_factory=list,
        description="Subcadenas de artículos, anexos o disposiciones que deben aparecer en los chunks recuperados.",
    )
    expected_numbers_any: list[str] = Field(
        default_factory=list,
        description="Cifras del Ground Truth que deben aparecer verificadas en el contexto/respuesta.",
    )
    ground_truth_reference: str = Field(
        description="Respuesta experta de referencia (Ground Truth) para cálculo de métricas RAGAS."
    )


def get_golden_dataset() -> list[GoldenTestCase]:
    """Devuelve los 12 casos verificados del Golden Dataset de AiHoreca."""
    return [
        # =========================================================================
        # BLOQUE 1: GUARDARRAÍLES DE ENTRADA Y ENRUTAMIENTO DETERMINISTA (4 CASOS)
        # =========================================================================
        GoldenTestCase(
            case_id="GD_01_OOS_TRAFICO",
            category="guardrail_routing",
            session_id="eval-oos-01",
            reset_session_before=True,
            query="¿Qué indemnización diaria me corresponde en 2025 por un latigazo cervical en un accidente de tráfico?",
            expected_intent="OUT_OF_SCOPE",
            expected_route="block_out_of_scope",
            expected_territorio=None,
            expected_layers=[],
            expected_citation_keywords=[],
            expected_numbers_any=[],
            ground_truth_reference=(
                "Bloqueo determinista por consulta ajena al derecho laboral de hostelería (derecho civil / baremo de tráfico)."
            ),
        ),
        GoldenTestCase(
            case_id="GD_02_OOS_OTRO_SECTOR",
            category="guardrail_routing",
            session_id="eval-oos-02",
            reset_session_before=True,
            query="Soy programador informático en una consultora tecnológica en Madrid, ¿qué trienios de antigüedad fija el convenio de consultoría?",
            expected_intent="OUT_OF_SCOPE",
            expected_route="block_out_of_scope",
            expected_territorio=None,
            expected_layers=[],
            expected_citation_keywords=[],
            expected_numbers_any=[],
            ground_truth_reference=(
                "Bloqueo determinista porque el sector de consultoría informática es ajeno al sector de hostelería y turismo."
            ),
        ),
        GoldenTestCase(
            case_id="GD_03_UNSUPPORTED_SEVILLA",
            category="guardrail_routing",
            session_id="eval-unsupported-03",
            reset_session_before=True,
            query="Trabajo de camarero de sala en un hotel de Sevilla, ¿cuál es mi salario base mensual según convenio?",
            expected_intent="UNSUPPORTED_PROVINCE",
            expected_route="block_out_of_scope",
            expected_territorio="sevilla",
            expected_layers=[],
            expected_citation_keywords=[],
            expected_numbers_any=[],
            ground_truth_reference=(
                "Aviso de cobertura territorial MVP: la consulta es de hostelería en Sevilla, pero el MVP solo indexa "
                "Madrid, Cataluña (Barcelona, Girona, Tarragona, Lleida) y León, negándose a inventar tablas salariales."
            ),
        ),
        GoldenTestCase(
            case_id="GD_04_HITL_FRIEGAPLATOS_SIN_PROVINCIA",
            category="multi_turn_memory",
            session_id="eval-ses-madrid",
            reset_session_before=True,
            query="Soy friegaplatos a jornada completa, ¿cuánto tengo que cobrar de sueldo base y cuántos días de vacaciones me tocan?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="ask_clarification",
            expected_territorio=None,
            expected_layers=[],
            expected_citation_keywords=[],
            expected_numbers_any=[],
            ground_truth_reference=(
                "El Gatekeeper Contextual detiene el flujo porque el salario base y las vacaciones dependen del Convenio Provincial, "
                "adelanta el encuadramiento estatal en el VI ALEH (Auxiliar de limpieza / Marmitón / Freganchín - Grupo Profesional III, "
                "Área Funcional 2ª o 3ª) y solicita indicar la provincia (Madrid, Barcelona/Cataluña o León)."
            ),
        ),
        # =========================================================================
        # BLOQUE 2: CONTINUIDAD MULTI-TURNO, HITL Y ANTI-STATE-POISONING (5 CASOS)
        # =========================================================================
        GoldenTestCase(
            case_id="GD_05_MADRID_HITL_RESOLUCION",
            category="multi_turn_memory",
            session_id="eval-ses-madrid",
            reset_session_before=False,
            query="Trabajo en Madrid.",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio="madrid",
            expected_layers=["convenio_provincial", "aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["Anexo", "18"],
            expected_numbers_any=["1.101,69", "1.032,48", "183,87", "30"],
            ground_truth_reference=(
                "Resuelve la consulta pendiente de friegaplatos (Fregador/a) en el Convenio de Hostelería de la Comunidad de Madrid: "
                "según el Anexo III (Tablas Salariales 2024), el salario base mensual para Fregador/a es de 1.101,69 € en establecimientos "
                "de Lujo, 1ª y 2ª categoría, y 1.032,48 € en 3ª y 4ª categoría (o niveles equivalentes en tabla salarial vigente), "
                "más Plus Convenio de 183,87 € mensuales (11 meses). En vacaciones (Artículo 18 del Convenio de Madrid y Art. 38 ET), "
                "corresponden 30 días naturales ininterrumpidos al año."
            ),
        ),
        GoldenTestCase(
            case_id="GD_06_MADRID_SEGUIMIENTO_NOCTURNIDAD",
            category="multi_turn_memory",
            session_id="eval-ses-madrid",
            reset_session_before=False,
            query="¿Y cómo se regula o cuánto se paga por nocturnidad en mi caso?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio="madrid",
            expected_layers=["convenio_provincial", "aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["27", "Anexo"],
            expected_numbers_any=["1", "25"],
            ground_truth_reference=(
                "Hereda de la sesión que trabaja como Fregador/a en Madrid. Según el Artículo 27 del Convenio Colectivo de Hostelería de Madrid, "
                "la nocturnidad se retribuye en dos tramos sobre el salario base: de 22:00h a 0:00h con un incremento del 1%, "
                "y de 0:00h a 8:00h con un incremento del 25% (fórmula: Salario Base x Porcentaje / (4 x 40)). "
                "Si trabaja 5 o más horas entre las 22:00h y las 8:00h (Art. 27.b), toda la jornada se considera nocturna."
            ),
        ),
        GoldenTestCase(
            case_id="GD_07_LEON_PINCHE_SALARIO",
            category="provincial_rag",
            session_id="eval-ses-leon",
            reset_session_before=True,
            query="Soy ayudante de cocina (pinche) en un restaurante en León. ¿En qué área funcional me encuadro y cuál es mi salario base según las tablas salariales?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio="leon",
            expected_layers=["convenio_provincial", "aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["ANEXO I", "Cocina"],
            expected_numbers_any=["1.223,60", "1.185,31", "1.147,07", "1.266,42"],
            ground_truth_reference=(
                "Según el VI ALEH y el Convenio Colectivo Provincial de Hostelería y Turismo de León, el puesto de Pinche / Ayudante de cocina "
                "se encuadra en el Área Funcional 2ª (Cocina y Economato), Grupo Profesional III. Según el Anexo I (Tablas Salariales) del "
                "Convenio de León, el salario base mensual para Pinche es: en Lujo y 1.ª clase 1.223,60 € (2023/2024) / 1.266,42 € (2025); "
                "en 2.ª clase 1.185,31 € / 1.226,80 €; y en 3.ª clase y sucesivas 1.147,07 € / 1.187,22 €, además del complemento en especie "
                "de manutención de 24,65 € (Art. 12.º)."
            ),
        ),
        GoldenTestCase(
            case_id="GD_08_LEON_SWITCH_VALENCIA_ANTI_POISONING",
            category="multi_turn_memory",
            session_id="eval-ses-leon",
            reset_session_before=False,
            query="¿Y si me ofrecen irme de cocinero a un hotel en Valencia, cuánto cobraría allí?",
            expected_intent="UNSUPPORTED_PROVINCE",
            expected_route="block_out_of_scope",
            expected_territorio="valencia",
            expected_layers=[],
            expected_citation_keywords=[],
            expected_numbers_any=[],
            ground_truth_reference=(
                "Detecta por precedencia temporal que el turno actual pregunta por Valencia (provincia no soportada en el MVP), "
                "bloquea determinísticamente la consulta sin inventar salarios, y preserva intacto el slot 'leon' de la sesión."
            ),
        ),
        GoldenTestCase(
            case_id="GD_09_LEON_RETORNO_VACACIONES_Y_PERMISOS",
            category="multi_turn_memory",
            session_id="eval-ses-leon",
            reset_session_before=False,
            query="Vale, quedándome aquí en mi restaurante actual, ¿cuántos días de vacaciones anuales y permisos retribuidos por matrimonio o fallecimiento tengo?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio="leon",
            expected_layers=["convenio_provincial", "aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["24", "23"],
            expected_numbers_any=["31", "15", "3"],
            ground_truth_reference=(
                "Recupera el territorio 'leon' de la sesión (no corrompido por la pregunta previa sobre Valencia). "
                "Según el Artículo 24.º del Convenio de Hostelería de León, corresponden 31 días naturales de vacaciones retribuidas al año "
                "(mejorando el mínimo de 30 días del Art. 38 del Estatuto de los Trabajadores). Según el Artículo 23.º del Convenio de León, "
                "los permisos retribuidos incluyen 15 días naturales por matrimonio o pareja de hecho, 5 días por hospitalización/enfermedad grave "
                "de parientes hasta 2.º grado, 3 días por fallecimiento (ampliables por desplazamiento), 2 días por traslado de domicilio y 1 día de asuntos propios."
            ),
        ),
        # =========================================================================
        # BLOQUE 3: CASOS MULTIPROVINCIALES CATALUÑA Y MARCO ESTATAL ALEH (3 CASOS)
        # =========================================================================
        GoldenTestCase(
            case_id="GD_10_BARCELONA_RIDER_Y_FALTA_MUY_GRAVE",
            category="provincial_rag",
            session_id="eval-ses-bcn",
            reset_session_before=True,
            query="Trabajo de rider repartiendo comida a domicilio para un restaurante en Barcelona. ¿Cómo se regula mi puesto y retribución en el convenio y qué consecuencias tiene cometer una falta muy grave?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio="barcelona",
            expected_layers=["convenio_provincial", "aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["Artículo 2", "Barcelona", "40"],
            expected_numbers_any=["1.516,39", "1.350,86", "4,00", "56,93"],
            ground_truth_reference=(
                "En el Convenio Colectivo Interprovincial de Hostelería y Turismo de Cataluña (Art. 2), la actividad de reparto de comidas a domicilio "
                "(riders) está expresamente incluida y encuadrada en el Grupo/Nivel V (o Nivel V bis para repartidor de restauración moderna sin experiencia). "
                "Según el Anexo II (A. Barcelona), para 2025 el salario en Nivel V es de 1.516,39 € (columnas A/B/E) o 1.481,86 € (C/D), y en Nivel V bis "
                "de 1.350,86 € / 1.333,53 €, con incremento pactado del 4,00% (Art. 35) y pluses de manutención (56,93 €), transporte (13,86 €) y ropa (14,80 €). "
                "Las faltas muy graves se regulan en el Art. 40 del VI ALEH / Anexo I del Convenio y el Art. 58 del Estatuto de los Trabajadores, "
                "pudiendo conllevar suspensión de empleo y sueldo (de 16 a 60 días) o despido disciplinario (con trámite de audiencia previa de 2 días, Art. 41 ALEH)."
            ),
        ),
        GoldenTestCase(
            case_id="GD_11_GIRONA_CAMARERA_PISOS",
            category="provincial_rag",
            session_id="eval-ses-girona",
            reset_session_before=True,
            query="Soy camarera de pisos en un hotel en Girona. ¿En qué área funcional del ALEH estoy encuadrada y qué tablas salariales del convenio me aplican?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio="girona",
            expected_layers=["convenio_provincial", "aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["Girona", "Anexo"],
            expected_numbers_any=["1.313,09", "1.365,62", "1.420,24", "1.462,85", "7"],
            ground_truth_reference=(
                "Verifica el mapeo multiprovincial de Girona al Convenio Colectivo Interprovincial de Hostelería y Turismo de Cataluña. "
                "Según el VI ALEH, el puesto de Camarera de pisos se encuadra en el Área Funcional 4ª (Pisos y Limpieza), Grupo Profesional II. "
                "En el Convenio de Cataluña (Anexo II C) Girona) se encuadra en el Nivel 7 (VII), con salario base mensual en Girona de "
                "1.313,09 € (2025), 1.365,62 € (2026), 1.420,24 € (2027) y 1.462,85 € (2028), más pluses de manutención, alojamiento, transporte y ropa."
            ),
        ),
        GoldenTestCase(
            case_id="GD_12_ESTATAL_ALEH_AREAS_FUNCIONALES",
            category="provincial_rag",
            session_id="eval-ses-estatal",
            reset_session_before=True,
            query="¿Cuáles son las 6 áreas funcionales en las que se estructura la clasificación profesional a nivel estatal según el VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)?",
            expected_intent="HOSPITALITY_CORE",
            expected_route="parallel_retrieve",
            expected_territorio=None,
            expected_layers=["aleh_estatal", "estatuto_trabajadores"],
            expected_citation_keywords=["Artículo 15", "Artículo 16", "Artículo 22"],
            expected_numbers_any=["6", "seis"],
            ground_truth_reference=(
                "Es una consulta estrictamente estatal sobre el VI ALEH que no requiere provincia (requiere_convenio_provincial=False). "
                "Según los Artículos 13, 14 y 15 del VI Acuerdo Laboral Estatal de Hostelería (ALEH VI), la clasificación profesional se organiza en "
                "6 Áreas Funcionales: Área Funcional Primera (Recepción-Conserjería, Relaciones Públicas, Administración y Gestión), "
                "Área Funcional Segunda (Cocina y Economato), Área Funcional Tercera (Restauración), Área Funcional Cuarta (Pisos y Limpieza), "
                "Área Funcional Quinta (Servicios de Mantenimiento y Servicios Auxiliares) y Área Funcional Sexta (Servicios Complementarios)."
            ),
        ),
    ]
