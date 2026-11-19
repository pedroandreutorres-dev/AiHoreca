"""Esquemas Pydantic v2 y definición del estado del grafo agéntico (LangGraph) para AiHoreca."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from src.retrieval.schemas import RetrievedDocument

IntentCategory = Literal[
    "HOSPITALITY_CORE",
    "UNSUPPORTED_PROVINCE",
    "OUT_OF_SCOPE",
]

RouteDecision = Literal[
    "block_out_of_scope",
    "ask_clarification",
    "parallel_retrieve",
]


class SessionContextSlots(BaseModel):
    """Memoria de trabajo estructurada (Working Memory Slots) aislada por session_id."""

    active_territorio: Optional[str] = Field(
        default=None,
        description="Provincia o territorio válido actualmente activo en la sesión (ej. 'madrid', 'barcelona', 'leon').",
    )
    active_oficio_coloquial: Optional[str] = Field(
        default=None,
        description="Último oficio coloquial declarado por el usuario en la sesión.",
    )
    active_categoria_aleh: Optional[str] = Field(
        default=None,
        description="Categoría profesional y grupo según VI ALEH vigentes en la sesión.",
    )
    active_area_funcional_aleh: Optional[str] = Field(
        default=None,
        description="Área Funcional del VI ALEH vigente en la sesión.",
    )
    pending_clarification_query: Optional[str] = Field(
        default=None,
        description=(
            "Consulta original del usuario que quedó detenida en ask_clarification_node "
            "a la espera de que el usuario indique la provincia."
        ),
    )


class SessionTurnRecord(BaseModel):
    """Registro persistido de un turno conversacional dentro de una sesión."""

    turn_index: int
    timestamp: str
    user_query: str
    effective_query: str
    intent: str
    route_decision: str
    territorio: Optional[str] = None
    oficio_detectado: Optional[str] = None
    categoria_aleh: Optional[str] = None
    area_funcional_aleh: Optional[str] = None
    assistant_summary: str
    final_response: str
    audit_passed: bool = True
    user_feedback: str = "SIN_VOTO"
    feedback_at: Optional[str] = None


class QueryAnalysisSchema(BaseModel):
    """Contrato estructurado (Nodo 0 + Nodo 1 + Nodo 2) para clasificación, gatekeeper territorial, correferencia y traducción ALEH VI."""

    intent: IntentCategory = Field(
        description=(
            "Clasificación estricta del dominio: "
            "'HOSPITALITY_CORE' si es una consulta laboral aplicable al sector de hostelería/turismo/restauración/alojamiento "
            "(incluyendo respuestas cortas como 'En Madrid' o 'En un hotel de Barcelona' cuando hay una consulta pendiente en la sesión); "
            "'UNSUPPORTED_PROVINCE' si es de hostelería pero pregunta por una provincia o territorio español fuera de los territorios indexados informados en el prompt; "
            "'OUT_OF_SCOPE' si es ajena al derecho laboral de hostelería (ej. accidentes de tráfico, divorcios, otros convenios como metal o construcción, medicina, programación, etc.)."
        )
    )
    motivo_clasificacion: str = Field(
        description="Justificación técnica breve de la decisión de enrutamiento, resolución de correferencia y dominio."
    )
    provincias_detectadas_en_turno_actual: list[str] = Field(
        description=(
            "Lista de provincias o comunidades autónomas españolas donde se ubican físicamente el/los centro(s) de trabajo mencionados en el MENSAJE ACTUAL, "
            "normalizadas en minúsculas y sin tildes (ej. ['madrid'], ['madrid', 'leon'] en consultas comparativas entre dos lugares, o [] si el mensaje actual no menciona ubicación). "
            "Si el usuario menciona uno o varios municipios, localidades o comarcas (incluso con errores tipográficos), resuelve y devuelve TODAS sus provincias administrativas oficiales en esta lista. "
            "REGLA DE DESAMBIGUACIÓN ENTIDAD vs. JURISDICCIÓN: Distingue siempre el lugar geográfico real donde radica cada centro de trabajo frente a topónimos "
            "que formen parte de la denominación comercial del negocio, marca o dirección postal. "
            "Si existe ambigüedad sobre la ubicación geográfica real o si el mensaje actual no menciona ubicación, devuelve []."
        ),
    )
    cambio_de_puesto_en_turno_actual: bool = Field(
        description="True si el usuario menciona explícitamente un puesto u oficio nuevo o distinto en el mensaje actual.",
    )
    requiere_consulta_normativa_bbdd: bool = Field(
        default=True,
        description=(
            "True ÚNICAMENTE cuando exista una consulta o duda jurídico-laboral concreta sobre el sector de hostelería/turismo "
            "que requiera buscar artículos o tablas en la base de datos normativa (ya sea formulada en el mensaje actual "
            "o resultante de aportar un dato que desbloquea `pending_clarification_query`). "
            "False cuando NO proceda consultar la base de datos normativa: "
            "(1) consultas no legales o ajenas al ámbito laboral ('OUT_OF_SCOPE'), "
            "(2) consultas legales de otras ramas/sectores o de provincias no indexadas ('OUT_OF_SCOPE' / 'UNSUPPORTED_PROVINCE'), o "
            "(3) interacciones conversacionales sin pregunta jurídica formulada todavía (saludos aislados como 'Hola', presentaciones sin consulta como 'Me llamo Pedro', agradecimientos, despedidas o mensajes introductorios sin duda concreta cuando `pending_clarification_query` es null)."
        ),
    )
    requiere_convenio_provincial: bool = Field(
        description=(
            "True si la respuesta exacta depende del Convenio Colectivo Provincial/Autonómico "
            "(ej. tablas salariales, sueldo base, pluses, nocturnidad, vacaciones concretas, jornada anual, descansos, licencias retribuidas, bajas/IT). "
            "False cuando la consulta se refiera exclusivamente a normativa estatal general (VI ALEH o Estatuto de los Trabajadores) sin depender de condiciones provinciales (ej. áreas funcionales, clasificación de grupos, régimen disciplinario general o prescripción de faltas)."
        )
    )
    requiere_puesto_o_categoria: bool = Field(
        description=(
            "True si para responder con exactitud la pregunta del usuario es imprescindible conocer el puesto de trabajo o categoría profesional del trabajador "
            "(ej. consultar cuál es su sueldo/salario base, calcular el precio de su hora extraordinaria, conocer su nivel retributivo, funciones de su puesto o periodo de prueba por grupo). "
            "Si el usuario formula varias sub-preguntas y al menos una depende del puesto (ej. pregunta por el sueldo Y cuántas pagas extras tiene), DEBE ser True. "
            "False únicamente si TODA la consulta es general del convenio o estatuto y aplica por igual sin importar el puesto (ej. cuántos días de vacaciones anuales hay, cuántas pagas extras tiene el convenio, cuál es la jornada máxima anual o días de permiso por mudanza/matrimonio)."
        ),
    )
    oficio_coloquial_detectado: Optional[str] = Field(
        default=None,
        description=(
            "Puesto u oficio concreto mencionado por el usuario en el mensaje actual o heredado de los slots de la sesión (ej. 'camarera', 'cocinero', 'pinche', 'friegaplatos', 'recepcionista', 'rider'), o null. "
            "PROHIBIDO adivinar o asumir un puesto solo porque el usuario mencione el tipo de negocio ('trabaja en un bar', 'en un restaurante', 'en un hotel'); si no especifica qué puesto ocupa, devuelve null."
        ),
    )
    categoria_profesional_aleh: Optional[str] = Field(
        default=None,
        description=(
            "Denominación oficial de la categoría y Grupo Profesional según el VI Acuerdo Laboral Estatal de Hostelería (ALEH VI) "
            "y los convenios de hostelería (ej. 'Auxiliar de limpieza / Friegaplatos - Grupo Profesional III', "
            "'Ayudante de cocina - Grupo Profesional III', 'Repartidor/a de comida a domicilio', 'Camarero/a - Grupo Profesional II'), "
            "o null si el usuario aún no ha indicado el puesto de trabajo."
        ),
    )
    area_funcional_aleh: Optional[str] = Field(
        default=None,
        description=(
            "Área Funcional del VI ALEH (1ª Recepción/Conserjería/Administración, 2ª Cocina y Economato, "
            "3ª Restauración/Sala/Bar/Similares, 4ª Pisos y Limpieza, 5ª Mantenimiento y Servicios Auxiliares, 6ª Servicios Complementarios)."
        ),
    )
    consulta_autocontenida_resuelta: str = Field(
        description=(
            "Reescritura autocontenida del MENSAJE ACTUAL en lenguaje natural resolviendo pronombres, elipsis o referencias implícitas "
            "con la sesión activa (territorio, puesto(s) o condiciones contractuales mencionadas en el diálogo). "
            "REGLA CRÍTICA DE PROGRESIÓN: (A) Si existía `pending_clarification_query` (consulta previa detenida esperando provincia o puesto) y el usuario aporta el dato faltante, "
            "fusiona aquella pregunta pendiente con el nuevo dato. (B) Si NO había consulta bloqueada y el usuario formula una repregunta o nueva duda en la conversación, "
            "reescribe ÚNICAMENTE la nueva duda de este turno contextualizada con el perfil activo de la sesión, "
            "PROHIBIDO volver a incluir en la reescritura preguntas que ya fueron respondidas en turnos anteriores. "
            "(C) CIERRE CUANTITATIVO: Cuando el usuario aporte en seguimiento cualquier parámetro cuantitativo, temporal o contractual (fecha de inicio/fin, duración, horas de jornada, días trabajados, antigüedad, número de festivos, etc.) vinculado a un derecho o magnitud cuantificable tratado en la sesión, formula la consulta exigiendo explícitamente el cálculo numérico exacto resultante para ese parámetro, no solo cómo afecta en abstracto."
        )
    )
    query_busqueda_hibrida: str = Field(
        description=(
            "Consulta técnica enriquecida para el motor híbrido Qdrant (Dense + BM25) y Cohere Rerank. "
            "Debe incluir los conceptos jurídicos clave de la consulta autocontenida resuelta, la categoría oficial ALEH, el área funcional, "
            "sinónimos en el convenio y el territorio aplicable."
        )
    )


class LegalCitation(BaseModel):
    """Cita normativa verificable empleada en la síntesis jurídica."""

    norma: str = Field(
        description="Nombre oficial de la norma (ej. 'Convenio Colectivo de Hostelería de León', 'VI ALEH', 'Estatuto de los Trabajadores')."
    )
    capa_normativa: str = Field(
        description="Capa jerárquica ('convenio_provincial', 'aleh_estatal', 'estatuto_trabajadores')."
    )
    articulo_o_tabla: str = Field(
        description="Artículo, disposición o anexo/tabla exacta citada (ej. 'Artículo 6', 'Anexo I - Tablas Salariales')."
    )
    extracto_clave: str = Field(
        description="Dato o regla exacta extraída de ese artículo o tabla."
    )


class DerivedCalculation(BaseModel):
    """Cálculo aritmético trazable (prorrateo de jornada parcial, prorrateo de pagas extras, cálculo de hora extra) a partir de cifras literales."""

    concepto: str = Field(
        description="Nombre descriptivo del concepto retributivo o laboral calculado."
    )
    formula_aplicada: str = Field(
        description="Expresión aritmética explícita utilizando únicamente cifras literales de los fragmentos recuperados o de la consulta."
    )
    operandos_base_del_corpus: list[str] = Field(
        default_factory=list,
        description="Cifras literales extraídas de los fragmentos normativos que intervienen como operandos en la fórmula.",
    )
    resultado_formateado: str = Field(
        description="Resultado numérico final formateado en español con su unidad correspondiente."
    )


class LegalSynthesisSchema(BaseModel):
    """Contrato estructurado (Nodo 4 + Nodo 5) para la síntesis jurídica con jerarquía normativa y guardrail de salida."""

    jerarquia_normativa_aplicada: str = Field(
        description=(
            "Explicación concisa de cómo se articulan las 3 capas normativas para esta consulta: "
            "1) Estatuto de los Trabajadores (mínimo de derecho necesario), "
            "2) VI ALEH (clasificación profesional, áreas funcionales y régimen disciplinario estatal), y "
            "3) Convenio Colectivo Provincial/Autonómico (mejora o regulación específica aplicable por principio de norma más favorable)."
        )
    )
    citas_normativas: list[LegalCitation] = Field(
        default_factory=list,
        description="Lista de citas normativas explícitas que respaldan la respuesta.",
    )
    cifras_y_datos_criticos: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de cifras literales (importes en €, porcentajes %, número de días, horas anuales) "
            "extraídas directamente de los fragmentos recuperados. No inventar ni extrapolar ninguna cifra."
        ),
    )
    calculos_derivados: list[DerivedCalculation] = Field(
        default_factory=list,
        description=(
            "Cálculos aritméticos aplicados al caso del usuario (prorrateo por jornada parcial, "
            "prorrateo de pagas extraordinarias o cálculo del valor en €/hora de horas extraordinarias/pluses aplicando la fórmula del convenio "
            "sobre los importes vigentes de las tablas). Cada cálculo debe declarar su fórmula y sus operandos literales del corpus."
        ),
    )
    respuesta_final_markdown: str = Field(
        description=(
            "Respuesta jurídica completa, rigurosa y estructurada en formato Markdown para el usuario final. "
            "Debe citar explícitamente los artículos y tablas de cada norma (Convenio Provincial, VI ALEH y Estatuto de los Trabajadores), "
            "priorizar siempre las tablas salariales del año más reciente/vigente disponible en los fragmentos (salvo que se pida un año histórico), "
            "mostrar tanto el salario base mensual/anual a jornada completa como el cálculo exacto adaptado al contrato del usuario (si indicó horas parciales o pidió calcular horas extras), "
            "y advertir con transparencia si algún dato específico no figura en los fragmentos."
        )
    )


class AuditReport(BaseModel):
    """Resultado determinista del Guardrail de Salida (Nodo 5)."""

    passed: bool = Field(description="True si supera todas las validaciones de citas y fidelidad numérica.")
    has_explicit_citations: bool = Field(description="True si incluye referencias normativas explícitas.")
    verified_numbers: list[str] = Field(default_factory=list, description="Cifras literales verificadas en los chunks recuperados.")
    derived_numbers: list[str] = Field(default_factory=list, description="Cifras calculadas mediante fórmulas trazables cuyos operandos base existen en los chunks.")
    unverified_numbers: list[str] = Field(default_factory=list, description="Cifras presentes en la síntesis que no aparecen en los chunks ni provienen de un cálculo verificado.")
    warnings: list[str] = Field(default_factory=list, description="Advertencias de auditoría.")


class StepUsageRecord(BaseModel):
    """Telemetría exacta de consumo de tokens y llamadas API para un paso del turno (Bloque D)."""

    step: str = Field(description="Identificador del paso o nodo (ej. analyze_and_route, synthesize_response, rerank_cohere).")
    provider: str = Field(default="gemini_llm", description="Proveedor o subsistema (gemini_llm, gemini_embedding, cohere_rerank).")
    model: str = Field(default="", description="Nombre exacto del modelo utilizado.")
    status: str = Field(default="ok", description="Estado del paso: ok, cache_hit, failed_429, failed_503_504, failed_truncated, failed_other.")
    input_tokens: int = Field(default=0, description="Tokens de entrada reportados por el servidor (prompt_token_count).")
    output_tokens: int = Field(default=0, description="Tokens de salida reportados por el servidor (candidates_token_count).")
    thoughts_tokens: int = Field(default=0, description="Tokens de razonamiento interno si el modelo los reporta (thoughts_token_count).")
    cached_tokens: int = Field(default=0, description="Tokens servidos desde caché de contexto (cached_content_token_count).")
    total_tokens: int = Field(default=0, description="Total de tokens reportado por el servidor (total_token_count).")
    api_calls: int = Field(default=0, description="Número real de llamadas HTTP a la API (0 si es cache_hit).")
    cache_hit: bool = Field(default=False, description="True si el vector/resultado provino de caché single-flight en memoria.")
    chars: int = Field(default=0, description="Caracteres de entrada enviados (útil para embeddings sin token count).")
    search_units: int = Field(default=0, description="Unidades de búsqueda reportadas por Cohere Rerank (billed_units.search_units).")
    docs_reranked: int = Field(default=0, description="Cantidad de documentos reordenados en Cohere.")
    latency_ms: float = Field(default=0.0, description="Latencia medida de la llamada en milisegundos.")
    attempt: int = Field(default=1, description="Número de intento dentro del paso.")
    error_detail: str = Field(default="", description="Detalle conciso en caso de failover o reintento.")


class TurnUsageSummary(BaseModel):
    """Resumen agregado de tokens y llamadas API de un turno completo."""

    input_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    embedding_chars: int = 0
    cohere_calls: int = 0
    cohere_search_units: int = 0
    cohere_docs_reranked: int = 0
    llm_calls: int = 0
    llm_failovers: int = 0
    rate_limit_429_count: int = 0
    server_503_504_count: int = 0
    truncated_retries: int = 0
    steps: list[StepUsageRecord] = Field(default_factory=list)


def build_turn_usage_summary(state: dict[str, Any] | None) -> TurnUsageSummary:
    """Agrega la lista `usage_events` de un estado de grafo en un `TurnUsageSummary` determinista."""
    raw_events = (state or {}).get("usage_events") or []
    steps: list[StepUsageRecord] = []
    summary = TurnUsageSummary()

    for raw in raw_events:
        if isinstance(raw, StepUsageRecord):
            rec = raw
        elif isinstance(raw, dict):
            try:
                rec = StepUsageRecord.model_validate(raw)
            except Exception:
                continue
        else:
            continue

        steps.append(rec)
        summary.input_tokens += max(0, int(rec.input_tokens or 0))
        summary.output_tokens += max(0, int(rec.output_tokens or 0))
        summary.thoughts_tokens += max(0, int(rec.thoughts_tokens or 0))
        summary.cached_tokens += max(0, int(rec.cached_tokens or 0))
        ev_total = int(rec.total_tokens or 0)
        if ev_total <= 0:
            ev_total = (
                max(0, int(rec.input_tokens or 0))
                + max(0, int(rec.output_tokens or 0))
                + max(0, int(rec.thoughts_tokens or 0))
            )
        summary.total_tokens += max(0, ev_total)

        if rec.provider == "gemini_llm":
            summary.llm_calls += max(0, int(rec.api_calls or 0))
            if rec.status != "ok":
                summary.llm_failovers += 1
            if rec.status == "failed_429":
                summary.rate_limit_429_count += 1
            elif rec.status == "failed_503_504":
                summary.server_503_504_count += 1
            elif rec.status == "failed_truncated":
                summary.truncated_retries += 1
        elif rec.provider == "gemini_embedding":
            summary.embedding_calls += max(0, int(rec.api_calls or 0))
            if rec.cache_hit or rec.status == "cache_hit":
                summary.embedding_cache_hits += 1
            summary.embedding_chars += max(0, int(rec.chars or 0))
        elif rec.provider == "cohere_rerank":
            summary.cohere_calls += max(0, int(rec.api_calls or 0))
            summary.cohere_search_units += max(0, int(rec.search_units or 0))
            summary.cohere_docs_reranked += max(0, int(rec.docs_reranked or 0))

    summary.steps = steps
    return summary


class LegalGraphState(TypedDict, total=False):
    """Estado tipado de la máquina de estados en LangGraph para AiHoreca (con soporte Multi-Sesión)."""

    # Entrada y Contexto de Sesión
    session_id: str
    query: str
    effective_query: str
    session_slots: dict[str, Any]
    conversation_history: list[dict[str, str]]

    # Resultado de Nodo 0 + 1 + 2 (Guardrail + Gatekeeper + Correferencia + Traductor ALEH)
    intent: IntentCategory
    motivo_clasificacion: str
    provincia_detectada: Optional[str]
    territorio_normalizado: Optional[str]
    territorios_normalizados: list[str]
    requiere_consulta_normativa_bbdd: bool
    requiere_convenio_provincial: bool
    requiere_puesto_o_categoria: bool
    datos_completos: bool
    route_decision: RouteDecision
    oficio_coloquial_detectado: Optional[str]
    categoria_profesional_aleh: Optional[str]
    area_funcional_aleh: Optional[str]
    query_busqueda_hibrida: str

    # Fan-Out / Fan-In con Send() + Nodo de Verificación / Expansión (Bloque B)
    branch_docs: Annotated[list[RetrievedDocument], operator.add]
    branch_timings: Annotated[list[tuple[str, float]], operator.add]
    retrieval_hop: int
    missing_cross_refs: list[dict[str, str]]
    needs_retrieval_expansion: bool

    # Resultado de Nodo 3 + Verificación (Recuperación Híbrida Paralela 3-Capas + Cohere Rerank)
    retrieved_docs: list[RetrievedDocument]

    # Resultado de Nodo 4 + 5 (Síntesis Jurídica + Guardrail de Salida con autocorrección)
    jerarquia_normativa_aplicada: str
    citas_normativas: list[dict[str, Any]]
    cifras_y_datos_criticos: list[str]
    calculos_derivados: list[dict[str, Any]]
    audit_report: dict[str, Any]
    audit_retry_count: int
    audit_needs_retry: bool
    final_response: str
    node_timings_ms: dict[str, float]
    models_used: dict[str, str]
    # Telemetría acumulada de consumo de tokens y llamadas API (Bloque D, reducer operator.add)
    usage_events: Annotated[list[dict[str, Any]], operator.add]
