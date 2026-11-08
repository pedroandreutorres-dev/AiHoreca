"""Esquemas Pydantic v2 y definición del estado del grafo agéntico (LangGraph) para AiHoreca."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

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
            "'UNSUPPORTED_PROVINCE' si es de hostelería pero pregunta por una provincia española fuera de la cobertura MVP (Madrid, Cataluña/Barcelona/Girona/Tarragona/Lleida, León); "
            "'OUT_OF_SCOPE' si es ajena al derecho laboral de hostelería (ej. accidentes de tráfico, divorcios, otros convenios como metal o construcción, medicina, programación, etc.)."
        )
    )
    motivo_clasificacion: str = Field(
        description="Justificación técnica breve de la decisión de enrutamiento, resolución de correferencia y dominio."
    )
    provincia_detectada_en_turno_actual: Optional[str] = Field(
        default=None,
        description=(
            "Provincia o comunidad autónoma española donde se ubica físicamente el centro de trabajo mencionado en el MENSAJE ACTUAL, "
            "normalizada en minúsculas y sin tildes (ej. 'madrid', 'barcelona', 'girona', 'tarragona', 'lleida', 'catalunya', 'leon', 'sevilla', 'valencia'). "
            "Si el usuario menciona un municipio o localidad (incluso con erratas tipográficas, ej. 'Ponferrada' o 'Ponferada' -> 'leon', 'Móstoles' -> 'madrid', 'Sitges' -> 'barcelona'), "
            "devuelve su provincia oficial. "
            "REGLA CRÍTICA DE DESAMBIGUACIÓN: Ignora topónimos que formen parte del nombre comercial del establecimiento o de una calle "
            "(ej. en 'trabaja en el Café Madrid de Ponferrada', 'Café Madrid' es el nombre del local y la provincia real es 'leon'). "
            "Si tienes dudas de si un nombre es el local o la ubicación geográfica, o si el mensaje actual no menciona ubicación, devuelve null."
        ),
    )
    cambio_de_puesto_en_turno_actual: bool = Field(
        default=False,
        description="True si el usuario menciona explícitamente un puesto u oficio nuevo o distinto en el mensaje actual.",
    )
    requiere_convenio_provincial: bool = Field(
        description=(
            "True si la respuesta exacta depende del Convenio Colectivo Provincial/Autonómico "
            "(ej. tablas salariales, sueldo base, pluses, nocturnidad, vacaciones concretas, jornada anual, descansos, licencias retribuidas, bajas/IT, faltas/sanciones específicas). "
            "False SOLO si la consulta se refiere exclusivamente a la estructura general estatal del ALEH VI o del Estatuto de los Trabajadores sin depender de condiciones provinciales."
        )
    )
    requiere_puesto_o_categoria: bool = Field(
        default=False,
        description=(
            "True si para responder con exactitud la pregunta del usuario es imprescindible conocer el puesto de trabajo o categoría profesional del trabajador "
            "(ej. consultar cuál es su sueldo/salario base, calcular el precio de su hora extraordinaria, conocer su nivel retributivo, funciones de su puesto o periodo de prueba por grupo). "
            "False si la consulta es general del convenio o estatuto y aplica por igual sin importar el puesto (ej. cuántos días de vacaciones anuales hay, cuántas pagas extras tiene el convenio, cuál es la jornada máxima anual o días de permiso por mudanza/matrimonio)."
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
            "con la sesión activa (territorio, puesto(s) o condiciones mencionadas en el diálogo). "
            "REGLA CRÍTICA DE PROGRESIÓN: (A) Si existía `pending_clarification_query` (consulta previa detenida esperando provincia) y el usuario aporta la provincia, "
            "fusiona aquella pregunta pendiente con la provincia. (B) Si NO había consulta bloqueada y el usuario hace una repregunta o nueva duda en la conversación "
            "(ej. '¿Y cómo sé a qué clase corresponde?', '¿Y cuántos días tengo por mudanza?'), reescribe ÚNICAMENTE la nueva duda de este turno "
            "contextualizada con los datos del caso (ej. 'Para una camarera con contrato de 30 horas en un bar de Ponferrada (León), ¿cómo puedo saber a qué categoría o clase de establecimiento corresponde el bar?'), "
            "PROHIBIDO volver a meter en la reescritura preguntas ya respondidas en turnos anteriores (como '¿cuál es su sueldo base?')."
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
    """Cálculo aritmético trazable (prorrateo de jornada parcial, cálculo de hora extra, salario anual/mensual) a partir de cifras literales."""

    concepto: str = Field(
        description=(
            "Concepto calculado (ej. 'Salario Base Mensual 30h/40h (2025 - Lujo y 1ª)', "
            "'Precio Hora Extra 2025 (Segunda clase)', 'Salario Bruto Anual Proporcional 30h')."
        )
    )
    formula_aplicada: str = Field(
        description="Expresión aritmética explícita utilizando únicamente cifras de los fragmentos o de la consulta (ej. '1.458,20 * (30 / 40)')."
    )
    operandos_base_del_corpus: list[str] = Field(
        default_factory=list,
        description="Cifras literales extraídas de los fragmentos normativos que intervienen en la fórmula (ej. ['1.458,20', '1790', '62']).",
    )
    resultado_formateado: str = Field(
        description="Resultado numérico final formateado en español (ej. '1.093,65 €', '18,32 €')."
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
            "Cálculos aritméticos aplicados al caso concreto del usuario (ej. prorrateo por contrato a tiempo parcial "
            "como 30h/40h, o cálculo del importe en €/hora de las horas extraordinarias aplicando la fórmula del convenio "
            "sobre el salario base vigente). Cada cálculo debe declarar su fórmula y sus operandos literales del corpus."
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
    requiere_convenio_provincial: bool
    requiere_puesto_o_categoria: bool
    datos_completos: bool
    route_decision: RouteDecision
    oficio_coloquial_detectado: Optional[str]
    categoria_profesional_aleh: Optional[str]
    area_funcional_aleh: Optional[str]
    query_busqueda_hibrida: str

    # Resultado de Nodo 3 (Recuperación Híbrida Paralela 3-Capas + Cohere Rerank)
    retrieved_docs: list[RetrievedDocument]

    # Resultado de Nodo 4 + 5 (Síntesis Jurídica + Guardrail de Salida)
    jerarquia_normativa_aplicada: str
    citas_normativas: list[dict[str, Any]]
    cifras_y_datos_criticos: list[str]
    audit_report: dict[str, Any]
    final_response: str
    node_timings_ms: dict[str, float]
    models_used: dict[str, str]
