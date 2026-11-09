"""Nodos de la máquina de estados agéntica (LangGraph) para AiHoreca con soporte Multi-Sesión.

Implementa:
- Nodo 0 + 1 + 2 (`analyze_and_route_node`): Guardrail de Entrada + Gatekeeper Contextual + Resolución de
  Correferencia Multi-Turno + Traductor ALEH VI en una única llamada estructurada (`Pydantic v2`) sobre
  `gemini-3.1-flash-lite` (con failover multi-modelo), aplicando precedencia temporal estricta
  (`Turno Actual > Slots de Sesión Activa`) para impedir contaminación de estado (*State Poisoning*).
- Nodo Bloqueo (`block_out_of_scope_node`): Cortocircuito de coste 0 tokens para consultas fuera de dominio
  o de provincias aún no indexadas en el MVP.
- Nodo Clarificación (`ask_clarification_node`): Cortocircuito HITL cuando la consulta requiere convenio provincial
  pero ni el turno actual ni los slots de la sesión contienen la provincia.
- Nodo 3 (`parallel_retrieval_node`): Recuperación híbrida paralela en las 3 capas normativas
  (Estatuto de los Trabajadores + VI ALEH + Convenio Colectivo Provincial estratificado) y reranking neuronal
  (`Cohere rerank-multilingual-v3.0`) con selección por cuotas jerárquicas.
- Nodo 4 + 5 (`legal_synthesis_and_audit_node`): Síntesis jurídica con jerarquía normativa (`Art. 3.3 ET`)
  y auditoría determinista de citas y fidelidad numérica (Guardrail de Salida).
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Any, Optional, Type, TypeVar

from google.genai import types
from pydantic import BaseModel

from src.config import (
    GEMINI_FALLBACK_MODELS,
    GEMINI_LITE_MODEL,
    GEMINI_REASONING_MODEL,
    PROCESSED_DATA_DIR,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_TOP_K_PER_LAYER,
    get_genai_client,
)
from src.graph.state import (
    AuditReport,
    LegalGraphState,
    LegalSynthesisSchema,
    QueryAnalysisSchema,
    RouteDecision,
    SessionContextSlots,
)
from src.retrieval.schemas import RetrievedDocument
from src.retrieval.vector_store import AiHorecaVectorStore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Cooldown en memoria por modelo ante picos 503/429 del Free Tier
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_COOLDOWN_SECONDS = 90.0

AUTONOMOUS_COMMUNITY_EXPANSION: dict[str, list[str]] = {
    "catalunya": ["barcelona", "girona", "tarragona", "lleida", "catalunya"],
}

# Normalización canónica de variantes ortográficas/cooficiales de nombres de provincia devueltos por el LLM
_CANONICAL_PROVINCE_MAP: dict[str, str] = {
    "la coruna": "a_coruna",
    "a coruna": "a_coruna",
    "coruna": "a_coruna",
    "araba": "alava",
    "alacant": "alicante",
    "illes balears": "baleares",
    "bcn": "barcelona",
    "castello": "castellon",
    "ciudad real": "ciudad_real",
    "gerona": "girona",
    "gipuzkoa": "guipuzcoa",
    "la rioja": "la_rioja",
    "las palmas": "las_palmas",
    "lerida": "lleida",
    "comunidad de madrid": "madrid",
    "orense": "ourense",
    "santa cruz de tenerife": "santa_cruz_tenerife",
    "bizkaia": "vizcaya",
    "cataluna": "catalunya",
}


def _normalize_text(text: str) -> str:
    """Elimina tildes y pasa a minúsculas para comparaciones deterministas."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _canonicalize_llm_province(raw_prov: Optional[str]) -> Optional[str]:
    """Normaliza el nombre de provincia devuelto por el LLM al identificador canónico de Qdrant."""
    if not raw_prov:
        return None
    cleaned = _normalize_text(raw_prov).strip()
    cleaned = re.sub(r"^(?:provincia\s+de\s+|comunidad\s+de\s+)", "", cleaned).strip()
    if not cleaned or cleaned in ("null", "none", "n/a", "ninguna", "estatal"):
        return None
    if cleaned in _CANONICAL_PROVINCE_MAP:
        return _CANONICAL_PROVINCE_MAP[cleaned]
    return cleaned.replace(" ", "_")


def get_indexed_territories() -> set[str]:
    """Descubre dinámicamente qué provincias/territorios están indexados en el corpus procesado."""
    indexed: set[str] = set()
    cache_path = PROCESSED_DATA_DIR / "boundaries_cache.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            for info in data.values():
                terr = _normalize_text(str(info.get("territorio", "")).strip())
                if terr and terr != "estatal":
                    indexed.add(terr)
                    if terr in AUTONOMOUS_COMMUNITY_EXPANSION:
                        indexed.update(AUTONOMOUS_COMMUNITY_EXPANSION[terr])
        except Exception as exc:
            logger.warning("No se pudo leer boundaries_cache.json: %s", exc)

    if not indexed:
        for md_file in PROCESSED_DATA_DIR.glob("convenio_*.md"):
            stem = _normalize_text(md_file.stem.replace("convenio_", "").replace("_texto", ""))
            if "catalunya" in stem or "bcn" in stem:
                indexed.update(["catalunya", "barcelona", "girona", "tarragona", "lleida"])
            else:
                indexed.add(stem)
    return indexed


def _call_structured_llm(
    prompt: str,
    schema: Type[T],
    preferred_model: str,
    system_instruction: str,
    temperature: float = 0.0,
) -> tuple[T, str]:
    """Ejecuta una llamada estructurada (JSON Schema Pydantic) con failover automático multi-modelo."""
    client = get_genai_client()
    candidate_models: list[str] = [preferred_model]
    for m in GEMINI_FALLBACK_MODELS:
        if m not in candidate_models:
            candidate_models.append(m)

    now = time.time()
    available_models = [
        m for m in candidate_models if _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now
    ]
    if not available_models:
        available_models = candidate_models

    last_error: Optional[Exception] = None
    for model_name in available_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            if response.parsed is not None and isinstance(response.parsed, schema):
                parsed_obj = response.parsed
            else:
                raw_text = (response.text or "").strip()
                parsed_obj = schema.model_validate_json(raw_text)

            # Verificación anti-truncamiento para respuestas largas de síntesis jurídica
            if hasattr(parsed_obj, "respuesta_final_markdown"):
                md_body = getattr(parsed_obj, "respuesta_final_markdown", "").strip()
                if len(md_body) < 300 or md_body.endswith("(") or md_body.endswith(":"):
                    raise ValueError(
                        f"Generación truncada prematuramente en {model_name} (len={len(md_body)})."
                    )
            return parsed_obj, model_name
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if any(code in err_str for code in ("429", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")):
                _MODEL_COOLDOWN_UNTIL[model_name] = time.time() + _COOLDOWN_SECONDS
                logger.warning(
                    "Modelo %s saturado/timeout (%s). Activando cooldown de %.0fs y rotando al siguiente modelo...",
                    model_name,
                    err_str[:120],
                    _COOLDOWN_SECONDS,
                )
                continue
            logger.warning("Error en modelo %s (%s). Probando fallback...", model_name, err_str[:120])
            continue

    raise RuntimeError(f"Todos los modelos de la cadena de failover fallaron. Último error: {last_error}")


class AiHorecaGraphNodes:
    """Contenedor de nodos del grafo agéntico con instancia compartida de Qdrant VectorStore."""

    def __init__(self, vector_store: Optional[AiHorecaVectorStore] = None) -> None:
        self.vector_store = vector_store or AiHorecaVectorStore()
        self.indexed_territories = get_indexed_territories()

    def analyze_and_route_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo 0 + 1 + 2: Guardrail + Gatekeeper + Resolución de Correferencia Multi-Turno + Traductor ALEH VI."""
        t0 = time.perf_counter()
        query = state.get("query", "").strip()
        history = state.get("conversation_history") or []
        slots = SessionContextSlots.model_validate(state.get("session_slots") or {})

        indexed_list_str = ", ".join(sorted(self.indexed_territories))

        system_instruction = (
            "Eres un Arquitecto Jurídico Laboral experto en el Sector de Hostelería y Turismo de España.\n"
            "Tu misión es ejecutar simultáneamente 4 tareas de alta precisión:\n"
            "1. RESOLUCIÓN DE CORREFERENCIA Y PROGRESIÓN CONVERSACIONAL (SIN RE-PREGUNTAR LO YA RESUELTO):\n"
            "   - CASO A (Desbloqueo de aclaración pendiente): Si en `SLOTS ACTIVOS DE LA SESIÓN` existe `pending_clarification_query` (una consulta previa que quedó "
            "detenida esperando provincia, puesto de trabajo o ambos) y el mensaje actual del usuario aporta el dato que faltaba (ej. 'Trabajo en Madrid', 'Es camarera', 'De cocinero en León'), "
            "clasifica como 'HOSPITALITY_CORE' (o 'UNSUPPORTED_PROVINCE' si la provincia no está soportada) y fusiona la pregunta pendiente "
            "con los datos conocidos de la sesión y el nuevo dato aportado en `consulta_autocontenida_resuelta`.\n"
            "   - CASO B (Repregunta o seguimiento en conversación activa): Si `pending_clarification_query` es null y el usuario plantea una repregunta o seguimiento, "
            "hereda de `SLOTS ACTIVOS DE LA SESIÓN` y de `VENTANA COMPACTA DE TURNOS RECIENTES` el territorio, puesto(s) y condiciones contractuales ya aportadas en la sesión, "
            "PERO centra `consulta_autocontenida_resuelta` y `query_busqueda_hibrida` EXCLUSIVAMENTE en responder la nueva duda planteada en el `MENSAJE ACTUAL`. "
            "NUNCA arrastres ni vuelvas a incluir preguntas que ya fueron respondidas en turnos anteriores.\n\n"
            "2. GUARDRAIL DE DOMINIO:\n"
            "   - Clasifica como 'OUT_OF_SCOPE' cualquier consulta ajena al derecho laboral de hostelería/restauración/alojamiento/turismo "
            "(ej. accidentes de tráfico, herencias, divorcios, convenios del metal/construcción/comercio general, programación, medicina, etc.).\n"
            f"   - Clasifica como 'UNSUPPORTED_PROVINCE' si la consulta es de hostelería pero el territorio aplicable es una provincia española "
            f"NO incluida en los territorios indexados actuales ({indexed_list_str}).\n"
            "   - Clasifica como 'HOSPITALITY_CORE' si es una consulta laboral aplicable al sector de hostelería/turismo/catering/reparto de comida (incluyendo respuestas breves que aportan un puesto o una provincia cuando había una consulta pendiente).\n\n"
            "3. GATEKEEPER CONTEXTUAL GENERALISTA (PROVINCIA Y PUESTO/CATEGORÍA):\n"
            "   - `provincia_detectada_en_turno_actual`: Identifica por contexto semántico la **provincia española donde se ubica físicamente el centro de trabajo** mencionado en el `MENSAJE ACTUAL`, en minúsculas y sin tildes.\n"
            "     * Si el usuario menciona un municipio, localidad o comarca (incluso con errores tipográficos), resuelve y devuelve su provincia administrativa oficial.\n"
            "     * **REGLA DE DESAMBIGUACIÓN ENTIDAD vs. JURISDICCIÓN**: Distingue siempre entre la ubicación geográfica real del centro de trabajo y los nombres comerciales de negocios, marcas, cadenas o calles que contengan topónimos. Si existe ambigüedad sobre cuál es la ubicación geográfica real o si no se menciona ubicación en este turno, devuelve null.\n"
            "   - `requiere_convenio_provincial`: Marca `True` SIEMPRE que la consulta trate sobre salarios, tablas salariales, sueldos, pluses, "
            "nocturnidad, horas extra, vacaciones, jornada anual, descansos, licencias/permisos retribuidos, bajas médicas (complemento IT), "
            "categorías/clases de establecimientos del convenio o régimen concreto de un centro de trabajo.\n"
            "   - `requiere_puesto_o_categoria`: Marca `True` cuando para responder la pregunta con exactitud sea imprescindible saber el puesto o categoría profesional del trabajador "
            "(ej. consultar cuál es su sueldo/salario base, calcular el precio de su hora extra, nivel salarial o periodo de prueba según grupo profesional). "
            "Marca `False` cuando la pregunta sea general del convenio o estatuto y aplique igual a todos los puestos (ej. cuántos días de vacaciones anuales corresponden, cuántas pagas extras hay al año, jornada máxima anual o días de permiso por mudanza/matrimonio).\n\n"
            "4. TRADUCTOR DE OFICIOS COLOQUIALES AL VI ALEH (PROHIBIDO ADIVINAR EL PUESTO POR EL TIPO DE NEGOCIO):\n"
            "   - **REGLA ESTRICTA**: Si el usuario solo menciona el tipo de establecimiento ('trabaja en un bar', 'en un restaurante', 'en un hotel', 'en una cafetería') pero **NO especifica qué puesto u oficio desempeña** (ni consta en `SLOTS ACTIVOS DE LA SESIÓN`), DEBES dejar `oficio_coloquial_detectado = null` y `categoria_profesional_aleh = null`. Jamás des por hecho ni inventes un puesto.\n"
            "   - Cuando el usuario SÍ indique un puesto u oficio (en el turno actual o heredado de los slots de la sesión), mapéalo a su denominación técnica en el VI ALEH:\n"
            "     * 'friegaplatos' / 'office' / 'pica' -> 'Auxiliar de limpieza / Marmitón / Freganchín (Área Funcional 2ª Cocina o 3ª Restauración, Grupo Profesional III / Nivel salarial inferior)'\n"
            "     * 'pinche' / 'ayudante de cocina' -> 'Ayudante/a de cocina / Auxiliar de cocina (Área Funcional 2ª Cocina y Economato, Grupo Profesional II/III)'\n"
            "     * 'cocinero' / 'chef' -> 'Jefe/a de cocina, Segundo/a Jefe/a de cocina, Jefe/a de partida o Cocinero/a (Área Funcional 2ª Cocina y Economato, Grupos Profesionales I y II)'\n"
            "     * 'camarero' / 'camarera' / 'chico de barra' / 'mesero' -> 'Camarero/a, Sumiller, Barman o Ayudante/a de camarero/a (Área Funcional 3ª Restauración, Grupo Profesional II/III)'\n"
            "     * 'rider' / 'repartidor' / 'delivery' -> 'Repartidor/a de comida a domicilio (Área Funcional 3ª Restauración / Disposición Adicional ALEH y Convenio Provincial)'\n"
            "     * 'camarera de pisos' / 'kelly' / 'limpiadora de habitaciones' -> 'Camarero/a de pisos / Gobernante/a / Auxiliar de pisos y limpieza (Área Funcional 4ª Pisos y Limpieza, Grupos Profesionales I/II/III)'\n"
            "     * 'recepcionista' / 'botones' / 'conserje' -> 'Recepcionista / Conserje / Auxiliar de recepción y conserjería (Área Funcional 1ª Recepción, Conserjería, Relaciones Públicas, Administración y Gestión)'\n"
            "     * 'mantenimiento' / 'socorrista' / 'animador' -> 'Área Funcional 5ª (Mantenimiento y Servicios Auxiliares) o Área Funcional 6ª (Servicios Complementarios)'\n"
            "   - Construye `query_busqueda_hibrida` combinando los conceptos clave de `consulta_autocontenida_resuelta`, la categoría oficial del ALEH, el área funcional "
            "y el territorio aplicable (NOTA: incluye los términos 'tablas salariales salario base gratificaciones extraordinarias pagas extras' cuando el turno actual pregunte por sueldos, salarios, nómina, retribuciones o cálculo de horas extras)."
        )

        prompt = (
            f"SLOTS ACTIVOS DE LA SESIÓN (WORKING MEMORY): {json.dumps(slots.model_dump(), ensure_ascii=False)}\n"
            f"VENTANA COMPACTA DE TURNOS RECIENTES: {json.dumps(history, ensure_ascii=False)}\n"
            f"MENSAJE ACTUAL DEL USUARIO: {query}\n"
            f"TERRITORIOS SOPORTADOS EN ÍNDICE QDRANT: {indexed_list_str}"
        )

        analysis, model_used = _call_structured_llm(
            prompt=prompt,
            schema=QueryAnalysisSchema,
            preferred_model=GEMINI_LITE_MODEL,
            system_instruction=system_instruction,
            temperature=0.0,
        )

        # 1. Resolución de Provincia con Precedencia Temporal Estricta (100% decidida por el análisis semántico del LLM):
        #    Prioridad 1: Provincia detectada semánticamente en el mensaje actual.
        #    Prioridad 2: Slot `active_territorio` heredado de la sesión (solo si no hubo mención en el turno actual).
        current_turn_prov = _canonicalize_llm_province(analysis.provincia_detectada_en_turno_actual)
        raw_prov = current_turn_prov if current_turn_prov is not None else slots.active_territorio

        intent = analysis.intent
        territorio_norm: Optional[str] = None

        if intent != "OUT_OF_SCOPE":
            if raw_prov:
                if raw_prov in self.indexed_territories:
                    intent = "HOSPITALITY_CORE"
                    territorio_norm = raw_prov
                else:
                    # Comprobar si es una consulta multi-provincia donde todas las mencionadas están indexadas
                    raw_tokens = [t.strip() for t in re.split(r"[,/|]|\s+y\s+", raw_prov.replace("_", " ")) if t.strip()]
                    canonical_tokens = [_canonicalize_llm_province(tok) for tok in raw_tokens if tok.strip()]
                    if canonical_tokens and all(c in self.indexed_territories for c in canonical_tokens):
                        intent = "HOSPITALITY_CORE"
                        territorio_norm = raw_prov
                    else:
                        intent = "UNSUPPORTED_PROVINCE"
                        territorio_norm = raw_prov

        # 2. Resolución de Categoría Profesional ALEH (Turno actual vs Herencia de Slots)
        if analysis.cambio_de_puesto_en_turno_actual or not slots.active_categoria_aleh:
            oficio_resuelto = analysis.oficio_coloquial_detectado or slots.active_oficio_coloquial
            cat_aleh_resuelta = analysis.categoria_profesional_aleh or slots.active_categoria_aleh
            area_aleh_resuelta = analysis.area_funcional_aleh or slots.active_area_funcional_aleh
        else:
            oficio_resuelto = slots.active_oficio_coloquial or analysis.oficio_coloquial_detectado
            cat_aleh_resuelta = slots.active_categoria_aleh or analysis.categoria_profesional_aleh
            area_aleh_resuelta = slots.active_area_funcional_aleh or analysis.area_funcional_aleh

        # 3. Consulta autocontenida resuelta (fusión de pending_clarification_query o correferencia)
        effective_query = (analysis.consulta_autocontenida_resuelta or query).strip()
        if slots.pending_clarification_query and len(query.split()) <= 10:
            # Garantía determinista: si había una pregunta bloqueada y el usuario respondió con un dato corto
            # (ej. "En Madrid" o "De camarera"), aseguramos que la pregunta original no se pierda
            if slots.pending_clarification_query.lower()[:20] not in effective_query.lower():
                context_parts: list[str] = []
                if territorio_norm:
                    context_parts.append(f"Provincia: {territorio_norm}")
                if oficio_resuelto:
                    context_parts.append(f"Puesto: {oficio_resuelto}")
                suffix = f" ({', '.join(context_parts)})" if context_parts else f" ({query})"
                effective_query = f"{slots.pending_clarification_query}{suffix}"

        # 4. Decisión determinista de enrutamiento (Gatekeeper de Provincia y Puesto/Categoría)
        missing_prov = bool(analysis.requiere_convenio_provincial and not territorio_norm)
        missing_role = bool(analysis.requiere_puesto_o_categoria and not cat_aleh_resuelta)

        if intent in ("OUT_OF_SCOPE", "UNSUPPORTED_PROVINCE"):
            datos_completos = False
            route_decision: RouteDecision = "block_out_of_scope"
        elif missing_prov or missing_role:
            datos_completos = False
            route_decision = "ask_clarification"
        else:
            datos_completos = True
            route_decision = "parallel_retrieve"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        timings = dict(state.get("node_timings_ms") or {})
        timings["analyze_and_route"] = elapsed_ms
        models_used = dict(state.get("models_used") or {})
        models_used["analyze_and_route"] = model_used

        return {
            "effective_query": effective_query,
            "intent": intent,
            "motivo_clasificacion": analysis.motivo_clasificacion,
            "provincia_detectada": raw_prov,
            "territorio_normalizado": territorio_norm,
            "requiere_convenio_provincial": analysis.requiere_convenio_provincial,
            "requiere_puesto_o_categoria": analysis.requiere_puesto_o_categoria,
            "datos_completos": datos_completos,
            "route_decision": route_decision,
            "oficio_coloquial_detectado": oficio_resuelto,
            "categoria_profesional_aleh": cat_aleh_resuelta,
            "area_funcional_aleh": area_aleh_resuelta,
            "query_busqueda_hibrida": analysis.query_busqueda_hibrida or effective_query,
            "node_timings_ms": timings,
            "models_used": models_used,
        }

    def block_out_of_scope_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo de Bloqueo Determinista (0 tokens de generación): rechaza consultas ajenas o provincias no indexadas."""
        t0 = time.perf_counter()
        intent = state.get("intent", "OUT_OF_SCOPE")
        prov = state.get("provincia_detectada")

        if intent == "UNSUPPORTED_PROVINCE" and prov:
            prov_pretty = prov.replace("_", " ").title()
            msg = (
                f"### Aviso de Cobertura Territorial (MVP AiHoreca)\n\n"
                f"He detectado que tu consulta pertenece al ámbito laboral de **Hostelería**, pero hace referencia a la provincia de **{prov_pretty}**.\n\n"
                f"- **Estado actual del índice normativo**: En esta versión MVP están indexados y verificados los convenios colectivos de "
                f"**Comunidad de Madrid**, **Cataluña (Barcelona, Girona, Tarragona y Lleida)** y **León**, además de las dos normas estatales "
                f"(**Estatuto de los Trabajadores** y **VI Acuerdo Laboral Estatal de Hostelería — ALEH VI**).\n"
                f"- **Por rigor jurídico (Cero Alucinación)**: No puedo aplicarte tablas salariales ni condiciones de otra provincia ni responderte solo con el Estatuto de los Trabajadores "
                f"ocultando las mejoras del Convenio Colectivo de {prov_pretty}.\n\n"
                f"Si deseas consultar únicamente la **clasificación profesional estatal (Áreas Funcionales del VI ALEH)** o los **mínimos generales del Estatuto de los Trabajadores**, "
                f"indícalo expresamente o realiza tu consulta para **Madrid, Cataluña (Barcelona/Girona/Tarragona/Lleida) o León**."
            )
        else:
            msg = (
                "### Consulta Fuera del Dominio de Especialización\n\n"
                "Soy **AiHoreca**, un asistente jurídico-laboral especializado exclusivamente en el **Sector de Hostelería y Turismo en España** "
                "(Estatuto de los Trabajadores, VI Acuerdo Laboral Estatal de Hostelería y Convenios Colectivos Provinciales/Autonómicos del sector).\n\n"
                f"- **Motivo de bloqueo**: {state.get('motivo_clasificacion', 'La consulta no pertenece al ámbito laboral de hostelería.')}\n"
                "- No puedo asesorarte sobre otras ramas del Derecho (civil, penal, tráfico, mercantil) ni sobre convenios colectivos de otros sectores."
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        timings = dict(state.get("node_timings_ms") or {})
        timings["block_out_of_scope"] = elapsed_ms

        return {
            "final_response": msg,
            "retrieved_docs": [],
            "audit_report": AuditReport(
                passed=True,
                has_explicit_citations=False,
                verified_numbers=[],
                unverified_numbers=[],
                warnings=["Bloqueo determinista activado correctamente en Guardrail de Entrada."],
            ).model_dump(),
            "node_timings_ms": timings,
        }

    def ask_clarification_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo de Clarificación HITL (0 tokens extra): solicita la provincia, el puesto de trabajo o ambos cuando son indispensables."""
        t0 = time.perf_counter()
        terr = state.get("territorio_normalizado")
        oficio = state.get("oficio_coloquial_detectado")
        cat_aleh = state.get("categoria_profesional_aleh")
        area_aleh = state.get("area_funcional_aleh")
        req_prov = bool(state.get("requiere_convenio_provincial", True))
        req_role = bool(state.get("requiere_puesto_o_categoria", False))

        missing_prov = req_prov and not terr
        missing_role = req_role and not cat_aleh

        aleh_note = ""
        if cat_aleh or area_aleh:
            aleh_note = (
                f"\n\n> **Adelanto de encuadramiento estatal (VI ALEH)**: "
                f"El puesto mencionado (*{oficio or 'indicado'}*) se encuadra a nivel estatal como **{cat_aleh or 'personal operativo'}** "
                f"dentro del **{area_aleh or 'Área Funcional correspondiente del VI ALEH'}**."
            )

        if missing_prov and missing_role:
            msg = (
                "### Información Requerida para el Cálculo (Provincia y Puesto de Trabajo)\n\n"
                "Para darte una respuesta jurídicamente exacta y sin asumir datos que alterarían las cifras, **necesito conocer dos datos básicos**:\n\n"
                "1. **¿En qué provincia o localidad está ubicado el centro de trabajo?** (Cada provincia tiene su propio Convenio Colectivo de Hostelería con tablas salariales, número de pagas extras y jornada anual propios).\n"
                "2. **¿Qué puesto u oficio desempeña dentro del establecimiento?** (Por ejemplo: *camarero/a, ayudante de camarero/a, cocinero/a, ayudante de cocina/pinche, friegaplatos/limpieza, recepcionista...*, ya que el salario base varía según el Grupo Profesional y Nivel salarial).\n\n"
                "**Territorios disponibles actualmente en el índice:**\n"
                "- **Madrid** (Comunidad de Madrid)\n"
                "- **Barcelona / Cataluña** (Barcelona, Girona, Tarragona o Lleida)\n"
                "- **León** (Provincia de León)"
            )
            warn_msg = "Gatekeeper contextual detuvo la consulta por falta de provincia y puesto de trabajo."
        elif missing_role and not missing_prov:
            terr_pretty = (terr or "").replace("_", " ").title()
            msg = (
                f"### Puesto de Trabajo Requerido (Convenio de Hostelería — {terr_pretty})\n\n"
                f"He registrado que el centro de trabajo se encuentra en la provincia/territorio de **{terr_pretty}**.\n\n"
                "Sin embargo, para calcular el **sueldo exacto o las condiciones retributivas** sin adivinar datos, "
                "**necesito saber qué puesto u oficio desempeña** dentro del establecimiento.\n\n"
                "En un establecimiento de hostelería coexisten distintos Grupos Profesionales y niveles salariales según la función:\n"
                "- **Sala / Barra:** Camarero/a (Grupo II), Ayudante de camarero/a (Grupo III), Barman/Sumiller.\n"
                "- **Cocina:** Jefe/a de cocina (Grupo I), Cocinero/a (Grupo II), Ayudante de cocina / Pinche (Grupo III).\n"
                "- **Limpieza / Servicios auxiliares:** Auxiliar de limpieza / Friegaplatos (Grupo III), Camarero/a de pisos.\n\n"
                "**¿Cuál es su puesto o categoría profesional?** *(Respóndeme directamente con el puesto y calcularé al instante su tabla salarial aplicable en "
                f"{terr_pretty}).*"
            )
            warn_msg = f"Gatekeeper contextual detuvo la consulta en {terr_pretty} por falta de puesto/categoría profesional."
        else:
            msg = (
                "### Información Territorial Requerida (Gatekeeper Contextual)\n\n"
                "Para darte una respuesta jurídicamente exacta y sin inventar cifras, **necesito saber en qué provincia trabajas**.\n\n"
                "En el sector de hostelería en España:\n"
                "1. El **Estatuto de los Trabajadores** fija únicamente los mínimos legales estatales.\n"
                "2. El **VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)** regula las Áreas Funcionales, Grupos Profesionales y el régimen disciplinario.\n"
                "3. Sin embargo, **las tablas salariales, pluses (nocturnidad, transporte, manutención), jornada anual, descansos y días de vacaciones/licencias** "
                "dependen directamente de tu **Convenio Colectivo Provincial o Autonómico**."
                f"{aleh_note}\n\n"
                "**¿En cuál de las provincias o territorios soportados prestas servicios?**\n"
                "- **Madrid** (Comunidad de Madrid)\n"
                "- **Barcelona / Cataluña** (Barcelona, Girona, Tarragona o Lleida)\n"
                "- **León** (Provincia de León)"
            )
            warn_msg = "Gatekeeper contextual detuvo la consulta por falta de provincia obligatoria."

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        timings = dict(state.get("node_timings_ms") or {})
        timings["ask_clarification"] = elapsed_ms

        return {
            "final_response": msg,
            "retrieved_docs": [],
            "audit_report": AuditReport(
                passed=True,
                has_explicit_citations=True,
                verified_numbers=[],
                unverified_numbers=[],
                warnings=[warn_msg],
            ).model_dump(),
            "node_timings_ms": timings,
        }

    def parallel_retrieval_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo 3: Recuperación Híbrida Paralela en 3 Capas + Estratificación Provincial + Cohere Rerank."""
        t0 = time.perf_counter()
        effective_query = state.get("effective_query") or state.get("query", "")
        enriched_query = state.get("query_busqueda_hibrida") or effective_query
        provincia = state.get("territorio_normalizado")
        oficio = state.get("oficio_coloquial_detectado")
        cat_aleh = state.get("categoria_profesional_aleh")
        area_aleh = state.get("area_funcional_aleh")

        role_parts = [p for p in (oficio, cat_aleh, area_aleh) if p]
        role_context = " | ".join(role_parts) if role_parts else None

        rerank_query = (
            f"{effective_query} ({role_context})" if role_context else effective_query
        )

        # Recuperación jerárquica paralela estratificada (1 sola llamada API de embedding + BM25 CPU + Cohere Rerank)
        docs = self.vector_store.parallel_hierarchical_retrieve(
            query=enriched_query,
            provincia=provincia,
            top_k_per_layer=RETRIEVAL_TOP_K_PER_LAYER,
            final_top_k=RETRIEVAL_FINAL_TOP_K,
            use_cohere_rerank=True,
            rerank_query=rerank_query,
            role_context=role_context,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        timings = dict(state.get("node_timings_ms") or {})
        timings["parallel_retrieval"] = elapsed_ms
        models_used = dict(state.get("models_used") or {})
        models_used["parallel_retrieval"] = "gemini-embedding-001 + Qdrant/bm25 + rerank-multilingual-v3.0"

        return {
            "retrieved_docs": docs,
            "node_timings_ms": timings,
            "models_used": models_used,
        }

    def legal_synthesis_and_audit_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo 4 + 5: Síntesis Jurídica con Jerarquía Normativa + Guardrail Determinista de Salida."""
        t0 = time.perf_counter()
        raw_query = state.get("query", "")
        effective_query = state.get("effective_query") or raw_query
        provincia = state.get("territorio_normalizado")
        cat_aleh = state.get("categoria_profesional_aleh")
        area_aleh = state.get("area_funcional_aleh")
        history = state.get("conversation_history") or []
        docs: list[RetrievedDocument] = state.get("retrieved_docs") or []

        context_blocks: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            score_val = doc.rerank_score if doc.rerank_score is not None else doc.score
            block = (
                f"--- FRAGMENTO #{idx} ---\n"
                f"NORMA: {doc.norma_titulo}\n"
                f"CAPA_NORMATIVA: {doc.capa_normativa} | TERRITORIO: {doc.territorio}\n"
                f"SECCIÓN: {doc.titulo_o_capitulo} | ARTÍCULO/TABLA: {doc.articulo} | TIPO: {doc.tipo_contenido}\n"
                f"RELEVANCIA_RERANK: {score_val:.4f}\n"
                f"CONTENIDO:\n{doc.text}\n"
            )
            context_blocks.append(block)

        full_context_str = "\n".join(context_blocks)

        system_instruction = (
            "Eres un Asesor Jurídico-Laboral Senior especializado en el Sector de Hostelería en España.\n"
            "Tu comunicación debe ser **natural, clara, directa y práctica**, manteniendo una precisión técnica y jurídica absoluta.\n"
            "Debes responder la consulta del usuario basándote en los FRAGMENTOS NORMATIVOS proporcionados, citando siempre de qué artículo, disposición o anexo/tabla sale cada dato.\n\n"
            "REGLAS INQUEBRANTABLES DE INGENIERÍA JURÍDICA Y CONVERSACIONAL:\n"
            "1. CONTINUIDAD CONVERSACIONAL, COHERENCIA TERRITORIAL Y PROHIBICIÓN DE REPETICIÓN ENTRE TURNOS:\n"
            "   - **PROHIBICIÓN ABSOLUTA DE SUSTITUCIÓN JURISDICCIONAL**: Está terminantemente prohibido aplicar tablas salariales, número de pagas o artículos de un convenio provincial/autonómico distinto al `TERRITORIO APLICABLE` del trabajador. Jamás utilices el convenio de otra provincia 'por analogía' ni ocultes a qué provincia pertenece el convenio citado.\n"
            "   - Revisa `HISTORIAL RECIENTE DE LA CONVERSACIÓN` para mantener todos los parámetros del caso ya aportados por el usuario en la sesión (jornada parcial o completa, puesto, modalidad, etc.) sin olvidarlos.\n"
            "   - **PROHIBIDO REPETIR EXPLICACIONES O TABLAS DE TURNOS ANTERIORES**: Si en un turno previo ya desglosaste una tabla o cálculo y en el `MENSAJE ACTUAL` el usuario plantea una repregunta o nueva duda, **responde directa y exclusivamente a la nueva pregunta del turno actual**, sin volver a volcar las tablas ni repetir el discurso del turno anterior salvo que el usuario pida recalcularlo.\n"
            "2. ADECUACIÓN DE LA ESTRUCTURA A LA INTENCIÓN DEL TURNO:\n"
            "   - Si la pregunta del turno actual es conceptual, clasificatoria, de permisos, vacaciones, jornada o aclaratoria sobre un requisito del convenio, responde de forma directa, natural y fundamentada en los fragmentos recuperados (integrando cuando proceda Convenio Provincial/Autonómico, VI ALEH y Estatuto de los Trabajadores), **sin forzar tablas salariales que no vienen al caso**.\n"
            "   - Si el convenio hace depender una condición de un parámetro del centro de trabajo o del contrato que el usuario no ha precisado (como la categoría o clasificación turística del establecimiento, su nivel o modalidad), expón las opciones previstas en los fragmentos recuperados e indícale dónde puede verificar ese dato en la práctica (contrato de trabajo, encabezado de la nómina o clasificación oficial del establecimiento).\n"
            "3. REGLAS PARA CONSULTAS SALARIALES / RETRIBUTIVAS (CUANDO EL TURNO ACTUAL PIDA SUELDOS O CÁLCULOS):\n"
            "   - **Año más reciente**: Si el usuario NO pide un año histórico concreto, utiliza siempre las tablas salariales del **año más reciente/vigente** presente en los fragmentos.\n"
            "   - **Sueldo mensual Sin Prorratear vs. Con Pagas Extras Prorrateadas**: Identifica cuántas **Gratificaciones Extraordinarias (pagas extras)** fija el convenio aplicable (citando su artículo) y presenta una **Tabla Markdown comparativa clara** en **`€ / mes`** mostrando:\n"
            "     * **Sueldo Base Mensual a Jornada Completa (40h)**.\n"
            "     * **Sueldo Base Mensual para su Jornada Contratada (SIN prorratear pagas extras)**: `Salario_40h * (Horas_Contrato / 40)` (cobrando las pagas extras aparte en sus meses de devengo).\n"
            "     * **Sueldo Base Mensual para su Jornada Contratada (CON pagas extras prorrateadas en 12 meses)**: `Sueldo_Mensual_Sin_Prorratear * (Total_Pagas_Anuales / 12)` (calcula y escribe siempre el importe numérico final en `€/mes`).\n"
            "     * Si la tabla del convenio distingue varios niveles o categorías de establecimiento para ese puesto y el usuario no ha especificado cuál es el suyo, desglosa una fila por cada opción de la tabla (e incluye el Salario Mínimo Anual Garantizado del convenio si existe).\n"
            "4. CÁLCULO COMPLETO EN `€/HORA` DE HORAS EXTRAORDINARIAS O PLUSES (CUANDO SE SOLICITE):\n"
            "   - **PROHIBIDO DEJAR FÓRMULAS SIN RESOLVER NUMÉRICAMENTE**: Jamás te limites a mostrar la fórmula teórica de la hora extraordinaria o plus sin dar el resultado numérico en **`€ / hora`**. Siempre debes sustituir los valores numéricos de las tablas salariales vigentes y calcular el importe final en `€ / hora` (bruto).\n"
            "   - Si en el convenio aplicable el Salario Base del puesto varía según la categoría, clase o nivel del establecimiento y el usuario no lo ha especificado, **presenta una Tabla Markdown comparativa con el importe exacto en `€ / hora` (bruto) para cada una de las opciones de la tabla**.\n"
            "   - Si la fórmula del convenio incluye complementos o pluses (*antigüedad/ad personam*, *asistencia*, *manutención*, etc.), calcula en la tabla el **importe mínimo garantizado en `€ / hora` sobre el Salario Base** (y opcionalmente otra columna incluyendo los pluses fijos del convenio cuyos importes consten en los fragmentos), aclarando cómo influyen.\n"
            "   - Registra todas las operaciones aritméticas realizadas en `calculos_derivados` con su fórmula y operandos base para su verificación por el Nodo 5.\n"
            "5. CRITERIO ESTRICTO SOBRE IMPORTES NETOS, IRPF Y COTIZACIONES (ALCANCE EXCLUSIVAMENTE BRUTO Y CARÁCTER ORIENTATIVO):\n"
            "   - Los convenios colectivos y las tablas salariales regulan **exclusivamente importes BRUTOS** (conforme al Art. 26.4 del Estatuto de los Trabajadores, las cargas fiscales y de Seguridad Social corren a cargo de la persona trabajadora).\n"
            "   - Siempre que el usuario pregunte por un **importe neto (sueldo neto, hora extra neta, nómina neta) o descuento de impuestos (IRPF / Seguridad Social)**, **es obligatorio advertir de forma destacada que cualquier cálculo neto es ESTRICTAMENTE ORIENTATIVO y aproximado**, ya que el sistema no realiza liquidaciones fiscales de IRPF (el porcentaje de retención depende de las circunstancias personales y familiares del Modelo 145, modalidad contractual y rendimientos anuales totales). Si muestras una simulación o ejemplo en neto, etiquétalo siempre expresamente como **«Neto Orientativo (aprox.)»**."
        )

        prompt = (
            f"HISTORIAL RECIENTE DE LA CONVERSACIÓN (TURNOS PREVIOS DE ESTA SESIÓN): {json.dumps(history, ensure_ascii=False)}\n"
            f"MENSAJE ACTUAL DEL USUARIO: {raw_query}\n"
            f"CONSULTA AUTOCONTENIDA RESUELTA (OBJETIVO DE ESTE TURNO): {effective_query}\n"
            f"TERRITORIO APLICABLE: {provincia or 'Estatal'}\n"
            f"TRADUCCIÓN TÉCNICA VI ALEH: Categoría={cat_aleh or 'N/A'} | Área Funcional={area_aleh or 'N/A'}\n\n"
            f"FRAGMENTOS NORMATIVOS RECUPERADOS DE QDRANT + COHERE RERANK:\n"
            f"{full_context_str}"
        )

        synthesis, model_used = _call_structured_llm(
            prompt=prompt,
            schema=LegalSynthesisSchema,
            preferred_model=GEMINI_REASONING_MODEL,
            system_instruction=system_instruction,
            temperature=0.0,
        )

        audit = self._audit_numerical_and_citation_fidelity(
            synthesis=synthesis,
            docs=docs,
            user_query=f"{raw_query} {effective_query}",
            expected_territorio=provincia,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        timings = dict(state.get("node_timings_ms") or {})
        timings["legal_synthesis_and_audit"] = elapsed_ms
        models_used = dict(state.get("models_used") or {})
        models_used["legal_synthesis_and_audit"] = model_used

        return {
            "jerarquia_normativa_aplicada": synthesis.jerarquia_normativa_aplicada,
            "citas_normativas": [c.model_dump() for c in synthesis.citas_normativas],
            "cifras_y_datos_criticos": synthesis.cifras_y_datos_criticos,
            "audit_report": audit.model_dump(),
            "final_response": synthesis.respuesta_final_markdown,
            "node_timings_ms": timings,
            "models_used": models_used,
        }

    @staticmethod
    def _normalize_number_token(num_str: str) -> str:
        """Normaliza representaciones numéricas españolas ('1.267,69' -> '1267.69') para verificación literal."""
        s = num_str.strip()
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        return s

    @classmethod
    def _audit_numerical_and_citation_fidelity(
        cls,
        synthesis: LegalSynthesisSchema,
        docs: list[RetrievedDocument],
        user_query: str = "",
        expected_territorio: Optional[str] = None,
    ) -> AuditReport:
        """Nodo 5 (Guardrail Determinista de Salida): verifica citas explícitas, coherencia territorial, cifras literales y cálculos derivados."""
        combined_corpus = "\n".join(d.text for d in docs)
        norm_corpus_numbers = {
            cls._normalize_number_token(m.group(0))
            for m in re.finditer(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:\.\d+)?\b", combined_corpus)
        }
        norm_query_numbers = {
            cls._normalize_number_token(m.group(0))
            for m in re.finditer(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:\.\d+)?\b", user_query)
        }
        # Constantes universales de calendario/jornada (12 meses, 11 meses laborables, 14/15 pagas, 30 días, 40h semanales, base 1 y 100%)
        standard_constants = {"1", "11", "12", "14", "15", "30", "40", "100"}

        corpus_floats: list[float] = []
        for n_str in norm_corpus_numbers:
            try:
                val = float(n_str)
                if val > 0:
                    corpus_floats.append(val)
            except ValueError:
                pass

        # Cualquier porcentaje P presente en los fragmentos recuperados habilita su factor decimal (P/100) y su multiplicador de recargo (1 + P/100)
        surcharge_multipliers: set[float] = {1.0}
        for c_val in corpus_floats:
            if 1.0 <= c_val <= 100.0:
                dec_factor = round(c_val / 100.0, 4)
                mult_factor = round(1.0 + (c_val / 100.0), 4)
                standard_constants.add(f"{dec_factor:g}")
                standard_constants.add(f"{mult_factor:g}")
                surcharge_multipliers.add(mult_factor)
                surcharge_multipliers.add(dec_factor)

        query_floats: list[float] = []
        for q_str in norm_query_numbers:
            try:
                q_val = float(q_str)
                if 1.0 <= q_val <= 40.0:
                    query_floats.append(q_val)
                    standard_constants.add(f"{round(q_val / 40.0, 4):g}")
            except ValueError:
                pass

        auto_derived_floats: set[float] = set()
        ratios_jornada = [1.0] + [h_val / 40.0 for h_val in query_floats]
        annual_hours_in_corpus = [v for v in corpus_floats if 1500.0 <= v <= 2000.0]
        for c_val in corpus_floats:
            if c_val >= 500.0:
                for r_jor in ratios_jornada:
                    scaled_val = c_val * r_jor
                    auto_derived_floats.add(round(scaled_val, 2))
                    for pagas in (12.0, 14.0, 15.0):
                        auto_derived_floats.add(round(scaled_val / pagas, 2))
                        auto_derived_floats.add(round(scaled_val * pagas, 2))
                        auto_derived_floats.add(round(scaled_val * (pagas / 12.0), 2))
                for div_val in annual_hours_in_corpus:
                    for pagas in (12.0, 14.0, 15.0):
                        base_hour = (c_val * pagas) / div_val
                        auto_derived_floats.add(round(base_hour, 2))
                        for rec in surcharge_multipliers:
                            auto_derived_floats.add(round(base_hour * rec, 2))

        # Validar cálculos derivados declarados cuyos operandos base existan en el corpus o en la consulta del usuario
        verified_derived_norms: set[str] = {f"{v:.2f}" for v in auto_derived_floats}
        derived_display_list: list[str] = []
        for calc in getattr(synthesis, "calculos_derivados", []) or []:
            ops_valid = True
            for op in calc.operandos_base_del_corpus:
                for op_m in re.finditer(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:[.,]\d+)?\b", op):
                    norm_op = cls._normalize_number_token(op_m.group(0))
                    if (
                        norm_op not in norm_corpus_numbers
                        and norm_op not in norm_query_numbers
                        and norm_op not in standard_constants
                    ):
                        ops_valid = False
                        break
            if ops_valid:
                for src_text in (calc.resultado_formateado, calc.formula_aplicada):
                    for res_m in re.finditer(
                        r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:[.,]\d+)?\b",
                        src_text,
                    ):
                        raw_res = res_m.group(0)
                        verified_derived_norms.add(cls._normalize_number_token(raw_res))
                for res_m in re.finditer(
                    r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:[.,]\d+)?\b",
                    calc.resultado_formateado,
                ):
                    raw_res = res_m.group(0)
                    if raw_res not in derived_display_list:
                        derived_display_list.append(raw_res)

        has_citations = len(synthesis.citas_normativas) > 0 and bool(
            re.search(r"(?i)\b(?:art[íi]culo|art\.|anexo|tabla|disposici[óo]n)\b", synthesis.respuesta_final_markdown)
        )

        extracted_tokens = re.findall(
            r"\b(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)\s*(?:€|euros?|%)",
            synthesis.respuesta_final_markdown,
            flags=re.IGNORECASE,
        )

        verified: list[str] = []
        derived: list[str] = list(derived_display_list)
        unverified: list[str] = []
        for raw_num in extracted_tokens:
            norm_num = cls._normalize_number_token(raw_num)
            is_derived_match = norm_num in verified_derived_norms
            if not is_derived_match:
                try:
                    f_val = float(norm_num)
                    is_derived_match = f"{f_val:.2f}" in verified_derived_norms or any(
                        abs(f_val - d_val) <= 0.02 for d_val in auto_derived_floats
                    )
                except ValueError:
                    is_derived_match = False

            if norm_num in norm_corpus_numbers or raw_num in combined_corpus:
                if raw_num not in verified:
                    verified.append(raw_num)
            elif is_derived_match:
                if raw_num not in derived:
                    derived.append(raw_num)
            else:
                if raw_num not in unverified:
                    unverified.append(raw_num)

        warnings: list[str] = []
        territory_consistent = True
        if expected_territorio:
            valid_terrs = {expected_territorio}
            for comm, provs in AUTONOMOUS_COMMUNITY_EXPANSION.items():
                if expected_territorio in provs:
                    valid_terrs.add(comm)
                    valid_terrs.update(provs)
            mismatched_terrs = {
                d.territorio
                for d in docs
                if d.capa_normativa == "convenio_provincial"
                and d.territorio
                and _normalize_text(d.territorio) not in valid_terrs
            }
            if mismatched_terrs:
                territory_consistent = False
                warnings.append(
                    f"Incoherencia territorial detectada: se recuperaron fragmentos de {mismatched_terrs} para una consulta de '{expected_territorio}'."
                )

        if not has_citations:
            warnings.append("La respuesta no contiene citas explícitas a artículos o anexos normativos.")
        if unverified:
            warnings.append(
                f"Se detectaron cifras monetarias/porcentuales no localizadas en los chunks ni en cálculos trazables: {unverified}"
            )

        passed = has_citations and len(unverified) == 0 and territory_consistent
        return AuditReport(
            passed=passed,
            has_explicit_citations=has_citations,
            verified_numbers=verified,
            derived_numbers=derived,
            unverified_numbers=unverified,
            warnings=warnings,
        )
