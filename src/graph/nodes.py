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
- Nodo 4 (`synthesize_response_node`): Síntesis jurídica con jerarquía normativa (`Art. 3.3 ET`).
- Nodo 5 (`audit_output_node`): Auditoría determinista en CPU de citas, territorio y fidelidad numérica con autocorrección (hasta 1 reintento).
"""

from __future__ import annotations

from datetime import date
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
    DerivedCalculation,
    LegalCitation,
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

# Cooldown en memoria por modelo ante cuotas (429) vs saturación transitoria (503/504) vs modelo inexistente (404)
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_MODEL_QUOTA_COOLDOWN_UNTIL: dict[str, float] = {}
_COOLDOWN_SECONDS = 45.0
_QUOTA_COOLDOWN_SECONDS = 120.0
_NOT_FOUND_COOLDOWN_SECONDS = 86400.0  # 24h para modelos no existentes o retirados de la API
_TRANSIENT_RETRY_BACKOFFS: tuple[float, ...] = (8.0, 15.0, 25.0)

# Atajo determinista de CPU para saludos y despedidas conversacionales aislados (0 tokens, <1ms)
_GREETING_OR_FAREWELL_RE = re.compile(
    r"^[¡¿]*(?:hola(?:\s+qu[eé]\s+tal|\s+buenas(?:\s+tardes|\s+noches)?)?|buenos\s+d[ií]as|buenas\s+(?:tardes|noches)|buenas|qu[eé]\s+tal|saludos|adi[oó]s|hasta\s+luego|hasta\s+pronto|muchas\s+gracias|gracias)\b[!.? ]*$",
    re.IGNORECASE,
)

# Detección determinista de solicitudes de comparativa abierta/global o indiferencia provincial
_GLOBAL_COMPARISON_RE = re.compile(
    r"\b(?:cada|todas?\s+las)\s+provincias?|da\s+igual\s+(?:en\s+qu[eé]\s+)?provincia|en\s+ninguna\s+provincia|de\s+toda\s+espa[nñ]a|comparativa\s+general|ranking\s+de\s+provincias\b",
    re.IGNORECASE,
)

AUTONOMOUS_COMMUNITY_EXPANSION: dict[str, list[str]] = {
    "catalunya": ["barcelona", "girona", "tarragona", "lleida", "catalunya"],
}

# Normalización canónica de variantes ortográficas/cooficiales de nombres de provincia devueltos por el LLM
_CANONICAL_PROVINCE_MAP: dict[str, str] = {
    "la coruna": "a_coruna",
    "a coruna": "a_coruna",
    "coruna": "a_coruna",
    "santiago": "a_coruna",
    "santiago de compostela": "a_coruna",
    "compostela": "a_coruna",
    "ferrol": "a_coruna",
    "naron": "a_coruna",
    "carballo": "a_coruna",
    "oleiros": "a_coruna",
    "arteixo": "a_coruna",
    "culleredo": "a_coruna",
    "ames": "a_coruna",
    "ribeira": "a_coruna",
    "betanzos": "a_coruna",
    "cambre": "a_coruna",
    "boiro": "a_coruna",
    "noia": "a_coruna",
    "araba": "alava",
    "araba/alava": "alava",
    "araba / alava": "alava",
    "alava/araba": "alava",
    "vitoria": "alava",
    "vitoria-gasteiz": "alava",
    "vitoria gasteiz": "alava",
    "gasteiz": "alava",
    "llodio": "alava",
    "laudio": "alava",
    "amurrio": "alava",
    "alacant": "alicante",
    "illes balears": "baleares",
    "islas baleares": "baleares",
    "balears": "baleares",
    "mallorca": "baleares",
    "menorca": "baleares",
    "ibiza": "baleares",
    "eivissa": "baleares",
    "formentera": "baleares",
    "bcn": "barcelona",
    "comunidad autonoma de cantabria": "cantabria",
    "santander": "cantabria",
    "torrelavega": "cantabria",
    "castro urdiales": "cantabria",
    "castro-urdiales": "cantabria",
    "camargo": "cantabria",
    "maliano": "cantabria",
    "pielagos": "cantabria",
    "el astillero": "cantabria",
    "astillero": "cantabria",
    "laredo": "cantabria",
    "santona": "cantabria",
    "los corrales de buelna": "cantabria",
    "corrales de buelna": "cantabria",
    "santa cruz de bezana": "cantabria",
    "bezana": "cantabria",
    "reinosa": "cantabria",
    "suances": "cantabria",
    "colindres": "cantabria",
    "cabezon de la sal": "cantabria",
    "santillana del mar": "cantabria",
    "san vicente de la barquera": "cantabria",
    "comillas": "cantabria",
    "potes": "cantabria",
    "noja": "cantabria",
    "isla": "cantabria",
    "arnuero": "cantabria",
    "medio cudeyo": "cantabria",
    "solares": "cantabria",
    "marina de cudeyo": "cantabria",
    "pedrena": "cantabria",
    "somo": "cantabria",
    "ribamontan al mar": "cantabria",
    "ampuero": "cantabria",
    "ramales de la victoria": "cantabria",
    "castello": "castellon",
    "ciudad real": "ciudad_real",
    "gerona": "girona",
    "gipuzkoa": "guipuzcoa",
    "donostia": "guipuzcoa",
    "san sebastian": "guipuzcoa",
    "donostia-san sebastian": "guipuzcoa",
    "donostia san sebastian": "guipuzcoa",
    "la rioja": "la_rioja",
    "las palmas": "las_palmas",
    "las palmas de gran canaria": "las_palmas",
    "gran canaria": "las_palmas",
    "lanzarote": "las_palmas",
    "fuerteventura": "las_palmas",
    "lerida": "lleida",
    "comunidad de madrid": "madrid",
    "principado de asturias": "asturias",
    "oviedo": "asturias",
    "gijon": "asturias",
    "xixon": "asturias",
    "aviles": "asturias",
    "siero": "asturias",
    "pola de siero": "asturias",
    "langreo": "asturias",
    "la felguera": "asturias",
    "sama": "asturias",
    "mieres": "asturias",
    "castrillon": "asturias",
    "piedras blancas": "asturias",
    "san martin del rey aurelio": "asturias",
    "corvera": "asturias",
    "villaviciosa": "asturias",
    "llanes": "asturias",
    "cangas del narcea": "asturias",
    "cangas de onis": "asturias",
    "ribadesella": "asturias",
    "navia": "asturias",
    "luarca": "asturias",
    "valdes": "asturias",
    "grado": "asturias",
    "tineo": "asturias",
    "pravia": "asturias",
    "carreno": "asturias",
    "candas": "asturias",
    "gozon": "asturias",
    "luanco": "asturias",
    "laviana": "asturias",
    "lena": "asturias",
    "aller": "asturias",
    "cudillero": "asturias",
    "region de murcia": "murcia",
    "nafarroa": "navarra",
    "comunidad foral de navarra": "navarra",
    "orense": "ourense",
    "vigo": "pontevedra",
    "vilagarcia": "pontevedra",
    "vilagarcia de arousa": "pontevedra",
    "villagarcia": "pontevedra",
    "redondela": "pontevedra",
    "cangas": "pontevedra",
    "marin": "pontevedra",
    "ponteareas": "pontevedra",
    "a estrada": "pontevedra",
    "lalin": "pontevedra",
    "o porrino": "pontevedra",
    "porrino": "pontevedra",
    "sanxenxo": "pontevedra",
    "o grove": "pontevedra",
    "baiona": "pontevedra",
    "nigran": "pontevedra",
    "tui": "pontevedra",
    "cambados": "pontevedra",
    "poio": "pontevedra",
    "moana": "pontevedra",
    "bueu": "pontevedra",
    "caldas de reis": "pontevedra",
    "mos": "pontevedra",
    "gondomar": "pontevedra",
    "vilanova de arousa": "pontevedra",
    "a guarda": "pontevedra",
    "tomino": "pontevedra",
    "santa cruz de tenerife": "santa_cruz_tenerife",
    "tenerife": "santa_cruz_tenerife",
    "la palma": "santa_cruz_tenerife",
    "la gomera": "santa_cruz_tenerife",
    "el hierro": "santa_cruz_tenerife",
    "bizkaia": "vizcaya",
    "calatayud": "zaragoza",
    "utebo": "zaragoza",
    "ejea de los caballeros": "zaragoza",
    "ejea": "zaragoza",
    "tarazona": "zaragoza",
    "caspe": "zaragoza",
    "la almunia de dona godina": "zaragoza",
    "la almunia": "zaragoza",
    "tauste": "zaragoza",
    "cuarte de huerva": "zaragoza",
    "zuera": "zaragoza",
    "alagon": "zaragoza",
    "la puebla de alfinden": "zaragoza",
    "borja": "zaragoza",
    "epila": "zaragoza",
    "fuentes de ebro": "zaragoza",
    "maria de huerva": "zaragoza",
    "villanueva de gallego": "zaragoza",
    "mallen": "zaragoza",
    "carinena": "zaragoza",
    "daroca": "zaragoza",
    "pedrola": "zaragoza",
    "pinseque": "zaragoza",
    "cadrete": "zaragoza",
    "figueruelas": "zaragoza",
    "pina de ebro": "zaragoza",
    "belchite": "zaragoza",
    "sos del rey catolico": "zaragoza",
    "alhama de aragon": "zaragoza",
    "jaraba": "zaragoza",
    "nuevalos": "zaragoza",
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
                terr = _canonicalize_llm_province(str(info.get("territorio", "")).strip())
                if terr:
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


_PRETTY_TERRITORY_LABELS: dict[str, str] = {
    "a_coruna": "A Coruña",
    "alava": "Álava",
    "baleares": "Islas Baleares",
    "cadiz": "Cádiz",
    "catalunya": "Cataluña",
    "castellon": "Castellón",
    "ciudad_real": "Ciudad Real",
    "cordoba": "Córdoba",
    "guipuzcoa": "Guipúzcoa",
    "jaen": "Jaén",
    "la_rioja": "La Rioja",
    "las_palmas": "Las Palmas",
    "leon": "León",
    "madrid": "Comunidad de Madrid",
    "malaga": "Málaga",
    "santa_cruz_tenerife": "Santa Cruz de Tenerife",
    "vizcaya": "Vizcaya",
}

_INSULAR_SUBTERRITORIES: dict[str, str] = {
    "baleares": "Mallorca, Menorca, Ibiza y Formentera",
    "las_palmas": "Gran Canaria, Lanzarote y Fuerteventura",
    "santa_cruz_tenerife": "Tenerife, La Palma, La Gomera y El Hierro",
}


def _pretty_territory_label(territory_id: str) -> str:
    """Convierte un identificador canónico de territorio en etiqueta legible."""
    key = territory_id.lower().strip()
    if key in _PRETTY_TERRITORY_LABELS:
        return _PRETTY_TERRITORY_LABELS[key]
    return key.replace("_", " ").title()


def _format_supported_territories_display(
    indexed_territories: set[str],
) -> tuple[str, list[str]]:
    """Formatea dinámicamente la lista de territorios indexados colapsando autonomías multiprovinciales."""
    subprovinces_covered: set[str] = set()
    entries: list[str] = []

    for comm, provs in AUTONOMOUS_COMMUNITY_EXPANSION.items():
        if comm in indexed_territories:
            subprovs = [p for p in provs if p != comm]
            subprovinces_covered.update(subprovs)
            sub_labels = ", ".join(_pretty_territory_label(p) for p in subprovs)
            entries.append(
                f"{_pretty_territory_label(comm)} ({sub_labels})"
                if sub_labels
                else _pretty_territory_label(comm)
            )

    for terr in sorted(indexed_territories):
        if terr in AUTONOMOUS_COMMUNITY_EXPANSION or terr in subprovinces_covered:
            continue
        label = _pretty_territory_label(terr)
        if terr in _INSULAR_SUBTERRITORIES:
            label = f"{label} ({_INSULAR_SUBTERRITORIES[terr]})"
        entries.append(label)

    if not entries:
        return "ningún convenio autonómico/provincial indexado", []
    if len(entries) == 1:
        inline = f"**{entries[0]}**"
    else:
        inline = ", ".join(f"**{e}**" for e in entries[:-1]) + f" y **{entries[-1]}**"
    bullets = [f"- **{e}**" for e in entries]
    return inline, bullets


def _extract_genai_usage(response: Any) -> dict[str, int]:
    """Extrae contadores exactos de usage_metadata de una respuesta google-genai sin estimar ni calcular costes."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "thoughts_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
    input_tok = int(getattr(meta, "prompt_token_count", 0) or 0)
    output_tok = int(getattr(meta, "candidates_token_count", 0) or 0)
    thoughts_tok = int(getattr(meta, "thoughts_token_count", 0) or 0)
    cached_tok = int(getattr(meta, "cached_content_token_count", 0) or 0)
    total_tok = int(getattr(meta, "total_token_count", 0) or 0)
    if total_tok <= 0 and (input_tok or output_tok or thoughts_tok):
        total_tok = input_tok + output_tok + thoughts_tok
    return {
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "thoughts_tokens": thoughts_tok,
        "cached_tokens": cached_tok,
        "total_tokens": total_tok,
    }


def _call_structured_llm(
    prompt: str,
    schema: Type[T],
    preferred_model: str,
    system_instruction: str,
    temperature: float = 0.0,
    usage_out: Optional[list[dict[str, Any]]] = None,
    step_name: str = "llm_call",
    timeout_ms: Optional[int] = None,
    thinking_level: Optional[str] = None,
) -> tuple[T, str]:
    """Ejecuta una llamada estructurada (JSON Schema Pydantic) con failover multi-modelo, control de thoughts y reintento acotado ante 503/504."""
    client = get_genai_client()
    candidate_models: list[str] = [preferred_model]
    for m in GEMINI_FALLBACK_MODELS:
        if m not in candidate_models:
            candidate_models.append(m)

    now = time.time()
    fresh_models = [
        m
        for m in candidate_models
        if _MODEL_QUOTA_COOLDOWN_UNTIL.get(m, 0.0) <= now
        and _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now
    ]
    transient_cooled_models = sorted(
        (
            m
            for m in candidate_models
            if m not in fresh_models
            and _MODEL_QUOTA_COOLDOWN_UNTIL.get(m, 0.0) <= now
            and _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now + 3600.0
        ),
        key=lambda m: _MODEL_COOLDOWN_UNTIL.get(m, 0.0),
    )
    round_models: list[str] = fresh_models if fresh_models else transient_cooled_models
    if not round_models:
        round_models = [m for m in candidate_models if _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now + 3600.0] or candidate_models

    last_error: Optional[Exception] = None
    max_rounds = 1 + len(_TRANSIENT_RETRY_BACKOFFS)
    attempt_idx = 0

    for round_idx in range(max_rounds):
        quota_failed_in_call: set[str] = set()
        transient_failed_in_call: set[str] = set()

        for model_name in round_models:
            attempt_idx += 1
            t_attempt = time.perf_counter()
            attempt_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "thoughts_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
            }
            try:
                gen_config_kwargs: dict[str, Any] = {
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                }
                if timeout_ms is not None:
                    gen_config_kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
                if thinking_level is not None:
                    gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**gen_config_kwargs),
                    )
                except Exception as cfg_exc:
                    if "thinking" in str(cfg_exc).lower() and "thinking_config" in gen_config_kwargs:
                        fallback_kwargs = dict(gen_config_kwargs)
                        fallback_kwargs.pop("thinking_config", None)
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(**fallback_kwargs),
                        )
                    else:
                        raise

                attempt_usage = _extract_genai_usage(response)
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
                elapsed_ms = round((time.perf_counter() - t_attempt) * 1000, 1)
                if usage_out is not None:
                    usage_out.append(
                        {
                            "step": step_name,
                            "provider": "gemini_llm",
                            "model": model_name,
                            "status": "ok",
                            "api_calls": 1,
                            "latency_ms": elapsed_ms,
                            "attempt": attempt_idx,
                            **attempt_usage,
                        }
                    )
                _MODEL_COOLDOWN_UNTIL.pop(model_name, None)
                return parsed_obj, model_name
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                elapsed_ms = round((time.perf_counter() - t_attempt) * 1000, 1)
                is_quota_err = any(code in err_str for code in ("429", "RESOURCE_EXHAUSTED"))
                is_transient_err = any(
                    code in err_str for code in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED")
                )
                is_not_found_err = any(
                    code in err_str for code in ("404", "NOT_FOUND", "is not found", "not supported for generateContent")
                )
                is_truncated_err = "truncada prematuramente" in err_str.lower()
                fail_status = (
                    "failed_429"
                    if is_quota_err
                    else (
                        "failed_503_504"
                        if is_transient_err
                        else (
                            "failed_404"
                            if is_not_found_err
                            else ("failed_truncated" if is_truncated_err else "failed_other")
                        )
                    )
                )
                if usage_out is not None:
                    usage_out.append(
                        {
                            "step": step_name,
                            "provider": "gemini_llm",
                            "model": model_name,
                            "status": fail_status,
                            "api_calls": 1,
                            "latency_ms": elapsed_ms,
                            "attempt": attempt_idx,
                            "error_detail": err_str[:160],
                            **attempt_usage,
                        }
                    )
                if is_quota_err:
                    _MODEL_QUOTA_COOLDOWN_UNTIL[model_name] = time.time() + _QUOTA_COOLDOWN_SECONDS
                    quota_failed_in_call.add(model_name)
                    logger.warning(
                        "Modelo %s sin cuota (%s). Cooldown de %.0fs y rotando al siguiente modelo...",
                        model_name,
                        err_str[:120],
                        _QUOTA_COOLDOWN_SECONDS,
                    )
                    continue
                if is_not_found_err:
                    _MODEL_COOLDOWN_UNTIL[model_name] = time.time() + _NOT_FOUND_COOLDOWN_SECONDS
                    logger.warning(
                        "Modelo %s no encontrado/no soportado en la API (404: %s). Quarantined 24h y probando fallback...",
                        model_name,
                        err_str[:120],
                    )
                    continue
                if is_transient_err:
                    _MODEL_COOLDOWN_UNTIL[model_name] = time.time() + _COOLDOWN_SECONDS
                    transient_failed_in_call.add(model_name)
                    logger.warning(
                        "Modelo %s saturado/timeout (%s). Cooldown de %.0fs y rotando al siguiente modelo...",
                        model_name,
                        err_str[:120],
                        _COOLDOWN_SECONDS,
                    )
                    continue
                logger.warning("Error en modelo %s (%s). Probando fallback...", model_name, err_str[:120])
                continue

        if round_idx < len(_TRANSIENT_RETRY_BACKOFFS) and transient_failed_in_call:
            now_after = time.time()
            next_models = [
                m
                for m in candidate_models
                if m not in quota_failed_in_call
                and _MODEL_QUOTA_COOLDOWN_UNTIL.get(m, 0.0) <= now_after
                and _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now_after + 3600.0
            ]
            if not next_models:
                break
            backoff_sec = _TRANSIENT_RETRY_BACKOFFS[round_idx]
            logger.warning(
                "Todos los modelos sufrieron fallos transitorios/cuota (ronda %d/%d). Esperando %.0fs para reintentar %s...",
                round_idx + 1,
                max_rounds,
                backoff_sec,
                next_models,
            )
            time.sleep(backoff_sec)
            round_models = next_models
        else:
            break

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

        # 0. Atajo determinista CPU para saludos y despedidas conversacionales aislados (0 tokens, latencia <1ms)
        if not slots.pending_clarification_query and _GREETING_OR_FAREWELL_RE.match(query):
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            timings = dict(state.get("node_timings_ms") or {})
            timings["analyze_and_route"] = elapsed_ms
            models_used = dict(state.get("models_used") or {})
            models_used["analyze_and_route"] = "cpu_deterministic_shortcut"
            return {
                "effective_query": query,
                "intent": "HOSPITALITY_CORE",
                "motivo_clasificacion": "Saludo o cortesía inicial sin consulta jurídica (atajo determinista CPU)",
                "provincia_detectada": None,
                "territorio_normalizado": slots.active_territorio,
                "territorios_normalizados": [slots.active_territorio] if slots.active_territorio else [],
                "requiere_consulta_normativa_bbdd": False,
                "requiere_convenio_provincial": False,
                "requiere_puesto_o_categoria": False,
                "es_comparativa_global": False,
                "datos_completos": False,
                "route_decision": "ask_clarification",
                "oficio_coloquial_detectado": slots.active_oficio_coloquial,
                "categoria_profesional_aleh": slots.active_categoria_aleh,
                "area_funcional_aleh": slots.active_area_funcional_aleh,
                "query_busqueda_hibrida": query,
                "node_timings_ms": timings,
                "models_used": models_used,
            }

        supported_inline, _ = _format_supported_territories_display(self.indexed_territories)
        indexed_list_str = (
            f"{', '.join(sorted(self.indexed_territories))} "
            f"(cobertura legible: {supported_inline.replace('**', '')})"
        )

        system_instruction = (
            "Eres un Arquitecto Jurídico Laboral experto en el Sector de Hostelería y Turismo de España.\n"
            "Tu misión es ejecutar simultáneamente 4 tareas de alta precisión:\n"
            "1. RESOLUCIÓN DE CORREFERENCIA Y PROGRESIÓN CONVERSACIONAL (SIN RE-PREGUNTAR LO YA RESUELTO):\n"
            "   - CASO A (Desbloqueo de aclaración pendiente): Si en `SLOTS ACTIVOS DE LA SESIÓN` existe `pending_clarification_query` (una consulta previa que quedó "
            "detenida esperando provincia, puesto de trabajo o ambos) y el mensaje actual del usuario aporta el dato que faltaba (ej. 'Trabajo en Madrid', 'Es camarera', 'De cocinero en León'), "
            "clasifica como 'HOSPITALITY_CORE' (o 'UNSUPPORTED_PROVINCE' si la provincia no está soportada) y fusiona la pregunta pendiente "
            "con los datos conocidos de la sesión y el nuevo dato aportado en `consulta_autocontenida_resuelta`.\n"
            "   - CASO B (Repregunta, seguimiento o cambio de ámbito en conversación activa): Si `pending_clarification_query` es null y el usuario plantea una repregunta, "
            "seguimiento o cambio de ámbito geográfico, el `MENSAJE ACTUAL` prevalece sobre los slots previos. Si el usuario menciona una nueva provincia o solicita comparar "
            "varias provincias (ej. '¿y en Madrid?', 'compara con Barcelona', 'compara Madrid y León'), identifica y devuelve TODAS las provincias del turno actual en la lista `provincias_detectadas_en_turno_actual` "
            "para actualizar el ámbito de la sesión, heredando de los slots activos únicamente los datos no modificados (puesto, jornada, etc.). "
            "Si el usuario aporta un parámetro cuantitativo, temporal o contractual (ej. una fecha de alta/baja, horas de jornada, días trabajados, antigüedad o número de festivos) en seguimiento a un derecho cuantificable tratado en la sesión, formula `consulta_autocontenida_resuelta` pidiendo explícitamente el cálculo numérico exacto resultante para ese parámetro (no solo cómo afecta en abstracto). "
            "NUNCA arrastres ni vuelvas a incluir preguntas que ya fueron respondidas en turnos anteriores.\n\n"
            "2. GUARDRAIL DE DOMINIO Y DECISIÓN DE ACCESO A LA BASE DE DATOS NORMATIVA (`requiere_consulta_normativa_bbdd`):\n"
            "   - Clasifica como 'OUT_OF_SCOPE' cualquier consulta no legal o ajena al derecho laboral de hostelería/restauración/alojamiento/turismo "
            "(ej. temas generales no jurídicos, recetas, medicina, programación, o consultas legales de otras ramas como accidentes de tráfico, herencias, divorcios, o convenios de otros sectores como metal/construcción/comercio general) y marca `requiere_consulta_normativa_bbdd = False`.\n"
            f"   - Clasifica como 'UNSUPPORTED_PROVINCE' si la consulta es de hostelería pero el territorio aplicable es una provincia española "
            f"NO incluida en los territorios indexados actuales ({indexed_list_str}) y marca `requiere_consulta_normativa_bbdd = False`.\n"
            "   - Clasifica como 'HOSPITALITY_CORE' si el mensaje se enmarca en una conversación laboral de hostelería/turismo/catering/reparto de comida:\n"
            "     * Marca `requiere_consulta_normativa_bbdd = True` ÚNICAMENTE cuando exista una consulta o duda jurídico-laboral concreta que deba verificarse en los convenios/ALEH/Estatuto (o cuando el mensaje actual aporte un dato para desbloquear `pending_clarification_query`).\n"
            "     * Marca `requiere_consulta_normativa_bbdd = False` cuando el mensaje sea puramente conversacional o introductorio y NO contenga todavía ninguna pregunta jurídica concreta ni desbloquee una pregunta pendiente (ej. saludos iniciales, presentaciones personales sin pregunta, agradecimientos, despedidas o anuncios genéricos sin cuestión laboral formulada).\n\n"
            "3. GATEKEEPER CONTEXTUAL GENERALISTA (PROVINCIA Y PUESTO/CATEGORÍA):\n"
            "   - `provincias_detectadas_en_turno_actual`: Identifica por contexto semántico TODAS las **provincias españolas donde se ubican físicamente el/los centro(s) de trabajo** mencionados en el `MENSAJE ACTUAL`, como lista en minúsculas y sin tildes (ej. `['madrid']`, o `['madrid', 'leon']` en comparativas).\n"
            "     * Si el usuario menciona uno o varios municipios, localidades o comarcas (incluso con errores tipográficos), resuelve y devuelve TODAS sus provincias administrativas oficiales dentro de la lista.\n"
            "     * **REGLA DE DESAMBIGUACIÓN ENTIDAD vs. JURISDICCIÓN**: Distingue siempre entre la ubicación geográfica real del centro de trabajo y los nombres comerciales de negocios, marcas, cadenas o calles que contengan topónimos. Si existe ambigüedad sobre la ubicación geográfica real o si no se menciona ubicación en este turno, devuelve `[]`.\n"
            "   - `requiere_convenio_provincial`: Marca `True` SIEMPRE que la consulta trate sobre salarios, tablas salariales, sueldos, pluses, "
            "nocturnidad, horas extra, vacaciones, jornada anual, descansos, licencias/permisos retribuidos, bajas médicas (complemento IT), "
            "categorías/clases de establecimientos del convenio o régimen concreto de un centro de trabajo. "
            "Marca `False` cuando la consulta verse exclusivamente sobre normativa estatal general (Estatuto de los Trabajadores o VI ALEH) sin depender de cuantías o reglas provinciales (ej. áreas funcionales, clasificación de grupos, régimen disciplinario/faltas/sanciones, prescripción de faltas o mínimos generales estatales).\n"
            "   - `requiere_puesto_o_categoria`: Marca `True` cuando para responder la pregunta con exactitud sea imprescindible saber el puesto o categoría profesional del trabajador "
            "(ej. consultar cuál es su sueldo/salario base, calcular el precio de su hora extra, nivel salarial o periodo de prueba según grupo profesional). "
            "Si la consulta incluye varias preguntas y al menos una de ellas depende del puesto (ej. pregunta por el sueldo Y por las pagas extras), marca `True`. "
            "Marca `False` únicamente cuando TODA la consulta sea general del convenio o estatuto y aplique igual a todos los puestos (ej. cuántos días de vacaciones anuales corresponden, cuántas pagas extras hay al año, jornada máxima anual o días de permiso por mudanza/matrimonio).\n"
            "   - `es_comparativa_global`: Marca `True` si el usuario solicita una comparativa general o abierta entre todas las provincias "
            "('de cada provincia', 'todas las provincias', 'qué provincia paga más') o si indica explícitamente que no trabaja en una provincia concreta o que 'da igual la provincia'.\n\n"
            "4. TRADUCTOR DE OFICIOS COLOQUIALES AL VI ALEH (PROHIBIDO ADIVINAR EL PUESTO POR EL TIPO DE NEGOCIO):\n"
            "   - **REGLA ESTRICTA**: Si el usuario solo menciona el tipo de establecimiento ('trabaja en un bar', 'en un restaurante', 'en un hotel', 'en una cafetería') pero **NO especifica qué puesto u oficio desempeña** (ni consta en `SLOTS ACTIVOS DE LA SESIÓN`), DEBES dejar `oficio_coloquial_detectado = null` y `categoria_profesional_aleh = null`. Jamás des por hecho ni inventes un puesto.\n"
            "   - Cuando el usuario SÍ indique un puesto u oficio (en el turno actual o heredado de los slots de la sesión), mapéalo a su denominación técnica en el VI ALEH:\n"
            "     * 'friegaplatos' / 'office' / 'pica' -> 'Auxiliar de limpieza / Marmitón / Freganchín (Área Funcional 2ª Cocina o 3ª Restauración, Grupo Profesional III / Nivel salarial inferior)'\n"
            "     * 'pinche' / 'ayudante de cocina' -> 'Ayudante/a de cocina / Auxiliar de cocina (Área Funcional 2ª Cocina y Economato, Grupo Profesional II/III)'\n"
            "     * 'cocinero' -> 'Cocinero/a (Área Funcional 2ª Cocina y Economato, Grupo Profesional II)'\n"
            "     * 'chef' / 'jefe de cocina' / 'jefe de partida' -> 'Jefe/a de cocina / Segundo/a Jefe/a de cocina / Jefe/a de partida (Área Funcional 2ª Cocina y Economato, Grupos Profesionales I y II)'\n"
            "     * 'camarero' / 'camarera' / 'chico de barra' / 'mesero' -> 'Camarero/a, Sumiller, Barman o Ayudante/a de camarero/a (Área Funcional 3ª Restauración, Grupo Profesional II/III)'\n"
            "     * 'rider' / 'repartidor' / 'delivery' -> 'Repartidor/a de comida a domicilio (Área Funcional 3ª Restauración / Disposición Adicional ALEH y Convenio Provincial)'\n"
            "     * 'camarera de pisos' / 'kelly' / 'limpiadora de habitaciones' -> 'Camarero/a de pisos / Auxiliar de pisos y limpieza (Área Funcional 4ª Pisos y Limpieza, Grupos Profesionales II y III)'\n"
            "     * 'recepcionista' / 'botones' / 'conserje' -> 'Recepcionista / Conserje / Auxiliar de recepción y conserjería (Área Funcional 1ª Recepción, Conserjería, Relaciones Públicas, Administración y Gestión)'\n"
            "     * 'mantenimiento' / 'socorrista' / 'animador' -> 'Área Funcional 5ª (Mantenimiento y Servicios Auxiliares) o Área Funcional 6ª (Servicios Complementarios)'\n"
            "   - Construye `query_busqueda_hibrida` combinando los conceptos clave de `consulta_autocontenida_resuelta` (incluyendo siempre el tipo y categoría/estrellas/tenedores del establecimiento si se menciona), la categoría oficial del ALEH, el área funcional "
            "y el territorio aplicable (NOTA: incluye los términos 'tablas salariales salario base gratificaciones extraordinarias pagas extras' cuando el turno actual pregunte por sueldos, salarios, nómina, retribuciones o cálculo de horas extras)."
        )

        prompt = (
            f"SLOTS ACTIVOS DE LA SESIÓN (WORKING MEMORY): {json.dumps(slots.model_dump(), ensure_ascii=False)}\n"
            f"VENTANA COMPACTA DE TURNOS RECIENTES: {json.dumps(history, ensure_ascii=False)}\n"
            f"MENSAJE ACTUAL DEL USUARIO: {query}\n"
            f"TERRITORIOS SOPORTADOS EN ÍNDICE QDRANT: {indexed_list_str}"
        )

        step_usage: list[dict[str, Any]] = []
        analysis, model_used = _call_structured_llm(
            prompt=prompt,
            schema=QueryAnalysisSchema,
            preferred_model=GEMINI_LITE_MODEL,
            system_instruction=system_instruction,
            temperature=0.0,
            usage_out=step_usage,
            step_name="analyze_and_route",
            timeout_ms=15000,
            thinking_level="minimal",
        )

        # 1. Resolución de Provincias (list[str]) con Precedencia Temporal Estricta (Turno Actual > Slots):
        def _parse_prov_items(raw_items: list[str]) -> list[str]:
            out_provs: list[str] = []
            seen_p: set[str] = set()
            for item in raw_items:
                for tok in re.split(r"[,/|]|\s+y\s+", str(item or "")):
                    canon = _canonicalize_llm_province(tok.strip(" _"))
                    if canon and canon not in seen_p:
                        seen_p.add(canon)
                        out_provs.append(canon)
            return out_provs

        current_turn_provs = _parse_prov_items(list(analysis.provincias_detectadas_en_turno_actual or []))
        candidate_provs = (
            current_turn_provs
            if current_turn_provs
            else _parse_prov_items([slots.active_territorio] if slots.active_territorio else [])
        )
        raw_prov: Optional[str] = (
            None
            if not candidate_provs
            else (candidate_provs[0] if len(candidate_provs) == 1 else ",_".join(candidate_provs))
        )

        intent = analysis.intent
        territorio_norm: Optional[str] = None
        territorios_norm_list: list[str] = []

        if intent != "OUT_OF_SCOPE" and candidate_provs:
            if all(c in self.indexed_territories for c in candidate_provs):
                intent = "HOSPITALITY_CORE"
                territorio_norm = raw_prov
                territorios_norm_list = candidate_provs
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

        # 4. Decisión determinista de enrutamiento (Acceso a BBDD + Gatekeeper de Provincia y Puesto/Categoría)
        has_pending = bool(slots.pending_clarification_query)
        was_unsupported_reclassified = (
            analysis.intent == "UNSUPPORTED_PROVINCE" and intent == "HOSPITALITY_CORE"
        )
        requiere_bbdd = bool(
            analysis.requiere_consulta_normativa_bbdd
            or has_pending
            or was_unsupported_reclassified
        )
        missing_prov = bool(analysis.requiere_convenio_provincial and not territorio_norm)
        missing_role = bool(analysis.requiere_puesto_o_categoria and not cat_aleh_resuelta)
        is_global_comp = bool(
            getattr(analysis, "es_comparativa_global", False)
            or _GLOBAL_COMPARISON_RE.search(query)
            or _GLOBAL_COMPARISON_RE.search(effective_query)
        )

        if intent in ("OUT_OF_SCOPE", "UNSUPPORTED_PROVINCE"):
            requiere_bbdd = False
            datos_completos = False
            route_decision: RouteDecision = "block_out_of_scope"
        elif not requiere_bbdd or missing_prov or missing_role:
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
            "territorios_normalizados": territorios_norm_list,
            "requiere_consulta_normativa_bbdd": requiere_bbdd,
            "requiere_convenio_provincial": analysis.requiere_convenio_provincial,
            "requiere_puesto_o_categoria": analysis.requiere_puesto_o_categoria,
            "es_comparativa_global": is_global_comp,
            "datos_completos": datos_completos,
            "route_decision": route_decision,
            "oficio_coloquial_detectado": oficio_resuelto,
            "categoria_profesional_aleh": cat_aleh_resuelta,
            "area_funcional_aleh": area_aleh_resuelta,
            "query_busqueda_hibrida": analysis.query_busqueda_hibrida or effective_query,
            "node_timings_ms": timings,
            "models_used": models_used,
            "usage_events": step_usage,
        }

    def block_out_of_scope_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo de Bloqueo Determinista (0 tokens de generación): rechaza consultas ajenas o provincias no indexadas."""
        t0 = time.perf_counter()
        intent = state.get("intent", "OUT_OF_SCOPE")
        prov = state.get("provincia_detectada")

        supported_inline, _ = _format_supported_territories_display(self.indexed_territories)

        if intent == "UNSUPPORTED_PROVINCE" and prov:
            prov_pretty = _pretty_territory_label(prov)
            msg = (
                f"### Aviso de Cobertura Territorial\n\n"
                f"He detectado que tu consulta pertenece al ámbito laboral de **Hostelería**, pero hace referencia a **{prov_pretty}**.\n\n"
                f"- **Estado actual del índice normativo**: Están indexados y verificados los convenios colectivos de "
                f"{supported_inline}, además de las dos normas estatales "
                f"(**Estatuto de los Trabajadores** y **VI Acuerdo Laboral Estatal de Hostelería — ALEH VI**).\n"
                f"- **Por rigor jurídico (Cero Alucinación)**: No puedo aplicarte tablas salariales ni condiciones de otra provincia ni responderte solo con el Estatuto de los Trabajadores "
                f"ocultando las mejoras del Convenio Colectivo de {prov_pretty}.\n\n"
                f"Si deseas consultar únicamente la **clasificación profesional estatal (Áreas Funcionales del VI ALEH)** o los **mínimos generales del Estatuto de los Trabajadores**, "
                f"indícalo expresamente o realiza tu consulta para uno de los territorios disponibles ({supported_inline})."
            )
        else:
            msg = (
                "### Consulta Fuera del Dominio de Especialización\n\n"
                "Soy **AiHoreca**, un asistente jurídico-laboral especializado exclusivamente en el **Sector de Hostelería y Turismo en España**.\n\n"
                f"- **Motivo de bloqueo**: {state.get('motivo_clasificacion', 'La consulta no pertenece al ámbito laboral de hostelería.')}\n"
                f"- **Ámbito de este RAG**: Resuelvo consultas sobre el **Estatuto de los Trabajadores**, el **VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)** y los **Convenios Colectivos de Hostelería** indexados ({supported_inline}).\n"
                "- **No soportado**: Consultas no jurídicas, otras ramas del Derecho (civil, penal, tráfico, mercantil, fiscal) o convenios colectivos de otros sectores."
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
        requiere_bbdd = bool(state.get("requiere_consulta_normativa_bbdd", True))
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

        _, supported_bullets = _format_supported_territories_display(self.indexed_territories)
        territories_block = "\n".join(supported_bullets)

        is_global_comp = bool(
            state.get("es_comparativa_global")
            or _GLOBAL_COMPARISON_RE.search(state.get("query", ""))
            or _GLOBAL_COMPARISON_RE.search(state.get("effective_query", ""))
        )

        if not requiere_bbdd:
            profile_parts: list[str] = []
            if terr:
                profile_parts.append(f"territorio **{_pretty_territory_label(terr)}**")
            if oficio or cat_aleh:
                profile_parts.append(f"puesto **{oficio or cat_aleh}**")
            profile_note = (
                f"\n\nHe tomado nota de tu contexto activo ({', '.join(profile_parts)})."
                if profile_parts
                else ""
            )
            msg = (
                "### ¡Hola! Soy AiHoreca — Asistente Jurídico-Laboral de Hostelería\n\n"
                "Estoy especializado en resolver dudas laborales del **Sector de Hostelería y Turismo en España** "
                "cruzando el **Estatuto de los Trabajadores**, el **VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)** "
                "y los **Convenios Colectivos Provinciales y Autonómicos**."
                f"{profile_note}"
                f"{aleh_note}\n\n"
                "**¿Qué consulta laboral necesitas resolver?** *(Para cálculos de salario, jornada, vacaciones o permisos, "
                "indícame tu **provincia/localidad** y tu **puesto de trabajo**):*\n"
                f"{territories_block}"
            )
            warn_msg = "Cortocircuito conversacional activado (sin pregunta normativa concreta: 0 llamadas a BBDD)."
        elif is_global_comp:
            msg = (
                "### Comparativa Interprovincial y Suelo Normativo Estatal (Hostelería)\n\n"
                "En el sector de hostelería y turismo en España **no existe un convenio colectivo estatal único que fije los salarios, jornadas ni vacaciones**, "
                "sino una estructura descentralizada compuesta por **más de 50 convenios colectivos provinciales y autonómicos independientes**, "
                "cada uno acordado en su propia mesa de negociación colectiva.\n\n"
                "#### 1. Mínimos Legales Comunes: Estatuto de los Trabajadores (ET) y VI ALEH\n"
                "Con independencia de la provincia o territorio donde se presten servicios, todo profesional de hostelería cuenta con el siguiente suelo normativo indisponible:\n"
                "- **Vacaciones (Art. 38.1 ET)**: Mínimo legal estatal de **30 días naturales** al año (los convenios provinciales pueden igualar o mejorar esta duración, por ejemplo fijando días laborables o festivos adicionales, pero nunca rebajarla).\n"
                "- **Jornada máxima (Art. 34.1 ET)**: Límite general de **40 horas semanales** de promedio en cómputo anual (~1.826 horas anuales). La mayoría de convenios provinciales de hostelería mejoran este tope situándolo habitualmente entre 1.770 h y 1.800 h al año.\n"
                "- **Pagas extraordinarias (Art. 31 ET)**: Mínimo de **dos gratificaciones extraordinarias** al año (verano y Navidad), con opción a una 3ª paga pactada en numerosos convenios provinciales.\n"
                "- **Descanso semanal (Art. 37.1 ET)**: Mínimo de **día y medio ininterrumpido** semanal (acumulable en periodos de hasta 14 días).\n"
                "- **Clasificación y régimen disciplinario**: Homogeneizado a nivel estatal por el **VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)** (Áreas Funcionales y Grupos Profesionales del I al III).\n\n"
                "#### 2. Cómo generar una tabla comparativa exacta y contrastada\n"
                "Para poder contrastar las cláusulas y tablas oficiales vigentes en una tabla comparativa detallada, **indícame qué provincias o territorios deseas comparar** (puedes elegir las que quieras de entre las disponibles actualmente):\n\n"
                f"{territories_block}\n\n"
                "*(Por ejemplo: «Compara las vacaciones y la jornada anual entre **Madrid**, **Barcelona**, **Baleares** y **Álava**» o cualquier combinación de las anteriores)*."
            )
            warn_msg = "Gatekeeper contextual ofreció el suelo estatutario estatal (ET) y solicitó seleccionar las provincias deseadas para la comparativa interprovincial."
        elif missing_prov and missing_role:
            msg = (
                "### Información Requerida para el Cálculo (Provincia y Puesto de Trabajo)\n\n"
                "Para darte una respuesta jurídicamente exacta y sin asumir datos que alterarían las cifras, **necesito conocer dos datos básicos**:\n\n"
                "1. **¿En qué provincia o localidad está ubicado el centro de trabajo?** (Cada provincia tiene su propio Convenio Colectivo de Hostelería con tablas salariales, número de pagas extras y jornada anual propios).\n"
                "2. **¿Qué puesto u oficio desempeña dentro del establecimiento?** (Por ejemplo: *camarero/a, ayudante de camarero/a, cocinero/a, ayudante de cocina/pinche, friegaplatos/limpieza, recepcionista...*, ya que el salario base varía según el Grupo Profesional y Nivel salarial).\n\n"
                "**Territorios disponibles actualmente en el índice:**\n"
                f"{territories_block}"
            )
            warn_msg = "Gatekeeper contextual detuvo la consulta por falta de provincia y puesto de trabajo."
        elif missing_role and not missing_prov:
            terr_pretty = _pretty_territory_label(terr or "")
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
                f"{territories_block}"
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

    @staticmethod
    def _build_retrieval_queries(state: dict[str, Any]) -> tuple[str, str, Optional[str]]:
        """Construye de forma uniforme la query enriquecida, query de rerank y contexto del puesto."""
        effective_query = state.get("effective_query") or state.get("query", "")
        enriched_query = state.get("query_busqueda_hibrida") or effective_query
        oficio = state.get("oficio_coloquial_detectado")
        cat_aleh = state.get("categoria_profesional_aleh")
        area_aleh = state.get("area_funcional_aleh")

        role_parts = [p for p in (oficio, cat_aleh, area_aleh) if p]
        role_context = " | ".join(role_parts) if role_parts else None
        rerank_query = (
            f"{effective_query} ({role_context})" if role_context else effective_query
        )
        return enriched_query, rerank_query, role_context

    def retrieve_state_layers_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Nodo B1 (Rama Send Estatal): recupera candidatos de Estatuto de los Trabajadores y VI ALEH."""
        t0 = time.perf_counter()
        enriched_query, _, role_context = self._build_retrieval_queries(state)
        step_usage: list[dict[str, Any]] = []
        docs = self.vector_store.retrieve_state_layer_candidates(
            query=enriched_query,
            top_k_per_layer=RETRIEVAL_TOP_K_PER_LAYER,
            use_cohere_rerank=True,
            role_context=role_context,
            usage_out=step_usage,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "branch_docs": docs,
            "branch_timings": [("state_layers", elapsed_ms)],
            "usage_events": step_usage,
        }

    def retrieve_territory_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Nodo B1 (Rama Send Territorial): recupera candidatos estratificados de un único convenio."""
        t0 = time.perf_counter()
        enriched_query, rerank_query, role_context = self._build_retrieval_queries(state)
        target_terr = str(state.get("target_territorio") or "").strip().lower()
        docs: list[RetrievedDocument] = []
        step_usage: list[dict[str, Any]] = []
        if target_terr:
            docs = self.vector_store.retrieve_territory_candidates(
                query=enriched_query,
                provincia=target_terr,
                top_k_per_layer=RETRIEVAL_TOP_K_PER_LAYER,
                use_cohere_rerank=True,
                role_context=role_context,
                usage_out=step_usage,
                topic_source_query=rerank_query,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "branch_docs": docs,
            "branch_timings": [(f"territory:{target_terr or 'none'}", elapsed_ms)],
            "usage_events": step_usage,
        }

    @staticmethod
    def _doc_matches_territory_key(doc: RetrievedDocument, terr_key: str) -> bool:
        """Comprueba si un fragmento provincial pertenece al territorio solicitado (incluye autonomías multiprovinciales)."""
        if doc.capa_normativa != "convenio_provincial":
            return False
        t_norm = terr_key.lower().strip()
        if doc.matched_territorio and doc.matched_territorio.lower().strip() == t_norm:
            return True
        doc_t = (doc.territorio or "").lower().strip()
        if doc_t == t_norm:
            return True
        return any(
            doc_t == comm and t_norm in provs
            for comm, provs in AUTONOMOUS_COMMUNITY_EXPANSION.items()
        )

    @staticmethod
    def _canonicalize_legal_ref(ref_type: str, raw_match: str) -> str:
        """Normaliza referencias cruzadas internas ('Art. 24º', 'Anexo II', 'Disposición adicional décima')."""
        flat = re.sub(r"\s+", " ", _normalize_text(raw_match)).strip(" .,:;()")
        if ref_type == "articulo":
            m = re.search(
                r"art(?:ic(?:ulo|le)|igo)?\.?\s*(\d+)(?:\.\d+)*\s*[ºª]?(?:\s+(bis|ter|quater))?",
                flat,
            )
            if m:
                suffix = f" {m.group(2)}" if m.group(2) else ""
                return f"art {m.group(1)}{suffix}"
        elif ref_type == "anexo":
            m = re.search(r"anexo\s+([ivxlcdm]+|\d+|unico)\b", flat)
            if m:
                return f"anexo {m.group(1)}"
        elif ref_type == "disposicion":
            m = re.search(
                r"disposicion\s+(adicional|transitoria|final)\s+([a-z0-9ºª]+)",
                flat,
            )
            if m:
                return f"disposicion {m.group(1)} {m.group(2)}"
        return flat

    @classmethod
    def _doc_covers_cross_ref(cls, doc: RetrievedDocument, ref: dict[str, str]) -> bool:
        """Comprueba determinísticamente si la cabecera o línea de título de un fragmento cubre la referencia buscada."""
        body_clean = doc.text.split("\n\n", 1)[-1].lstrip()
        first_line = body_clean.splitlines()[0][:140] if body_clean else ""
        heading_line = (
            first_line
            if re.match(r"^(?:#+\s*|\*\*|_+)?(?:art|anexo|disposici[óo]n|tabla)", first_line, flags=re.IGNORECASE)
            else ""
        )
        surface = _normalize_text(f"{doc.titulo_o_capitulo} | {doc.articulo} | {heading_line}")
        ref_type = ref.get("ref_type", "")
        ref_label = ref.get("ref_label", "")
        if not ref_label:
            return False

        if ref_type == "articulo":
            surface_art = re.sub(
                r"\bart(?:ic(?:ulo|le)|igo)?\.?\s*(\d+)(?:\.\d+)*\s*[ºª]?(?:\s+(bis|ter|quater))?",
                lambda m: f"art {m.group(1)}" + (f" {m.group(2)}" if m.group(2) else ""),
                surface,
            )
            return bool(re.search(rf"(?<!\w){re.escape(ref_label)}(?!\w)", surface_art))

        if ref_type == "anexo":
            roman_arabic = {
                "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
                "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
            }
            variants = {ref_label}
            tok = ref_label.removeprefix("anexo ").strip()
            if tok in roman_arabic:
                variants.add(f"anexo {roman_arabic[tok]}")
            for r_k, a_v in roman_arabic.items():
                if tok == a_v:
                    variants.add(f"anexo {r_k}")
            return any(bool(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", surface)) for v in variants)

        return bool(re.search(rf"(?<!\w){re.escape(ref_label)}(?!\w)", surface))

    @classmethod
    def _extract_missing_cross_refs(
        cls,
        selected_docs: list[RetrievedDocument],
        territorios: list[str],
        enriched_query: str,
        role_context: Optional[str],
    ) -> list[dict[str, str]]:
        """Detecta remisiones normativas internas ausentes o cobertura territorial baja tras Cohere Rerank."""
        disp_re = re.compile(
            r"(?i)disposici[oó]n\s+(?:adicional|transitoria|final)\s+"
            r"(?:[uú]nica|primeira|primera|segunda|terceira|tercera|cuarta|quinta|sexta|s[eé]ptima|octava|novena|"
            r"d[eé]cima|und[eé]cima|duod[eé]cima|decimotercera|decimocuarta|\d+[ªº]?)\b"
        )
        anexo_re = re.compile(r"(?i)\banexo\s+(?:[IVXLCDM]+|\d+|[uú]nico)\b")
        remission_art_re = re.compile(
            r"(?i)(?:previst[oa]s?|establecid[oa]s?|regulad[oa]s?|dispuest[oa]s?|indicad[oa]s?|"
            r"recogid[oa]s?|recollid[oa]s?|remite(?:n)?|rem[íi]tese|conforme|seg[úu]n(?:\s+lo\s+(?:previsto|dispuesto))?)"
            r"\s+(?:en\s+(?:el|los|o)|al|ao|no|do|a\s+lo\s+(?:dispuesto|previsto)\s+en\s+(?:el|o))\s+"
            r"(art(?:[íi]c(?:ulo|le)|igo)?\.?\s*\d+(?:\.\d+)*\s*[ºª]?(?:\s+(?:bis|ter|quater))?\b)"
        )
        ext_suffix_re = re.compile(
            r"(?i)^\s*(?:y\s+\d+\s*)?(?:de(?:l| la)?\s+)?(?:"
            r"estatuto(?:\s+de\s+los\s+trabajadores)?|e\.?t\.?\b|aleh\b|vi\s+aleh|"
            r"acuerdo\s+laboral\s+estatal|real\s+decreto|ley\b|c[óo]digo\b|"
            r"constituc|seguridad\s+social|texto\s+refundido|r\.?d\.?\b|directiva|reglamento\s+ue)"
        )
        ext_prefix_re = re.compile(
            r"(?i)\b(?:estatuto\s+de\s+los\s+trabajadores|vi\s+aleh|real\s+decreto)\b"
        )

        missing_refs: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for terr in territorios:
            terr_docs = [d for d in selected_docs if cls._doc_matches_territory_key(d, terr)]
            max_terr_score = max(((d.rerank_score or 0.0) for d in terr_docs), default=0.0)
            if not terr_docs or max_terr_score < 0.20:
                key = (terr, "low_coverage", enriched_query[:80])
                if key not in seen_keys:
                    seen_keys.add(key)
                    missing_refs.append(
                        {
                            "territorio": terr,
                            "ref_type": "low_coverage",
                            "ref_label": enriched_query[:120],
                            "query_hint": f"{enriched_query} {role_context or ''}".strip(),
                        }
                    )
                if len(missing_refs) >= 3:
                    break
                continue

            textual_top = [
                d
                for d in terr_docs
                if d.tipo_contenido in ("articulo", "anexo_o_disposicion")
                and not any(
                    tok in _normalize_text(d.articulo)
                    for tok in ("preambulo", "encabezado", "indice")
                )
            ][:2]

            per_terr_added = 0
            for doc in textual_top:
                if per_terr_added >= 2 or len(missing_refs) >= 3:
                    break
                body_flat = re.sub(r"\s+", " ", doc.text.split("\n\n", 1)[-1])

                candidates_found: list[tuple[str, str, int, int]] = []
                for m in disp_re.finditer(body_flat):
                    candidates_found.append(("disposicion", m.group(0), m.start(), m.end()))
                for m in anexo_re.finditer(body_flat):
                    candidates_found.append(("anexo", m.group(0), m.start(), m.end()))
                for m in remission_art_re.finditer(body_flat):
                    candidates_found.append(("articulo", m.group(1), m.start(1), m.end(1)))

                for ref_type, raw_match, start_pos, end_pos in candidates_found:
                    if per_terr_added >= 2 or len(missing_refs) >= 3:
                        break
                    suffix = body_flat[end_pos : end_pos + 70]
                    prefix = body_flat[max(0, start_pos - 35) : start_pos]
                    if ext_suffix_re.search(suffix) or ext_prefix_re.search(prefix):
                        continue
                    norm_label = cls._canonicalize_legal_ref(ref_type, raw_match)
                    if not norm_label:
                        continue
                    ref_item = {
                        "territorio": terr,
                        "ref_type": ref_type,
                        "ref_label": norm_label,
                        "raw_label": raw_match,
                    }
                    if any(cls._doc_covers_cross_ref(td, ref_item) for td in terr_docs):
                        continue
                    dedup_key = (terr, ref_type, norm_label)
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        missing_refs.append(ref_item)
                        per_terr_added += 1

        return missing_refs[:3]

    def verify_retrieval_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo B2 (Fan-In + Verificador CPU Determinista): reranquea ramas y detecta remisiones ausentes."""
        t0 = time.perf_counter()
        enriched_query, rerank_query, role_context = self._build_retrieval_queries(state)
        territorios = list(state.get("territorios_normalizados") or [])
        branch_docs = list(state.get("branch_docs") or [])
        branch_timings = list(state.get("branch_timings") or [])

        t_rerank0 = time.perf_counter()
        step_usage: list[dict[str, Any]] = []
        if branch_docs:
            selected_docs = self.vector_store.rerank_and_select_by_territory(
                candidates=branch_docs,
                query=enriched_query,
                territorios=territorios,
                final_top_k=RETRIEVAL_FINAL_TOP_K,
                use_cohere_rerank=True,
                rerank_query=rerank_query,
                role_context=role_context,
                usage_out=step_usage,
            )
        else:
            selected_docs = []
        rerank_ms = (time.perf_counter() - t_rerank0) * 1000.0
        max_branch_ms = max((float(ms) for _, ms in branch_timings), default=0.0)

        t_verify0 = time.perf_counter()
        hop = int(state.get("retrieval_hop") or 0)
        if hop < 1 and territorios:
            missing_refs = self._extract_missing_cross_refs(
                selected_docs=selected_docs,
                territorios=territorios,
                enriched_query=enriched_query,
                role_context=role_context,
            )
        else:
            missing_refs = []
        verify_ms = round((time.perf_counter() - t_verify0) * 1000.0, 2)

        timings = dict(state.get("node_timings_ms") or {})
        timings["parallel_retrieval"] = round(max_branch_ms + rerank_ms, 1)
        timings["verify_retrieval"] = verify_ms
        models_used = dict(state.get("models_used") or {})
        models_used["parallel_retrieval"] = "gemini-embedding-001 + Qdrant/bm25 + rerank-multilingual-v3.0"
        models_used["verify_retrieval"] = "deterministic-cpu-verifier"

        return {
            "retrieved_docs": selected_docs,
            "missing_cross_refs": missing_refs,
            "needs_retrieval_expansion": bool(missing_refs),
            "node_timings_ms": timings,
            "models_used": models_used,
            "usage_events": step_usage,
        }

    def expand_retrieval_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo B2 (2º Salto Condicional Acotado): recupera remisiones normativas ausentes para ese territorio."""
        t0 = time.perf_counter()
        enriched_query, _, role_context = self._build_retrieval_queries(state)
        missing_refs = list(state.get("missing_cross_refs") or [])
        current_docs: list[RetrievedDocument] = list(state.get("retrieved_docs") or [])
        branch_docs: list[RetrievedDocument] = list(state.get("branch_docs") or [])
        effective_query = str(state.get("effective_query") or state.get("query") or "")
        split_territories = self.vector_store.get_split_subsector_territories()

        seen_ids = {d.chunk_id for d in current_docs}
        added_docs: list[RetrievedDocument] = []
        step_usage: list[dict[str, Any]] = []

        for ref in missing_refs:
            if len(added_docs) >= 3:
                break
            terr = ref.get("territorio", "").strip().lower()
            ref_type = ref.get("ref_type", "")
            raw_or_label = ref.get("raw_label") or ref.get("ref_label", "")
            if not terr:
                continue

            found_for_ref: list[RetrievedDocument] = []
            # Paso A (0 ms): revisar primero si la rama paralela ya había traído el fragmento en branch_docs
            for bdoc in branch_docs:
                if bdoc.chunk_id in seen_ids or not self._doc_matches_territory_key(bdoc, terr):
                    continue
                if ref_type == "low_coverage" or self._doc_covers_cross_ref(bdoc, ref):
                    item = bdoc.model_copy()
                    item.matched_territorio = terr
                    found_for_ref.append(item)
                    if len(found_for_ref) >= 2:
                        break

            # Paso B: búsqueda dirigida en Qdrant solo si no estaba en los candidatos en memoria
            if not found_for_ref:
                target_query = (
                    ref.get("query_hint") or enriched_query
                    if ref_type == "low_coverage"
                    else f"{raw_or_label} {terr} {enriched_query[:80]}".strip()
                )
                tipo_contenido = "articulo" if ref_type == "articulo" else None
                agreement_subsector = (
                    self.vector_store._extract_query_agreement_subsector(
                        effective_query, role_context=role_context
                    )
                    if terr in split_territories
                    else None
                )
                hits = self.vector_store.hybrid_search(
                    query=target_query,
                    capa_normativa="convenio_provincial",
                    provincia=terr,
                    tipo_contenido=tipo_contenido,
                    subsector=agreement_subsector,
                    top_k=4,
                    include_vectors=False,
                    usage_out=step_usage,
                    step_name=f"embed_expand_{terr}",
                )
                for h in hits:
                    if h.chunk_id in seen_ids:
                        continue
                    if ref_type != "low_coverage" and not self._doc_covers_cross_ref(h, ref):
                        continue
                    item = h.model_copy()
                    item.matched_territorio = terr
                    found_for_ref.append(item)
                    if len(found_for_ref) >= 2:
                        break

            for doc_add in found_for_ref:
                if doc_add.chunk_id not in seen_ids and len(added_docs) < 3:
                    added_docs.append(doc_add)
                    seen_ids.add(doc_add.chunk_id)

        updated_docs = current_docs + added_docs
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        timings = dict(state.get("node_timings_ms") or {})
        timings["expand_retrieval"] = elapsed_ms
        models_used = dict(state.get("models_used") or {})
        models_used["expand_retrieval"] = "targeted-qdrant-hop"

        if added_docs:
            logger.info(
                "Expansión B2 (hop 1): recuperados %d fragmentos complementarios para %s.",
                len(added_docs),
                [f"{r.get('territorio')}:{r.get('ref_label')}" for r in missing_refs],
            )

        return {
            "retrieved_docs": updated_docs,
            "retrieval_hop": int(state.get("retrieval_hop") or 0) + 1,
            "needs_retrieval_expansion": False,
            "node_timings_ms": timings,
            "models_used": models_used,
            "usage_events": step_usage,
        }

    def parallel_retrieval_node(self, state: LegalGraphState) -> LegalGraphState:
        """Envoltorio retrocompatible: recupera en 3 capas normativas y consolida en una sola llamada."""
        t0 = time.perf_counter()
        enriched_query, rerank_query, role_context = self._build_retrieval_queries(state)
        provincia = state.get("territorio_normalizado")

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

    def synthesize_response_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo 4: Síntesis Jurídica con Jerarquía Normativa (con soporte de autocorrección tras auditoría)."""
        t0 = time.perf_counter()
        raw_query = state.get("query", "")
        effective_query = state.get("effective_query") or raw_query
        provincia = state.get("territorio_normalizado")
        cat_aleh = state.get("categoria_profesional_aleh")
        area_aleh = state.get("area_funcional_aleh")
        history = state.get("conversation_history") or []
        docs: list[RetrievedDocument] = state.get("retrieved_docs") or []
        retry_count = int(state.get("audit_retry_count") or 0)
        prev_audit = state.get("audit_report") or {}
        today = date.today()
        current_year = today.year

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
            "   - **PRINCIPIO DE ATRIBUCIÓN JURISDICCIONAL ESTRICTA**: Responde a la consulta basándote estricta y exclusivamente en los FRAGMENTOS NORMATIVOS recuperados. Cita siempre con precisión la norma, el artículo o anexo y el territorio al que pertenece cada dato. No inventes condiciones de provincias que no consten en los fragmentos ni traslades condiciones de un convenio a otro salvo que la consulta solicite expresamente contrastarlos.\n"
            "   - Revisa `HISTORIAL RECIENTE DE LA CONVERSACIÓN` para mantener todos los parámetros del caso ya aportados por el usuario en la sesión (jornada parcial o completa, puesto, modalidad, etc.) sin olvidarlos.\n"
            "   - **PROHIBIDO REPETIR EXPLICACIONES O TABLAS DE TURNOS ANTERIORES**: Si en un turno previo ya desglosaste una tabla o cálculo y en el `MENSAJE ACTUAL` el usuario plantea una repregunta o nueva duda, **responde directa y exclusivamente a la nueva pregunta del turno actual**, sin volver a volcar las tablas ni repetir el discurso del turno anterior salvo que el usuario pida recalcularlo.\n"
            "2. ADECUACIÓN DE LA ESTRUCTURA A LA INTENCIÓN DEL TURNO Y CIERRE CUANTITATIVO EN TURNO ÚNICO:\n"
            "   - Si la pregunta del turno actual es conceptual, clasificatoria, de permisos, vacaciones, jornada o aclaratoria sobre un requisito del convenio, responde de forma directa, natural y fundamentada en los fragmentos recuperados (integrando cuando proceda Convenio Provincial/Autonómico, VI ALEH y Estatuto de los Trabajadores), **sin forzar tablas salariales que no vienen al caso**.\n"
            "   - **REGLA GENERALISTA DE CIERRE CUANTITATIVO (MONETARIO, TEMPORAL Y PROPORCIONAL)**: Siempre que la normativa en los fragmentos fije una cuantía, duración, límite o regla de proporcionalidad —ya sea en **`€`** (salarios, pluses, horas extra, finiquito), en **`días` / `horas` / `meses`** (vacaciones proporcionales según fecha de inicio/fin o tiempo trabajado, permisos/horas anuales ajustadas a jornada parcial, descansos compensatorios por festivos trabajados) o en **`%`**— y el usuario haya aportado los parámetros necesarios para aplicarla a su caso (en el turno actual o en el historial de la sesión), **es obligatorio ejecutar la operación aritmética completa paso a paso hasta entregar el resultado numérico final** y registrarla en `calculos_derivados` con su fórmula y operandos base. **Está prohibido limitarse a explicar la regla teórica de proporcionalidad o dejar planteados los datos sin calcular el número final**.\n"
            "   - Si el convenio hace depender una condición de un parámetro del centro de trabajo o del contrato que el usuario no ha precisado (como la categoría o clasificación turística del establecimiento, su nivel o modalidad), expón las opciones previstas en los fragmentos recuperados e indícale dónde puede verificar ese dato en la práctica (contrato de trabajo, encabezado de la nómina o clasificación oficial del establecimiento). **Importante:** Cuando se trate de cuantías económicas o fórmulas salariales/horas extras, esta indicación práctica es solo complementaria; NUNCA sustituye el cálculo numérico obligatorio de todas las opciones en tabla (Reglas 3 y 4).\n"
            "3. REGLAS PARA CONSULTAS SALARIALES / RETRIBUTIVAS (CUANDO EL TURNO ACTUAL PIDA SUELDOS O CÁLCULOS):\n"
            f"   - **Año en curso por defecto ({current_year}), tramos plurianuales e incrementos pactados**: Si el usuario NO especifica un año concreto, **asume siempre que consulta por el año actual ({current_year})**.\n"
            f"     * Si los fragmentos ya incluyen tablas salariales publicadas específicamente para {current_year} (o tramos plurianuales sucesivos del convenio en vigor, p. ej. 2025-2026, 2026-2027 y 2027-2028), utiliza directamente la tabla de {current_year} (y, en convenios plurianuales con varios tramos en el mismo fragmento, muestra una fila por tramo temporal vigente para el nivel y categoría aplicables), **sin volver a sumar porcentajes que ya estén incorporados en dicha tabla**.\n"
            f"     * **ACTUALIZACIÓN OBLIGATORIA CUANDO NO HAY TABLA PUBLICADA DE {current_year} PERO SÍ INCREMENTO PACTADO**: Si la tabla salarial más reciente publicada en los fragmentos corresponde a un año anterior (ej. {current_year - 1}) **y** otro fragmento recuperado establece un **incremento porcentual fijo y obligatorio aplicable a {current_year}** (por ejemplo, una subida del +4,00% desde el 1 de enero de {current_year} sobre las tablas vigentes a 31 de diciembre de {current_year - 1}), **DEBES indicar siempre la cifra literal de la tabla publicada de origen (citando su año y acuerdo) Y calcular obligatoriamente el importe actualizado para {current_year}** (`Base_Tabla * (1 + Porcentaje/100)`, y su equivalente prorrateado si procede), registrando el cálculo en `calculos_derivados`.\n"
            f"     * Si la tabla más reciente es anterior a {current_year} y NO consta ningún acuerdo o porcentaje de incremento para {current_year} en los fragmentos recuperados, informa claramente del año de la última tabla publicada disponible sin inventar subidas.\n"
            "   - **Distinción entre Grupo Profesional (VI ALEH) y Nivel Salarial del Convenio**: No confundas ni equipares nunca el número de Grupo Profesional estatal del VI ALEH (ej. Grupo Profesional II) con el Nivel Salarial o Retributivo del convenio provincial/autonómico (ej. Nivel IV en Baleares). Determina siempre el Nivel Salarial y la Categoría del establecimiento buscando literalmente el puesto y el tipo de centro dentro de los Anexos de clasificación del convenio recuperados.\n"
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
            "   - Siempre que el usuario pregunte por un **importe neto (sueldo neto, hora extra neta, nómina neta) o descuento de impuestos (IRPF / Seguridad Social)**, **es obligatorio advertir de forma destacada que cualquier cálculo neto es ESTRICTAMENTE ORIENTATIVO y aproximado**, ya que el sistema no realiza liquidaciones fiscales de IRPF (el porcentaje de retención depende de las circunstancias personales y familiares del Modelo 145, modalidad contractual y rendimientos anuales totales), **sin inventar ni estimar porcentajes específicos de memoria si no constan en los fragmentos recuperados**. Si muestras una simulación o ejemplo en neto, etiquétalo siempre expresamente como **«Neto Orientativo (aprox.)»**.\n"
            "6. FORMATO MATEMÁTICO LIMPIO Y LEGIBLE EN PANTALLA:\n"
            "   - Para operaciones aritméticas directas y proporciones en línea (ej. `30h / 40h = 75%`, `80h × 0,75 = 60 horas/año`, `15h × 0,75 = 11,25 horas/mes`), utiliza exclusivamente texto plano o negritas limpias. **Está prohibido usar comandos LaTeX complejos (`\\frac`, `\\text`, `\\mathbf`) dentro de delimitadores en línea `$...$`** para evitar errores de renderizado en la interfaz web. Reserva los bloques `$$...$$` en línea aparte únicamente si citas una fórmula oficial extensa del convenio."
        )

        retry_correction_block = ""
        if retry_count > 0 and prev_audit and not prev_audit.get("passed", True):
            retry_correction_block = (
                f"\n\n⚠️ CORRECCIÓN OBLIGATORIA TRAS AUDITORÍA DETERMINISTA (REINTENTO #{retry_count}):\n"
                "Tu respuesta anterior no superó el guardrail de verificación numérica/citación.\n"
                f"- Cifras no verificadas detectadas: {prev_audit.get('unverified_numbers') or []}\n"
                f"- Advertencias del auditor: {prev_audit.get('warnings') or []}\n"
                "INSTRUCCIÓN DE REPARACIÓN: Elimina o corrige inmediatamente toda cifra no respaldada literalmente por los fragmentos recuperados o por un cálculo explícito en `calculos_derivados` cuyos operandos figuren en dichos fragmentos, y asegura citas explícitas a artículo/anexo."
            )

        prompt = (
            f"FECHA ACTUAL DEL SISTEMA: {today.strftime('%d/%m/%Y')} (AÑO EN CURSO: {current_year})\n"
            f"HISTORIAL RECIENTE DE LA CONVERSACIÓN (TURNOS PREVIOS DE ESTA SESIÓN): {json.dumps(history, ensure_ascii=False)}\n"
            f"MENSAJE ACTUAL DEL USUARIO: {raw_query}\n"
            f"CONSULTA AUTOCONTENIDA RESUELTA (OBJETIVO DE ESTE TURNO): {effective_query}\n"
            f"ÁMBITO TERRITORIAL RECUPERADO: {provincia or 'Estatal'}\n"
            f"TRADUCCIÓN TÉCNICA VI ALEH: Categoría={cat_aleh or 'N/A'} | Área Funcional={area_aleh or 'N/A'}\n\n"
            f"FRAGMENTOS NORMATIVOS RECUPERADOS DE QDRANT + COHERE RERANK:\n"
            f"{full_context_str}"
            f"{retry_correction_block}"
        )

        step_usage: list[dict[str, Any]] = []
        step_name = f"synthesize_retry_{retry_count}" if retry_count > 0 else "synthesize_response"
        synthesis, model_used = _call_structured_llm(
            prompt=prompt,
            schema=LegalSynthesisSchema,
            preferred_model=GEMINI_REASONING_MODEL,
            system_instruction=system_instruction,
            temperature=0.0,
            usage_out=step_usage,
            step_name=step_name,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        timings = dict(state.get("node_timings_ms") or {})
        timings["synthesize_response"] = round(float(timings.get("synthesize_response", 0.0)) + elapsed_ms, 1)
        models_used = dict(state.get("models_used") or {})
        models_used["synthesize_response"] = model_used

        return {
            "jerarquia_normativa_aplicada": synthesis.jerarquia_normativa_aplicada,
            "citas_normativas": [c.model_dump() for c in synthesis.citas_normativas],
            "cifras_y_datos_criticos": synthesis.cifras_y_datos_criticos,
            "calculos_derivados": [c.model_dump() for c in synthesis.calculos_derivados],
            "final_response": synthesis.respuesta_final_markdown,
            "audit_needs_retry": False,
            "node_timings_ms": timings,
            "models_used": models_used,
            "usage_events": step_usage,
        }

    def audit_output_node(self, state: LegalGraphState) -> LegalGraphState:
        """Nodo 5 (CPU Determinista): audita cifras/citas y dispara hasta 1 reintento de síntesis si falla."""
        t0 = time.perf_counter()
        raw_query = state.get("query", "")
        effective_query = state.get("effective_query") or raw_query
        provincia = state.get("territorio_normalizado")
        docs: list[RetrievedDocument] = state.get("retrieved_docs") or []

        citas_objs = [LegalCitation.model_validate(c) for c in (state.get("citas_normativas") or [])]
        calcs_objs = [DerivedCalculation.model_validate(c) for c in (state.get("calculos_derivados") or [])]
        final_md = state.get("final_response", "")

        synthesis_obj = LegalSynthesisSchema(
            jerarquia_normativa_aplicada=state.get("jerarquia_normativa_aplicada", ""),
            citas_normativas=citas_objs,
            cifras_y_datos_criticos=list(state.get("cifras_y_datos_criticos") or []),
            calculos_derivados=calcs_objs,
            respuesta_final_markdown=final_md,
        )

        audit = self._audit_numerical_and_citation_fidelity(
            synthesis=synthesis_obj,
            docs=docs,
            user_query=f"{raw_query} {effective_query}",
            expected_territorio=provincia,
        )

        current_retry = int(state.get("audit_retry_count") or 0)
        if audit.passed:
            audit_needs_retry = False
            next_retry = current_retry
        elif current_retry < 1:
            audit_needs_retry = True
            next_retry = current_retry + 1
            logger.warning(
                "Auditoría determinista fallida (warnings=%s). Activando autocorrección (reintento %d/1)...",
                audit.warnings,
                next_retry,
            )
        else:
            audit_needs_retry = False
            next_retry = current_retry
            if audit.warnings:
                final_md = (
                    f"{final_md}\n\n"
                    "> ⚠️ **Aviso de verificación automática:** Algunas cifras o citas requieren comprobación directa en el texto oficial citado "
                    f"({'; '.join(audit.warnings)})."
                )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        timings = dict(state.get("node_timings_ms") or {})
        timings["audit_output"] = round(float(timings.get("audit_output", 0.0)) + elapsed_ms, 2)
        models_used = dict(state.get("models_used") or {})
        models_used["audit_output"] = "deterministic-cpu-auditor"

        return {
            "audit_report": audit.model_dump(),
            "audit_retry_count": next_retry,
            "audit_needs_retry": audit_needs_retry,
            "final_response": final_md,
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
            for m in re.finditer(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:[.,]\d+)?\b", combined_corpus)
        }
        norm_query_numbers = {
            cls._normalize_number_token(m.group(0))
            for m in re.finditer(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+(?:[.,]\d+)?\b", user_query)
        }
        # Constantes universales de calendario/jornada (12 meses, 11 meses laborables, 14/15 pagas, 30 días, 40h semanales, base 1 y 100%)
        standard_constants = {"1", "11", "12", "14", "15", "30", "40", "100"}

        corpus_inline_products: set[float] = set()
        for pm in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*[x×\*]\s*(\d+(?:[.,]\d+)?)\b", combined_corpus, flags=re.IGNORECASE):
            try:
                p_a = float(cls._normalize_number_token(pm.group(1)))
                p_b = float(cls._normalize_number_token(pm.group(2)))
                if p_a > 0 and p_b > 0:
                    prod_val = round(p_a * p_b, 4)
                    corpus_inline_products.add(prod_val)
                    norm_corpus_numbers.add(
                        str(int(round(prod_val)))
                        if abs(prod_val - round(prod_val)) < 1e-6
                        else f"{prod_val:g}"
                    )
            except ValueError:
                pass

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
        revision_multipliers = {1.0} | {m for m in surcharge_multipliers if 1.0 < m <= 1.25}
        ratios_jornada = [1.0] + [h_val / 40.0 for h_val in query_floats]
        annual_hours_in_corpus = [v for v in corpus_floats if 1500.0 <= v <= 2000.0]
        monthly_divisors_in_corpus = sorted(
            {v for v in (set(corpus_floats) | corpus_inline_products) if 100.0 <= v <= 250.0}
        )
        for c_val in corpus_floats:
            if 0.5 <= c_val < 500.0:
                for rev_mult in revision_multipliers:
                    auto_derived_floats.add(round(c_val * rev_mult, 2))
                for mult in (11.0, 12.0, 20.0, 21.0, 22.0, 30.0):
                    auto_derived_floats.add(round(c_val * mult, 2))
            if c_val >= 500.0:
                for r_jor in ratios_jornada:
                    scaled_val = c_val * r_jor
                    for rev_mult in revision_multipliers:
                        rev_val = scaled_val * rev_mult
                        auto_derived_floats.add(round(rev_val, 2))
                        for pagas in (12.0, 14.0, 15.0):
                            auto_derived_floats.add(round(rev_val / pagas, 2))
                            auto_derived_floats.add(round(rev_val * pagas, 2))
                            auto_derived_floats.add(round(rev_val * (pagas / 12.0), 2))
                for div_val in annual_hours_in_corpus:
                    for pagas in (12.0, 14.0, 15.0):
                        base_hour = (c_val * pagas) / div_val
                        auto_derived_floats.add(round(base_hour, 2))
                        for rec in surcharge_multipliers:
                            auto_derived_floats.add(round(base_hour * rec, 2))
                for m_div in monthly_divisors_in_corpus:
                    base_month_hour = c_val / m_div
                    auto_derived_floats.add(round(base_month_hour, 2))
                    for rec in surcharge_multipliers:
                        auto_derived_floats.add(round(base_month_hour * rec, 2))

        # Sumas multiconcepto generales (Salario Base o Prorrateado + subconjuntos de complementos monetarios del corpus)
        import itertools
        euro_corpus_floats: set[float] = set()
        for em in re.finditer(r"\b(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)\s*(?:€|euros?)", combined_corpus, flags=re.IGNORECASE):
            try:
                euro_corpus_floats.add(float(cls._normalize_number_token(em.group(1))))
            except ValueError:
                pass
        candidate_bases = [v for v in auto_derived_floats if 400.0 <= v <= 4500.0] + [
            v for v in corpus_floats if 400.0 <= v <= 4500.0
        ]
        monetary_pluses = (
            {v for v in euro_corpus_floats if 1.0 <= v <= 500.0}
            | {v for v in corpus_floats if 1.0 <= v <= 500.0 and abs(v - round(v)) > 1e-6}
        )
        candidate_pluses = sorted(monetary_pluses)[:15]
        for base_val in candidate_bases:
            for k_terms in range(1, min(len(candidate_pluses) + 1, 5)):
                for comb in itertools.combinations(candidate_pluses, k_terms):
                    auto_derived_floats.add(round(base_val + sum(comb), 2))

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
            re.search(
                r"(?i)\b(?:art[íi]culos?|artigos?|articles?|arts?\.(?=\s|\d|$)|arts?\s+\d|anexos?\b|tablas?\b|t[áa]boas?\b|disposici[óo]n(?:es|s)?\b)",
                synthesis.respuesta_final_markdown,
            )
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
            raw_exp_tokens = [
                _canonicalize_llm_province(t)
                for t in re.split(r"[,/|]|\s+y\s+", expected_territorio.replace("_", " "))
                if t.strip()
            ]
            valid_terrs = {t for t in raw_exp_tokens if t}
            for t in list(valid_terrs):
                for comm, provs in AUTONOMOUS_COMMUNITY_EXPANSION.items():
                    if t in provs or t == comm:
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
