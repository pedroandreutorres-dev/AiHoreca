"""Motor de tarifas oficiales en vivo (Google Gemini Paid Tier + Cohere Rerank) y cálculo de Unit Economics."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Optional
import urllib.request

from pydantic import BaseModel, Field

from src.config import (
    COHERE_RERANK_USD_PER_1K_UNITS,
    GEMINI_PRICING_URL,
    PAID_TIER_FLASH_SYNTHESIS_MODEL,
    PAID_TIER_LITE_ROUTER_MODEL,
    PAID_TIER_LITE_SYNTHESIS_MODEL,
    PRICING_SNAPSHOT_PATH,
    USD_TO_EUR_RATE,
)
from src.graph.state import TurnUsageSummary


class ModelRate(BaseModel):
    """Tarifa unitaria en USD por 1M de tokens (Paid Tier Standard)."""

    model_id: str
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cached_input_per_1m: float = 0.0
    note: str = ""


class PricingCatalog(BaseModel):
    """Catálogo completo de tarifas vigentes consultadas para la auditoría."""

    checked_at: str
    as_of_date: str
    source: str  # live_official | cached_snapshot | fallback_builtin
    pricing_url: str = GEMINI_PRICING_URL
    usd_to_eur: float = USD_TO_EUR_RATE
    cohere_usd_per_1k_units: float = COHERE_RERANK_USD_PER_1K_UNITS
    models: dict[str, ModelRate] = Field(default_factory=dict)


class TurnCostBreakdown(BaseModel):
    """Coste estimado en API de pago para un turno (real vs. stacks recomendados)."""

    coste_gemini_real_usd: float = 0.0
    coste_embeddings_usd: float = 0.0
    coste_cohere_usd: float = 0.0
    coste_total_real_usd: float = 0.0
    coste_total_real_eur: float = 0.0
    coste_gemini_tier_lite_usd: float = 0.0
    coste_total_tier_lite_usd: float = 0.0
    coste_total_tier_lite_eur: float = 0.0
    coste_gemini_tier_flash_usd: float = 0.0
    coste_total_tier_flash_usd: float = 0.0
    coste_total_tier_flash_eur: float = 0.0
    embedding_tokens_est: int = 0


# Tarifas oficiales de respaldo verificadas el 27/11/2026 en https://ai.google.dev/gemini-api/docs/pricing.md.txt
_BUILTIN_FALLBACK_RATES: dict[str, tuple[float, float, float, str]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01, "Standard fallback 2026-11-27"),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025, "Standard fallback 2026-11-27"),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03, "Standard fallback 2026-11-27"),
    "gemini-2.5-flash": (0.30, 2.50, 0.03, "Standard fallback 2026-11-27"),
    "gemini-3-flash-preview": (0.50, 3.00, 0.05, "Standard fallback 2026-11-27"),
    "gemini-3.6-flash": (0.75, 3.75, 0.075, "Standard fallback 2026-11-27 (step-up 2027-01-01)"),
    "gemini-3.8-flash": (0.75, 3.75, 0.075, "Standard fallback 2026-11-27 (step-up 2027-01-01)"),
    "gemini-4-hydrogen": (1.50, 7.50, 0.075, "Standard fallback 2026-11-27"),
    "gemini-2.5-pro": (1.25, 10.00, 0.125, "Standard fallback 2026-11-27"),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20, "Standard fallback 2026-11-27"),
    "gemini-embedding-001": (0.15, 0.0, 0.0, "Standard fallback 2026-11-27"),
    "gemini-embedding-2": (0.20, 0.0, 0.0, "Standard fallback 2026-11-27"),
}

_MONTHS_EN: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_DATED_PRICE_RE = re.compile(
    r"\$([0-9]+(?:\.[0-9]+)?)\s+(through|starting)\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
    re.IGNORECASE,
)
_USD_RE = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)")


def _extract_effective_usd(cell_text: str, as_of: date) -> Optional[float]:
    """Extrae el precio USD aplicable para la fecha `as_of`, ignorando audio/almacenamiento si hay texto."""
    clean = re.sub(r"\{%.*?%\}", " ", cell_text or "", flags=re.S)
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    # Descartar líneas de almacenamiento por hora en contexto caching y audio si existen líneas de texto
    filtered = [ln for ln in lines if "storage price" not in ln.lower() and "/ 1,000,000 tokens per hour" not in ln.lower()]
    text_lines = [ln for ln in filtered if not re.match(r"^\$[0-9.]+\s*\(audio\)", ln, flags=re.I)]
    candidate_lines = text_lines or filtered or lines

    dated_matches: list[tuple[float, str, date]] = []
    for ln in candidate_lines:
        for price_s, kind, month_s, day_s, year_s in _DATED_PRICE_RE.findall(ln):
            month_num = _MONTHS_EN.get(month_s.lower())
            if not month_num:
                continue
            try:
                bound_dt = date(int(year_s), month_num, int(day_s))
            except ValueError:
                continue
            dated_matches.append((float(price_s), kind.lower(), bound_dt))

    if dated_matches:
        starting_matches = sorted(
            ((dt, p) for p, kind, dt in dated_matches if kind == "starting" and as_of >= dt),
            key=lambda x: x[0],
        )
        if starting_matches:
            return starting_matches[-1][1]
        through_matches = sorted(
            ((dt, p) for p, kind, dt in dated_matches if kind == "through" and as_of <= dt),
            key=lambda x: x[0],
        )
        if through_matches:
            return through_matches[0][1]

    for ln in candidate_lines:
        if "free of charge" in ln.lower() or "not available" in ln.lower():
            continue
        m_usd = _USD_RE.search(ln)
        if m_usd:
            return float(m_usd.group(1))
    return None


def parse_official_gemini_pricing_markdown(
    markdown_text: str,
    as_of: Optional[date] = None,
) -> dict[str, ModelRate]:
    """Parsea la página Markdoc oficial `pricing.md.txt` de Google AI Studio / Gemini API."""
    eff_date = as_of or date.today()
    if eff_date.year == 2026:
        # Asegura usar tarifa vigente si el reloj local coincide; ya es `date.today()`.
        pass
    sections = re.split(r"(?=^## )", markdown_text or "", flags=re.M)
    parsed: dict[str, ModelRate] = {}

    for sec in sections:
        m_head = re.match(r"^##\s+([^\n{]+)", sec)
        if not m_head or "gemini" not in m_head.group(1).lower():
            continue
        title = m_head.group(1).strip()
        if any(skip in title.lower() for skip in ("image", "tts", "audio", "transcribe", "robotics", "computer", "live")):
            continue

        header_slice = sec[:800]
        model_ids = re.findall(r"`(gemini-[a-z0-9.\-]+)`", header_slice, flags=re.I)
        for m_id in re.findall(r'id="(gemini-[a-z0-9.\-]+)"', sec[:220], flags=re.I):
            if m_id not in model_ids:
                model_ids.append(m_id)
        if not model_ids:
            continue

        m_std = re.search(r'{%\s*tab\s+title="Standard"\s*%}(.*?){%\s*/tab\s*%}', sec, flags=re.S)
        std_block = m_std.group(1) if m_std else sec
        rows = re.split(r"\n\s*---\s*\n", std_block)

        in_price: Optional[float] = None
        out_price: Optional[float] = None
        cache_price: Optional[float] = None

        for row in rows:
            cells = [
                c.strip()
                for c in re.findall(r"(?m)^\s*\*\s+(.*?)(?=\n\s*\*\s+|\Z)", row, flags=re.S)
            ]
            if len(cells) < 3:
                continue
            feat = re.sub(r"\{%.*?%\}", "", cells[0].splitlines()[0]).strip().lower()
            paid_cell = cells[2]
            if "text input price" in feat or feat == "input price" or feat.startswith("input price ("):
                val = _extract_effective_usd(paid_cell, eff_date)
                if val is not None and in_price is None:
                    in_price = val
            elif feat.startswith("output price"):
                val = _extract_effective_usd(paid_cell, eff_date)
                if val is not None and out_price is None:
                    out_price = val
            elif feat.startswith("context caching price"):
                val = _extract_effective_usd(paid_cell, eff_date)
                if val is not None and cache_price is None:
                    cache_price = val

        if in_price is None and out_price is None:
            continue

        note_str = f"Parsed live ({title}, as_of={eff_date.isoformat()})"
        for raw_id in model_ids:
            mid = raw_id.lower().strip()
            # No pisar modelos ya resueltos con más precisión si hay enlaces secundarios
            if mid not in parsed:
                parsed[mid] = ModelRate(
                    model_id=mid,
                    input_per_1m=round(in_price or 0.0, 6),
                    output_per_1m=round(out_price or 0.0, 6),
                    cached_input_per_1m=round(
                        cache_price if cache_price is not None else (in_price or 0.0) * 0.1,
                        6,
                    ),
                    note=note_str,
                )

    return parsed


def _builtin_fallback_models() -> dict[str, ModelRate]:
    return {
        mid: ModelRate(
            model_id=mid,
            input_per_1m=vals[0],
            output_per_1m=vals[1],
            cached_input_per_1m=vals[2],
            note=vals[3],
        )
        for mid, vals in _BUILTIN_FALLBACK_RATES.items()
    }


def load_pricing_catalog(
    fetch_live: bool = True,
    snapshot_path: Path = PRICING_SNAPSHOT_PATH,
    pricing_url: str = GEMINI_PRICING_URL,
    timeout_sec: float = 8.0,
    as_of: Optional[date] = None,
) -> PricingCatalog:
    """Obtiene tarifas oficiales Paid Tier en vivo, guardando snapshot JSON con fallback offline."""
    eff_date = as_of or date.today()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if fetch_live and pricing_url:
        try:
            req = urllib.request.Request(pricing_url, headers={"User-Agent": "AiHoreca-AuditRunner/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                md_text = resp.read().decode("utf-8", errors="ignore")
            parsed_models = parse_official_gemini_pricing_markdown(md_text, as_of=eff_date)
            # Completar modelos base conocidos si faltara alguno secundario
            if len(parsed_models) >= 5 and "gemini-3.5-flash-lite" in parsed_models:
                for k, v in _builtin_fallback_models().items():
                    parsed_models.setdefault(k, v)
                catalog = PricingCatalog(
                    checked_at=now_iso,
                    as_of_date=eff_date.isoformat(),
                    source="live_official",
                    pricing_url=pricing_url,
                    usd_to_eur=USD_TO_EUR_RATE,
                    cohere_usd_per_1k_units=COHERE_RERANK_USD_PER_1K_UNITS,
                    models=parsed_models,
                )
                try:
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
                except Exception:
                    pass
                return catalog
        except Exception:
            pass

    if snapshot_path.exists():
        try:
            snap = PricingCatalog.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
            if snap.models:
                snap.source = "cached_snapshot"
                snap.usd_to_eur = USD_TO_EUR_RATE
                snap.cohere_usd_per_1k_units = COHERE_RERANK_USD_PER_1K_UNITS
                return snap
        except Exception:
            pass

    return PricingCatalog(
        checked_at=now_iso,
        as_of_date=eff_date.isoformat(),
        source="fallback_builtin",
        pricing_url=pricing_url,
        usd_to_eur=USD_TO_EUR_RATE,
        cohere_usd_per_1k_units=COHERE_RERANK_USD_PER_1K_UNITS,
        models=_builtin_fallback_models(),
    )


def resolve_model_rate(catalog: PricingCatalog, model_name: Optional[str]) -> Optional[ModelRate]:
    """Resuelve la tarifa de un modelo Gemini contemplando alias `-latest` / `-preview`."""
    raw = (model_name or "").strip().lower()
    if not raw or raw in ("n/d", "-", "cpu_deterministic_shortcut", "targeted-qdrant-hop"):
        return None
    if raw.startswith("models/"):
        raw = raw[len("models/") :]

    aliases = {
        "gemini-flash-lite-latest": "gemini-2.5-flash-lite",
        "gemini-flash-latest": "gemini-2.5-flash",
        "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    }
    candidates = [raw]
    if raw in aliases:
        candidates.append(aliases[raw])
    stripped = re.sub(r"-(?:preview|latest)(?:-\d{2}-\d{4})?$", "", raw)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    for cand in candidates:
        if cand in catalog.models:
            return catalog.models[cand]
    # Coincidencia parcial segura para variantes menores
    for cand in candidates:
        for known_id, rate in catalog.models.items():
            if cand.startswith(known_id) or known_id.startswith(cand):
                return rate
    # Respaldo conservador si llega un modelo nuevo no catalogado aún
    return catalog.models.get(PAID_TIER_LITE_SYNTHESIS_MODEL) or next(iter(catalog.models.values()), None)


def _compute_llm_token_cost_usd(
    input_tokens: int,
    output_tokens: int,
    thoughts_tokens: int,
    cached_tokens: int,
    rate: Optional[ModelRate],
) -> float:
    if rate is None:
        return 0.0
    cached_in = min(max(0, cached_tokens), max(0, input_tokens))
    uncached_in = max(0, input_tokens - cached_in)
    billable_out = max(0, output_tokens) + max(0, thoughts_tokens)
    usd = (
        uncached_in * rate.input_per_1m
        + cached_in * rate.cached_input_per_1m
        + billable_out * rate.output_per_1m
    ) / 1_000_000.0
    return usd


def compute_turn_cost(usage: TurnUsageSummary, catalog: PricingCatalog) -> TurnCostBreakdown:
    """Calcula el coste real y proyectado (Tier Lite y Tier Flash) de un turno."""
    real_gemini_usd = 0.0
    lite_gemini_usd = 0.0
    flash_gemini_usd = 0.0

    lite_router_rate = resolve_model_rate(catalog, PAID_TIER_LITE_ROUTER_MODEL)
    lite_synth_rate = resolve_model_rate(catalog, PAID_TIER_LITE_SYNTHESIS_MODEL)
    flash_synth_rate = resolve_model_rate(catalog, PAID_TIER_FLASH_SYNTHESIS_MODEL)

    llm_steps = [s for s in usage.steps if s.provider == "gemini_llm"]
    if llm_steps:
        for st in llm_steps:
            if not (st.input_tokens or st.output_tokens or st.thoughts_tokens):
                continue
            step_rate = resolve_model_rate(catalog, st.model)
            real_gemini_usd += _compute_llm_token_cost_usd(
                st.input_tokens,
                st.output_tokens,
                st.thoughts_tokens,
                st.cached_tokens,
                step_rate,
            )
            is_synth = "synth" in (st.step or "").lower()
            lite_gemini_usd += _compute_llm_token_cost_usd(
                st.input_tokens,
                st.output_tokens,
                st.thoughts_tokens,
                st.cached_tokens,
                lite_synth_rate if is_synth else lite_router_rate,
            )
            flash_gemini_usd += _compute_llm_token_cost_usd(
                st.input_tokens,
                st.output_tokens,
                st.thoughts_tokens,
                st.cached_tokens,
                flash_synth_rate if is_synth else lite_router_rate,
            )
    elif usage.total_tokens > 0:
        real_gemini_usd = _compute_llm_token_cost_usd(
            usage.input_tokens, usage.output_tokens, usage.thoughts_tokens, usage.cached_tokens, lite_synth_rate
        )
        lite_gemini_usd = real_gemini_usd
        flash_gemini_usd = _compute_llm_token_cost_usd(
            usage.input_tokens, usage.output_tokens, usage.thoughts_tokens, usage.cached_tokens, flash_synth_rate
        )

    # Embeddings: solo los caracteres de llamadas NO cacheadas (`api_calls > 0` / `cache_hit=False`)
    emb_chars_uncached = 0
    emb_steps = [s for s in usage.steps if s.provider == "gemini_embedding"]
    if emb_steps:
        emb_chars_uncached = sum(
            s.chars for s in emb_steps if not s.cache_hit and s.status != "cache_hit" and s.api_calls > 0
        )
    elif usage.embedding_calls > 0:
        emb_chars_uncached = usage.embedding_chars
    emb_tokens_est = math.ceil(emb_chars_uncached / 4) if emb_chars_uncached > 0 else 0
    emb_rate = resolve_model_rate(catalog, "gemini-embedding-001")
    emb_usd = (emb_tokens_est * (emb_rate.input_per_1m if emb_rate else 0.15)) / 1_000_000.0

    # Cohere Rerank
    cohere_usd = max(0, int(usage.cohere_search_units or 0)) * (catalog.cohere_usd_per_1k_units / 1000.0)

    total_real_usd = real_gemini_usd + emb_usd + cohere_usd
    total_lite_usd = lite_gemini_usd + emb_usd + cohere_usd
    total_flash_usd = flash_gemini_usd + emb_usd + cohere_usd

    eur_rate = catalog.usd_to_eur
    return TurnCostBreakdown(
        coste_gemini_real_usd=round(real_gemini_usd, 6),
        coste_embeddings_usd=round(emb_usd, 6),
        coste_cohere_usd=round(cohere_usd, 6),
        coste_total_real_usd=round(total_real_usd, 6),
        coste_total_real_eur=round(total_real_usd * eur_rate, 6),
        coste_gemini_tier_lite_usd=round(lite_gemini_usd, 6),
        coste_total_tier_lite_usd=round(total_lite_usd, 6),
        coste_total_tier_lite_eur=round(total_lite_usd * eur_rate, 6),
        coste_gemini_tier_flash_usd=round(flash_gemini_usd, 6),
        coste_total_tier_flash_usd=round(total_flash_usd, 6),
        coste_total_tier_flash_eur=round(total_flash_usd * eur_rate, 6),
        embedding_tokens_est=emb_tokens_est,
    )
