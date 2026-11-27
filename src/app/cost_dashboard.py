"""Generador de dashboard HTML autocontenido (`data/logs/dashboard_costes.html`) a partir de los CSVs permanentes y tarifas vivas."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from src.app.pricing import PricingCatalog, load_pricing_catalog, resolve_model_rate
from src.config import (
    COST_DASHBOARD_PATH,
    EMBEDDING_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_LITE_MODEL,
    GEMINI_REASONING_MODEL,
    PAID_TIER_FLASH_SYNTHESIS_MODEL,
    PAID_TIER_LITE_ROUTER_MODEL,
    PAID_TIER_LITE_SYNTHESIS_MODEL,
    SESSION_COSTS_HISTORY_PATH,
    TURN_COSTS_HISTORY_PATH,
)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _to_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val)) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _build_dashboard_payload(
    sessions_csv_path: Path,
    turns_csv_path: Path,
    catalog: PricingCatalog,
) -> dict[str, Any]:
    raw_sessions = _read_csv_dicts(sessions_csv_path)
    raw_turns = _read_csv_dicts(turns_csv_path)

    sessions_telem = []
    for r in raw_sessions:
        if _to_int(r.get("turnos_con_telemetria")) <= 0:
            continue
        sessions_telem.append({
            "sid": r.get("session_id", ""),
            "fecha": r.get("fecha_inicio", ""),
            "territorio": r.get("territorio") or "n/d",
            "oficio": r.get("oficio") or "n/d",
            "turnos": _to_int(r.get("total_turnos")),
            "rag": _to_int(r.get("turnos_rag")),
            "sc": _to_int(r.get("turnos_cortocircuito")),
            "ok": _to_int(r.get("votos_ok")),
            "ko": _to_int(r.get("votos_ko")),
            "tok_in": _to_int(r.get("tokens_entrada_total")),
            "tok_out": _to_int(r.get("tokens_salida_total")),
            "tok_th": _to_int(r.get("tokens_thoughts_total")),
            "tok_cache": _to_int(r.get("tokens_cached_total")),
            "tok": _to_int(r.get("tokens_total")),
            "coh": _to_int(r.get("cohere_search_units")),
            "c_gem_real": round(_to_float(r.get("coste_gemini_real_usd")), 6),
            "c_emb": round(_to_float(r.get("coste_embeddings_usd")), 6),
            "c_coh": round(_to_float(r.get("coste_cohere_usd")), 6),
            "c_real": round(_to_float(r.get("coste_total_real_usd")), 6),
            "c_lite": round(_to_float(r.get("coste_total_tier_lite_usd")), 6),
            "c_flash": round(_to_float(r.get("coste_total_tier_flash_usd")), 6),
        })

    turns_telem = []
    for t in raw_turns:
        if _to_int(t.get("has_telemetry")) <= 0:
            continue
        turns_telem.append({
            "id": _to_int(t.get("turn_id")),
            "sid": t.get("session_id", ""),
            "idx": _to_int(t.get("turn_index")),
            "fecha": t.get("fecha", ""),
            "route": t.get("route_decision", ""),
            "terr": t.get("territorio") or "n/d",
            "mr": t.get("modelo_router", ""),
            "ms": t.get("modelo_sintesis", ""),
            "lat": round(_to_float(t.get("latencia_ms")), 1),
            "tin": _to_int(t.get("tokens_entrada")),
            "tout": _to_int(t.get("tokens_salida")),
            "th": _to_int(t.get("tokens_thoughts")),
            "tc": _to_int(t.get("tokens_cached")),
            "tot": _to_int(t.get("tokens_total")),
            "coh": _to_int(t.get("cohere_search_units")),
            "emb": round(_to_float(t.get("coste_embeddings_usd")), 6),
            "c_real": round(_to_float(t.get("coste_total_real_usd")), 6),
            "c_real_llm": round(_to_float(t.get("coste_gemini_real_usd")), 6),
            "c_lite": round(_to_float(t.get("coste_total_tier_lite_usd")), 6),
            "c_flash": round(_to_float(t.get("coste_total_tier_flash_usd")), 6),
        })

    # Tabla de modelos configurados + referencia de pago
    tracked_specs = [
        (GEMINI_LITE_MODEL, "Free Tier · Router (actual)"),
        (GEMINI_REASONING_MODEL, "Free Tier · Síntesis (actual)"),
        *[(m, "Free Tier · Fallback") for m in GEMINI_FALLBACK_MODELS if m not in (GEMINI_LITE_MODEL, GEMINI_REASONING_MODEL)],
        (PAID_TIER_FLASH_SYNTHESIS_MODEL, "Paid Tier · Síntesis Flash"),
        ("gemini-4-hydrogen", "Paid Tier · Flash 2027+"),
        ("gemini-3.1-pro-preview", "Paid Tier · Pro (rescate A1)"),
        (EMBEDDING_MODEL, "Embeddings RAG"),
    ]
    seen_m: set[str] = set()
    tracked_models = []
    for mid, role in tracked_specs:
        key = (mid or "").strip().lower()
        if not key or key in seen_m:
            continue
        seen_m.add(key)
        rate = resolve_model_rate(catalog, key)
        tracked_models.append({
            "model": mid,
            "role": role,
            "in_1m": rate.input_per_1m if rate else 0.0,
            "cache_1m": rate.cached_input_per_1m if rate else 0.0,
            "out_1m": rate.output_per_1m if rate else 0.0,
            "deprecated": bool(rate.deprecated) if rate else False,
            "upcoming": rate.upcoming_change if rate else "",
        })

    r_lr = resolve_model_rate(catalog, PAID_TIER_LITE_ROUTER_MODEL)
    r_ls = resolve_model_rate(catalog, PAID_TIER_LITE_SYNTHESIS_MODEL)
    r_fs = resolve_model_rate(catalog, PAID_TIER_FLASH_SYNTHESIS_MODEL)

    return {
        "generated_at": catalog.checked_at,
        "as_of_date": catalog.as_of_date,
        "pricing_source": catalog.source,
        "usd_to_eur": catalog.usd_to_eur,
        "cohere_per_1k": catalog.cohere_usd_per_1k_units,
        "total_sessions_all": len(raw_sessions),
        "total_turns_all": len(raw_turns),
        "sessions": sessions_telem,
        "turns": turns_telem,
        "radar": catalog.radar.model_dump(),
        "tracked_models": tracked_models,
        "rates": {
            "lite_router_in": r_lr.input_per_1m if r_lr else 0.25,
            "lite_router_out": r_lr.output_per_1m if r_lr else 1.50,
            "lite_synth_in": r_ls.input_per_1m if r_ls else 0.30,
            "lite_synth_out": r_ls.output_per_1m if r_ls else 2.50,
            "flash_synth_in": r_fs.input_per_1m if r_fs else 0.75,
            "flash_synth_out": r_fs.output_per_1m if r_fs else 3.75,
        },
    }


def generate_cost_dashboard_html(
    sessions_csv_path: Path = SESSION_COSTS_HISTORY_PATH,
    turns_csv_path: Path = TURN_COSTS_HISTORY_PATH,
    pricing_catalog: Optional[PricingCatalog] = None,
    output_path: Path = COST_DASHBOARD_PATH,
) -> dict[str, Any]:
    """Genera el dashboard HTML autocontenido (0 CDNs, abrible offline) con métricas y radar de tarifas."""
    catalog = pricing_catalog or load_pricing_catalog(fetch_live=False)
    payload = _build_dashboard_payload(sessions_csv_path, turns_csv_path, catalog)
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AiHoreca · Dashboard de Costes, Modelos y Unit Economics</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-2: #162032;
      --border: #334155;
      --fg: #f8fafc;
      --muted: #94a3b8;
      --primary: #38bdf8;
      --primary-fg: #0f172a;
      --ok: #22c55e;
      --warn: #f59e0b;
      --err: #ef4444;
      --c1: #38bdf8;
      --c2: #818cf8;
      --c3: #34d399;
      --c4: #fbbf24;
      --c5: #f472b6;
    }}
    body.light {{
      --bg: #f8fafc;
      --card: #ffffff;
      --card-2: #f1f5f9;
      --border: #cbd5e1;
      --fg: #0f172a;
      --muted: #475569;
      --primary: #0284c7;
      --primary-fg: #ffffff;
      --ok: #16a34a;
      --warn: #d97706;
      --err: #dc2626;
      --c1: #0284c7;
      --c2: #6366f1;
      --c3: #059669;
      --c4: #d97706;
      --c5: #db2777;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      background: var(--bg);
      color: var(--fg);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }}
    .row {{ display: grid; gap: 16px; }}
    .g4 {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .g5 {{ grid-template-columns: 3fr 2fr; }}
    .g3 {{ grid-template-columns: 1fr 2fr; }}
    @media (max-width: 900px) {{ .g5, .g3 {{ grid-template-columns: 1fr; }} }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }}
    .subcard {{
      background: var(--card-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-end;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 4px 0 0; font-size: 22px; font-weight: 700; }}
    h2 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .kpi {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .btn-group {{
      display: inline-flex;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 3px;
      gap: 2px;
    }}
    button, select {{
      background: transparent;
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    .btn-group button {{ border: none; color: var(--muted); font-weight: 600; }}
    .btn-group button.active {{ background: var(--primary); color: var(--primary-fg); }}
    select {{ background: var(--card-2); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 7px 6px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    td.r, th.r {{ text-align: right; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      border: 1px solid var(--border);
    }}
    .badge.ok {{ color: var(--ok); border-color: var(--ok); }}
    .badge.warn {{ color: var(--warn); border-color: var(--warn); }}
    .badge.err {{ color: var(--err); border-color: var(--err); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); padding-top: 8px; border-top: 1px solid var(--border); margin-top: 8px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }}
    .bar-track {{ width: 100%; height: 26px; border-radius: 8px; overflow: hidden; display: flex; border: 1px solid var(--border); background: var(--card-2); }}
    .bar-seg {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #fff; transition: width 0.2s; }}
    input[type=range] {{ width: 100%; accent-color: var(--primary); }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div class="muted mono">AIHORECA · AUDIT RUNNER · DATA/LOGS/DASHBOARD_COSTES.HTML</div>
        <h1>Dashboard de Costes, Tarifas Oficiales y Monetización</h1>
        <div id="hdrMeta" class="muted"></div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
        <div class="btn-group">
          <button id="btnUsd" onclick="setCurrency('USD')">USD ($)</button>
          <button id="btnEur" onclick="setCurrency('EUR')">EUR (€)</button>
        </div>
        <div class="btn-group">
          <button id="btnPerConv" onclick="setScale(1)">Por conversación</button>
          <button id="btnPer1k" onclick="setScale(1000)">Por 1.000 conv.</button>
        </div>
        <button onclick="toggleTheme()" title="Cambiar tema claro/oscuro">☀️ / 🌙</button>
      </div>
    </header>

    <!-- Radar de Modelos y Precios -->
    <section class="card" style="display:flex;flex-direction:column;gap:12px;">
      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;">
        <div>
          <h2>0. Radar Automático de Modelos y Tarifas Oficiales</h2>
          <div class="muted">Comprobado automáticamente frente a Google AI Studio (<span class="mono">pricing.md.txt</span>) y tu configuración en <span class="mono">src/config.py</span></div>
        </div>
        <span id="radarBadge" class="badge ok">Sin cambios urgentes</span>
      </div>
      <div id="radarAlerts" class="row g4"></div>
    </section>

    <!-- KPI Cards -->
    <section class="row g4">
      <div class="card">
        <div class="muted">Coste medio · Turno RAG completo</div>
        <div id="kpiRag" class="kpi num">—</div>
        <div id="kpiRagSub" class="muted num">—</div>
      </div>
      <div class="card">
        <div class="muted">Coste medio · Cortocircuito (sin BBDD)</div>
        <div id="kpiSc" class="kpi num" style="color:var(--ok);">—</div>
        <div id="kpiScSub" class="muted num">—</div>
      </div>
      <div class="card">
        <div class="muted">Peso Cohere Rerank (en Tier Lite)</div>
        <div id="kpiCohere" class="kpi num" style="color:var(--warn);">—</div>
        <div id="kpiCohereSub" class="muted num">—</div>
      </div>
      <div class="card">
        <div class="muted">Ahorro Prompt Caching / Volumen total</div>
        <div id="kpiTotal" class="kpi num" style="color:var(--c1);">—</div>
        <div id="kpiTotalSub" class="muted num">—</div>
      </div>
    </section>

    <!-- Escenarios + Tabla Modelos -->
    <section class="row g5">
      <div class="card" style="display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;">
          <div>
            <h2>1. Comparativa de Coste por Escenario</h2>
            <div class="muted">Real histórico vs. Tier Lite, Tier Flash 2026/2027 y Tier Pro</div>
          </div>
          <select id="sessionFilter1" onchange="renderAll()"></select>
        </div>
        <div id="scenarioChartWrap"></div>
        <div class="legend">
          <span><span class="dot" style="background:var(--c1)"></span>Gemini LLM</span>
          <span><span class="dot" style="background:var(--c4)"></span>Cohere Rerank</span>
          <span><span class="dot" style="background:var(--c3)"></span>Embeddings</span>
        </div>
      </div>

      <div class="card" style="overflow-x:auto;">
        <h2>Modelos Vigilados y Tarifas (1M tok)</h2>
        <div class="muted" style="margin-bottom:8px;">Cohere Rerank: <span id="cohereRateLbl" class="mono">—</span> · Cambio: <span id="fxLbl" class="mono">—</span></div>
        <table>
          <thead>
            <tr><th>Modelo</th><th>Rol</th><th class="r">In</th><th class="r">Cache</th><th class="r">Out</th><th>Estado</th></tr>
          </thead>
          <tbody id="modelsTableBody" class="num"></tbody>
        </table>
      </div>
    </section>

    <!-- Turnos -->
    <section class="card" style="display:flex;flex-direction:column;gap:10px;">
      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;">
        <div>
          <h2>2. Anatomía Turno a Turno (últimos 30 turnos con telemetría)</h2>
          <div class="muted">Pasa el ratón o haz clic sobre cualquier barra para inspeccionar su coste y consumo exacto</div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <select id="turnMetricSelect" onchange="renderAll()">
            <option value="lite">Escenario: Tier Lite</option>
            <option value="flash26">Escenario: Tier Flash 2026</option>
            <option value="flash27">Escenario: Tier Flash 2027+</option>
            <option value="real">Escenario: Real Histórico</option>
          </select>
          <select id="turnSessionFilter" onchange="renderAll()"></select>
        </div>
      </div>
      <div id="turnChartWrap"></div>
      <div id="turnTooltip" class="subcard muted">Selecciona o pasa el cursor sobre cualquier barra para ver el detalle del turno.</div>
    </section>

    <!-- Tabla de sesiones -->
    <section class="card" style="overflow-x:auto;">
      <h2>3. Histórico por Conversación (sin PII)</h2>
      <div class="muted" style="margin-bottom:8px;">Extraído de <span class="mono">data/logs/historico_sesiones_costes.csv</span></div>
      <table>
        <thead>
          <tr>
            <th>Sesión</th><th>Fecha</th><th>Territorio</th><th class="r">Turnos (RAG)</th>
            <th class="r">Tokens</th><th class="r">Cohere</th><th class="r">Real</th><th class="r">Tier Lite</th><th class="r">Tier Flash</th>
          </tr>
        </thead>
        <tbody id="sessionsTableBody" class="num"></tbody>
      </table>
    </section>

    <!-- Simulador SaaS -->
    <section class="card" style="display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;padding-bottom:10px;border-bottom:1px solid var(--border);">
        <div>
          <h2>4. Simulador Interactivo de Monetización y Margen Bruto</h2>
          <div id="simRatioHint" class="muted">Basado en la media real de tus sesiones auditadas</div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
          <select id="simStack" onchange="renderSimulator()">
            <option value="lite">Stack Tier Lite (3.1 + 3.5)</option>
            <option value="hybrid">Híbrido (95% Flash 2026 + 5% Pro)</option>
            <option value="flash26">Stack Tier Flash 2026 (3.1 + 3.8)</option>
            <option value="flash27">Stack Tier Flash 2027+ (3.1 + 3.8)</option>
            <option value="pro">Stack Tier Pro (3.1 + 3.1-pro)</option>
          </select>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;">
            <input id="simUseCohere" type="checkbox" checked onchange="renderSimulator()" />
            Incluir Cohere
          </label>
        </div>
      </div>

      <div class="row g3">
        <div style="display:flex;flex-direction:column;gap:12px;font-size:12px;">
          <div>
            <div style="display:flex;justify-content:space-between;"><span class="muted">Clientes de pago</span><strong id="valUsers" class="num">150</strong></div>
            <input id="rngUsers" type="range" min="10" max="2000" step="10" value="150" oninput="renderSimulator()" />
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;"><span class="muted">Consultas / cliente / mes</span><strong id="valTurns" class="num">60</strong></div>
            <input id="rngTurns" type="range" min="10" max="250" step="5" value="60" oninput="renderSimulator()" />
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;"><span class="muted">Precio suscripción (EUR/mes)</span><strong id="valPrice" class="num">19 €</strong></div>
            <input id="rngPrice" type="range" min="5" max="99" step="1" value="19" oninput="renderSimulator()" />
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;"><span class="muted">Infra fija (EUR/mes)</span><strong id="valInfra" class="num">45 €</strong></div>
            <input id="rngInfra" type="range" min="0" max="200" step="5" value="45" oninput="renderSimulator()" />
          </div>
        </div>

        <div style="display:flex;flex-direction:column;justify-content:space-between;gap:14px;">
          <div class="row g4">
            <div class="subcard"><div class="muted">MRR</div><div id="outMrr" style="font-size:18px;font-weight:700;" class="num">—</div></div>
            <div class="subcard"><div class="muted">Coste API + Infra</div><div id="outCost" style="font-size:18px;font-weight:700;color:var(--warn);" class="num">—</div></div>
            <div class="subcard"><div class="muted">Margen Bruto</div><div id="outMargin" style="font-size:18px;font-weight:700;color:var(--ok);" class="num">—</div></div>
            <div class="subcard"><div class="muted">Break-even</div><div id="outBreakeven" style="font-size:18px;font-weight:700;" class="num">—</div></div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;" class="muted">
              <span>Reparto de ingresos mensuales</span>
              <span id="outPerUser" class="num">—</span>
            </div>
            <div class="bar-track">
              <div id="barMargin" class="bar-seg" style="background:var(--ok);"></div>
              <div id="barLlm" class="bar-seg" style="background:var(--c1);"></div>
              <div id="barCohere" class="bar-seg" style="background:var(--c4);"></div>
              <div id="barInfra" class="bar-seg" style="background:var(--c5);"></div>
            </div>
            <div class="legend">
              <span><span class="dot" style="background:var(--ok)"></span>Margen</span>
              <span><span class="dot" style="background:var(--c1)"></span>Gemini + Emb</span>
              <span><span class="dot" style="background:var(--c4)"></span>Cohere</span>
              <span><span class="dot" style="background:var(--c5)"></span>Infra fija</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const DATA = {payload_json};
    let currency = 'USD';
    let scale = 1;

    function toggleTheme() {{ document.body.classList.toggle('light'); }}
    function setCurrency(c) {{ currency = c; renderAll(); }}
    function setScale(s) {{ scale = s; renderAll(); }}

    function fmt(usdVal, decimals = 4) {{
      const val = (currency === 'EUR' ? usdVal * DATA.usd_to_eur : usdVal) * scale;
      const sym = currency === 'EUR' ? '€' : '$';
      return scale === 1000 ? `${{sym}}${{val.toFixed(2)}}` : `${{sym}}${{val.toFixed(decimals)}}`;
    }}
    function fmtUnit(usdVal, decimals = 5) {{
      const val = currency === 'EUR' ? usdVal * DATA.usd_to_eur : usdVal;
      const sym = currency === 'EUR' ? '€' : '$';
      return `${{sym}}${{val.toFixed(decimals)}}`;
    }}

    function splitTurn(t, mode) {{
      const coh = t.coh * (DATA.cohere_per_1k / 1000.0);
      const emb = t.emb || 0;
      if (mode === 'real') return {{ llm: t.c_real_llm, coh, emb, total: t.c_real }};
      if (mode === 'lite') return {{ llm: Math.max(0, t.c_lite - coh - emb), coh, emb, total: t.c_lite }};
      if (mode === 'flash26') return {{ llm: Math.max(0, t.c_flash - coh - emb), coh, emb, total: t.c_flash }};
      const liteLlm = Math.max(0, t.c_lite - coh - emb);
      const fl26Llm = Math.max(0, t.c_flash - coh - emb);
      if (t.route !== 'parallel_retrieve') return {{ llm: liteLlm, coh: 0, emb: 0, total: liteLlm }};
      const routerEst = (t.tin * 0.25 * DATA.rates.lite_router_in + t.tout * 0.25 * DATA.rates.lite_router_out) / 1e6;
      const synth26 = Math.max(0.0005, fl26Llm - routerEst);
      if (mode === 'flash27') {{
        const llm = routerEst + synth26 * 2.0;
        return {{ llm, coh, emb, total: llm + coh + emb }};
      }}
      const llm = routerEst + synth26 * 3.15;
      return {{ llm, coh, emb, total: llm + coh + emb }};
    }}

    function initSelectors() {{
      const f1 = document.getElementById('sessionFilter1');
      f1.innerHTML = '<option value="avg_rag">Media sesiones RAG</option>';
      DATA.sessions.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.sid;
        opt.textContent = `${{s.sid}} · ${{s.territorio}} (${{s.turnos}}t, ${{(s.tok/1000).toFixed(1)}}k tok)`;
        f1.appendChild(opt);
      }});

      const f2 = document.getElementById('turnSessionFilter');
      f2.innerHTML = `<option value="all">Últimos ${{Math.min(30, DATA.turns.length)}} turnos (de ${{DATA.turns.length}})</option>`;
      DATA.sessions.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.sid;
        opt.textContent = `${{s.sid}} (${{s.turnos}}t)`;
        f2.appendChild(opt);
      }});
    }}

    function renderHeaderAndRadar() {{
      document.getElementById('hdrMeta').textContent =
        `Actualizado: ${{DATA.generated_at}} | Fuente tarifas: ${{DATA.pricing_source}} (${{DATA.as_of_date}}) | Sesiones CSV: ${{DATA.total_sessions_all}} (${{DATA.sessions.length}} con telemetría) · Turnos CSV: ${{DATA.total_turns_all}}`;
      document.getElementById('cohereRateLbl').textContent = `$${{DATA.cohere_per_1k.toFixed(2)}} / 1k searches`;
      document.getElementById('fxLbl').textContent = `1 USD = ${{DATA.usd_to_eur}} EUR`;

      const r = DATA.radar || {{}};
      const badge = document.getElementById('radarBadge');
      const cards = [];
      const addCard = (title, items, cls, emptyTxt) => {{
        const txt = (items && items.length) ? items.map(x => `<div class="mono" style="margin-top:3px;">• ${{x}}</div>`).join('') : `<div class="muted" style="margin-top:4px;">${{emptyTxt}}</div>`;
        cards.push(`<div class="subcard"><div style="font-weight:600;color:var(--${{cls}});">${{title}}</div>${{txt}}</div>`);
      }};
      addCard('Cambios de tarifa hoy', r.price_changes, (r.price_changes||[]).length ? 'warn' : 'ok', 'Sin cambios frente al snapshot previo');
      addCard('Subidas programadas futuras', r.upcoming_price_changes, (r.upcoming_price_changes||[]).length ? 'warn' : 'ok', 'Ninguna subida fechada detectada');
      addCard('Modelos nuevos / retirados', [...(r.new_models||[]).map(m=>'+ '+m), ...(r.removed_models||[]).map(m=>'- '+m)], ((r.new_models||[]).length||(r.removed_models||[]).length) ? 'primary' : 'ok', 'Catálogo estable frente al snapshot previo');
      addCard('Salud modelos en config.py', r.deprecated_configured_models, (r.deprecated_configured_models||[]).length ? 'err' : 'ok', 'Todos los modelos configurados vigentes');

      document.getElementById('radarAlerts').innerHTML = cards.join('');
      if ((r.deprecated_configured_models || []).length) {{
        badge.className = 'badge err';
        badge.textContent = 'Atención: modelo deprecado o ausente';
      }} else if (r.needs_attention) {{
        badge.className = 'badge warn';
        badge.textContent = 'Avisos / cambios programados detectados';
      }} else {{
        badge.className = 'badge ok';
        badge.textContent = 'Todo al día';
      }}

      document.getElementById('modelsTableBody').innerHTML = DATA.tracked_models.map(m => {{
        let st = '<span class="badge ok">Vigente</span>';
        if (m.deprecated) st = '<span class="badge err">Deprecado</span>';
        else if (m.upcoming) st = `<span class="badge warn" title="${{m.upcoming}}">Subida prog.</span>`;
        return `<tr>
          <td class="mono">${{m.model}}</td>
          <td class="muted">${{m.role}}</td>
          <td class="r">$${{m.in_1m.toFixed(2)}}</td>
          <td class="r">$${{m.cache_1m.toFixed(3)}}</td>
          <td class="r">$${{m.out_1m.toFixed(2)}}</td>
          <td>${{st}}</td>
        </tr>`;
      }}).join('');
    }}

    function renderKpis() {{
      const rag = DATA.turns.filter(t => t.route === 'parallel_retrieve');
      const sc = DATA.turns.filter(t => t.route !== 'parallel_retrieve');
      const avgRagLite = rag.length ? rag.reduce((a, t) => a + t.c_lite, 0) / rag.length : 0;
      const avgRagFl = rag.length ? rag.reduce((a, t) => a + t.c_flash, 0) / rag.length : 0;
      const avgScLite = sc.length ? sc.reduce((a, t) => a + t.c_lite, 0) / sc.length : 0;
      const ragCohereSum = rag.reduce((a, t) => a + t.coh * (DATA.cohere_per_1k / 1000.0), 0);
      const ragLiteSum = rag.reduce((a, t) => a + t.c_lite, 0);
      const cohPct = ragLiteSum > 0 ? (100 * ragCohereSum / ragLiteSum) : 0;
      const totLite = DATA.turns.reduce((a, t) => a + t.c_lite, 0);
      const totTok = DATA.turns.reduce((a, t) => a + t.tot, 0);
      const totCached = DATA.turns.reduce((a, t) => a + t.tc, 0);

      document.getElementById('kpiRag').textContent = fmtUnit(avgRagLite, 5);
      document.getElementById('kpiRagSub').textContent = `Tier Flash 2026: ${{fmtUnit(avgRagFl, 5)}} (${{rag.length}} turnos RAG)`;
      document.getElementById('kpiSc').textContent = fmtUnit(avgScLite, 5);
      document.getElementById('kpiScSub').textContent = avgScLite > 0 ? `${{(avgRagLite/avgScLite).toFixed(1)}}x más barato (${{sc.length}} cortocircuitos)` : 'Sin cortocircuitos';
      document.getElementById('kpiCohere').textContent = `${{cohPct.toFixed(1)}} %`;
      document.getElementById('kpiCohereSub').textContent = `${{rag.reduce((a,t)=>a+t.coh,0)}} search units · $${{ragCohereSum.toFixed(3)}} USD`;
      document.getElementById('kpiTotal').textContent = fmtUnit(totLite, 4);
      document.getElementById('kpiTotalSub').textContent = `${{totTok.toLocaleString('es-ES')}} tok (${{totCached.toLocaleString('es-ES')}} caché)`;
    }}

    function renderScenarioChart() {{
      if (!DATA.turns.length) {{
        document.getElementById('scenarioChartWrap').innerHTML = '<div class="muted">Aún no hay turnos con telemetría detallada en los CSVs.</div>';
        return;
      }}
      const sel = document.getElementById('sessionFilter1').value;
      let groups = [];
      if (sel === 'avg_rag') {{
        const ragSids = DATA.sessions.filter(s => s.rag > 0).map(s => s.sid);
        groups = (ragSids.length ? ragSids : DATA.sessions.map(s => s.sid)).map(sid => DATA.turns.filter(t => t.sid === sid));
      }} else {{
        groups = [DATA.turns.filter(t => t.sid === sel)];
      }}
      const modes = [
        {{ key: 'real', label: 'Real Histórico' }},
        {{ key: 'lite', label: 'Tier Lite (3.1 + 3.5)' }},
        {{ key: 'flash26', label: 'Flash 2026 (3.8-flash)' }},
        {{ key: 'flash27', label: 'Flash 2027+ / 4-Hyd.' }},
        {{ key: 'pro', label: 'Tier Pro (3.1-pro)' }}
      ];
      const rows = modes.map(m => {{
        let llm = 0, coh = 0, emb = 0;
        groups.forEach(g => g.forEach(t => {{
          const sp = splitTurn(t, m.key);
          llm += sp.llm; coh += sp.coh; emb += sp.emb;
        }}));
        const div = Math.max(1, groups.length);
        llm /= div; coh /= div; emb /= div;
        return {{ ...m, llm, coh, emb, total: llm + coh + emb }};
      }});
      const maxVal = Math.max(0.0001, ...rows.map(r => r.total)) * 1.18;
      const W = 580, H = 220, padL = 165, padR = 85, padT = 14, barH = 24, gap = 14;
      let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;">`;
      rows.forEach((d, i) => {{
        const y = padT + i * (barH + gap);
        const wTot = ((W - padL - padR) * d.total) / maxVal;
        const wLlm = d.total > 0 ? (wTot * d.llm) / d.total : 0;
        const wCoh = d.total > 0 ? (wTot * d.coh) / d.total : 0;
        const wEmb = Math.max(0, wTot - wLlm - wCoh);
        svg += `<text x="${{padL - 8}}" y="${{y + 16}}" text-anchor="end" fill="var(--fg)" font-size="12">${{d.label}}</text>`;
        svg += `<rect x="${{padL}}" y="${{y}}" width="${{wLlm.toFixed(1)}}" height="${{barH}}" rx="3" fill="var(--c1)"/>`;
        if (wCoh > 0.5) svg += `<rect x="${{(padL + wLlm).toFixed(1)}}" y="${{y}}" width="${{wCoh.toFixed(1)}}" height="${{barH}}" fill="var(--c4)"/>`;
        if (wEmb > 0.5) svg += `<rect x="${{(padL + wLlm + wCoh).toFixed(1)}}" y="${{y}}" width="${{wEmb.toFixed(1)}}" height="${{barH}}" fill="var(--c3)"/>`;
        svg += `<text x="${{(padL + wTot + 8).toFixed(1)}}" y="${{y + 16}}" fill="var(--fg)" font-size="12" font-weight="600">${{fmt(d.total, 4)}}</text>`;
      }});
      svg += `</svg>`;
      document.getElementById('scenarioChartWrap').innerHTML = svg;
    }}

    function showTurnDetail(turnId, mode) {{
      const t = DATA.turns.find(x => x.id === turnId);
      if (!t) return;
      const s = splitTurn(t, mode);
      const mult = currency === 'EUR' ? DATA.usd_to_eur : 1;
      const sym = currency === 'EUR' ? '€' : '$';
      document.getElementById('turnTooltip').innerHTML =
        `<strong>Turno #${{t.id}}</strong> (${{t.sid}} · t${{t.idx}} · ${{t.route}} · ${{t.terr}}) — ` +
        `Tokens: <strong>${{t.tot.toLocaleString('es-ES')}}</strong> (in ${{t.tin.toLocaleString('es-ES')}}, out ${{t.tout.toLocaleString('es-ES')}}, th ${{t.th}}, cache ${{t.tc}}) · ` +
        `Latencia: <strong>${{(t.lat/1000).toFixed(1)}}s</strong> · ` +
        `Coste: <strong style="color:var(--primary);">${{sym}}${{(s.total*mult).toFixed(5)}}</strong> (LLM ${{sym}}${{(s.llm*mult).toFixed(5)}} + Cohere ${{sym}}${{(s.coh*mult).toFixed(5)}})`;
    }}

    function renderTurnChart() {{
      const mode = document.getElementById('turnMetricSelect').value;
      const sf = document.getElementById('turnSessionFilter').value;
      const raw = sf === 'all' ? DATA.turns.slice(-30) : DATA.turns.filter(t => t.sid === sf);
      if (!raw.length) {{
        document.getElementById('turnChartWrap').innerHTML = '<div class="muted">Sin turnos disponibles.</div>';
        return;
      }}
      const items = raw.map(t => ({{ t, s: splitTurn(t, mode) }}));
      const maxVal = Math.max(0.0001, ...items.map(x => x.s.total)) * 1.18;
      const W = 860, H = 220, padL = 54, padR = 16, padT = 18, padB = 38;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const colW = plotW / items.length;
      const barW = Math.min(34, colW * 0.7);
      const mult = currency === 'EUR' ? DATA.usd_to_eur : 1;
      const sym = currency === 'EUR' ? '€' : '$';

      let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;">`;
      [0, 0.5, 1].forEach(frac => {{
        const y = padT + plotH * (1 - frac);
        svg += `<line x1="${{padL}}" y1="${{y.toFixed(1)}}" x2="${{W - padR}}" y2="${{y.toFixed(1)}}" stroke="var(--border)" stroke-dasharray="3 3"/>`;
        svg += `<text x="${{padL - 6}}" y="${{(y + 4).toFixed(1)}}" text-anchor="end" fill="var(--muted)" font-size="11">${{sym}}${{(maxVal * frac * mult).toFixed(4)}}</text>`;
      }});

      items.forEach((item, idx) => {{
        const {{ t, s }} = item;
        const cx = padL + idx * colW + colW / 2;
        const x = cx - barW / 2;
        const hTot = (plotH * s.total) / maxVal;
        const hCoh = (plotH * s.coh) / maxVal;
        const hLlm = Math.max(0, hTot - hCoh);
        const yCoh = padT + plotH - hCoh;
        const yLlm = yCoh - hLlm;
        const fill = t.route === 'parallel_retrieve' ? 'var(--c1)' : 'var(--c2)';
        svg += `<g style="cursor:pointer;" onmouseenter="showTurnDetail(${{t.id}}, '${{mode}}')" onclick="showTurnDetail(${{t.id}}, '${{mode}}')">`;
        svg += `<rect x="${{x.toFixed(1)}}" y="${{yLlm.toFixed(1)}}" width="${{barW.toFixed(1)}}" height="${{hLlm.toFixed(1)}}" rx="2" fill="${{fill}}"/>`;
        if (hCoh > 0.5) svg += `<rect x="${{x.toFixed(1)}}" y="${{yCoh.toFixed(1)}}" width="${{barW.toFixed(1)}}" height="${{hCoh.toFixed(1)}}" fill="var(--c4)"/>`;
        svg += `<text x="${{cx.toFixed(1)}}" y="${{H - 18}}" text-anchor="middle" fill="var(--fg)" font-size="10">#${{t.id}}</text>`;
        svg += `<text x="${{cx.toFixed(1)}}" y="${{H - 5}}" text-anchor="middle" fill="var(--muted)" font-size="9">${{t.route === 'parallel_retrieve' ? 'RAG' : 'HITL'}}</text>`;
        svg += `</g>`;
      }});
      svg += `</svg>`;
      document.getElementById('turnChartWrap').innerHTML = svg;
    }}

    function renderSessionsTable() {{
      const tbody = document.getElementById('sessionsTableBody');
      tbody.innerHTML = [...DATA.sessions].reverse().map(s => `
        <tr>
          <td class="mono">${{s.sid}}</td>
          <td>${{s.fecha}}</td>
          <td>${{s.territorio}}</td>
          <td class="r">${{s.turnos}} (${{s.rag}})</td>
          <td class="r">${{s.tok.toLocaleString('es-ES')}}</td>
          <td class="r">${{s.coh}}</td>
          <td class="r">${{fmtUnit(s.c_real, 4)}}</td>
          <td class="r">${{fmtUnit(s.c_lite, 4)}}</td>
          <td class="r">${{fmtUnit(s.c_flash, 4)}}</td>
        </tr>
      `).join('');
    }}

    function renderSimulator() {{
      const users = +document.getElementById('rngUsers').value;
      const turnsPerUser = +document.getElementById('rngTurns').value;
      const priceEur = +document.getElementById('rngPrice').value;
      const infraEur = +document.getElementById('rngInfra').value;
      const stack = document.getElementById('simStack').value;
      const useCohere = document.getElementById('simUseCohere').checked;

      document.getElementById('valUsers').textContent = users.toLocaleString('es-ES');
      document.getElementById('valTurns').textContent = turnsPerUser;
      document.getElementById('valPrice').textContent = `${{priceEur}} €`;
      document.getElementById('valInfra').textContent = `${{infraEur}} €`;

      const rag = DATA.turns.filter(t => t.route === 'parallel_retrieve');
      const sc = DATA.turns.filter(t => t.route !== 'parallel_retrieve');
      const ragShare = DATA.turns.length ? (rag.length / DATA.turns.length) : 0.7;

      const avgCost = (modeKey) => {{
        const rAvg = rag.length ? rag.reduce((a, t) => {{
          const sp = splitTurn(t, modeKey);
          return {{ llm: a.llm + sp.llm + sp.emb, coh: a.coh + sp.coh }};
        }}, {{ llm: 0, coh: 0 }}) : {{ llm: 0.0065, coh: 0.002 }};
        if (rag.length) {{ rAvg.llm /= rag.length; rAvg.coh /= rag.length; }}
        const sAvg = sc.length ? sc.reduce((a, t) => a + splitTurn(t, modeKey).llm, 0) / sc.length : 0.0017;
        return {{
          llmUsd: ragShare * rAvg.llm + (1 - ragShare) * sAvg,
          cohUsd: useCohere ? ragShare * rAvg.coh : 0,
        }};
      }};

      let perTurn;
      if (stack === 'hybrid') {{
        const fl = avgCost('flash26');
        const pr = avgCost('pro');
        perTurn = {{ llmUsd: 0.95 * fl.llmUsd + 0.05 * pr.llmUsd, cohUsd: fl.cohUsd }};
      }} else {{
        perTurn = avgCost(stack);
      }}

      const mrr = users * priceEur;
      const totalTurns = users * turnsPerUser;
      const llmEur = totalTurns * perTurn.llmUsd * DATA.usd_to_eur;
      const cohEur = totalTurns * perTurn.cohUsd * DATA.usd_to_eur;
      const totalCost = llmEur + cohEur + infraEur;
      const margin = mrr - totalCost;
      const marginPct = mrr > 0 ? (100 * margin / mrr) : 0;
      const varPerUser = turnsPerUser * (perTurn.llmUsd + perTurn.cohUsd) * DATA.usd_to_eur;
      const contrib = priceEur - varPerUser;
      const be = contrib > 0 ? Math.ceil(infraEur / contrib) : Infinity;

      document.getElementById('simRatioHint').textContent = `Distribución real observada en turnos: ${{(ragShare*100).toFixed(0)}}% RAG / ${{((1-ragShare)*100).toFixed(0)}}% cortocircuitos`;
      document.getElementById('outMrr').textContent = `${{mrr.toFixed(0)}} €/mes`;
      document.getElementById('outCost').textContent = `${{totalCost.toFixed(1)}} €`;
      document.getElementById('outMargin').textContent = `${{margin.toFixed(0)}} € (${{marginPct.toFixed(1)}}%)`;
      document.getElementById('outBreakeven').textContent = isFinite(be) ? `${{be}} clientes` : 'Inviable';
      document.getElementById('outPerUser').textContent = `Coste variable/cliente: ${{varPerUser.toFixed(2)}} €/mes`;

      const denom = Math.max(mrr, totalCost, 1);
      const pM = Math.max(0, (100 * margin) / denom);
      document.getElementById('barMargin').style.width = `${{pM.toFixed(1)}}%`;
      document.getElementById('barMargin').textContent = pM > 10 ? `${{pM.toFixed(0)}}%` : '';
      document.getElementById('barLlm').style.width = `${{((100 * llmEur) / denom).toFixed(1)}}%`;
      document.getElementById('barCohere').style.width = `${{((100 * cohEur) / denom).toFixed(1)}}%`;
      document.getElementById('barInfra').style.width = `${{((100 * infraEur) / denom).toFixed(1)}}%`;
    }}

    function updateButtons() {{
      document.getElementById('btnUsd').className = currency === 'USD' ? 'active' : '';
      document.getElementById('btnEur').className = currency === 'EUR' ? 'active' : '';
      document.getElementById('btnPerConv').className = scale === 1 ? 'active' : '';
      document.getElementById('btnPer1k').className = scale === 1000 ? 'active' : '';
    }}

    function renderAll() {{
      updateButtons();
      renderHeaderAndRadar();
      renderKpis();
      renderScenarioChart();
      renderTurnChart();
      renderSessionsTable();
      renderSimulator();
    }}

    initSelectors();
    renderAll();
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".html.tmp")
    tmp_path.write_text(html, encoding="utf-8")
    tmp_path.replace(output_path)

    return {
        "dashboard_path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sessions_in_dashboard": len(payload["sessions"]),
        "turns_in_dashboard": len(payload["turns"]),
        "radar_needs_attention": catalog.radar.needs_attention,
    }
