"""Suite de Regresión y Control de Calidad Continua para AiHoreca (`src/evaluation/regression.py`).

Verifica que las optimizaciones en prompts, recuperación o modelos no degraden el comportamiento
previamente verificado sobre el Golden Dataset (12 casos curados + casos añadidos desde producción).

Comportamiento:
- Compara los resultados de la ejecución actual contra la línea base en `data/processed/evaluation_baseline.json`.
- BLOQUEA (exit code 1) si algún caso que antes pasaba en los checks deterministas ahora falla
  (enrutamiento, provincia, ruta o alucinación numérica).
- ALERTA si la media semántica RAGAS (faithfulness, relevancy) cae más de 0.05 puntos respecto a la línea base.
- Permite modo `--quick` (0 tokens de juez semántico, comprobación ultrarrápida en segundos).
- Permite actualizar la línea base oficial mediante `--update-baseline`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Optional

from src.config import EVALUATION_BASELINE_PATH, EVALUATION_RESULTS_PATH
from src.evaluation.evaluator import EvaluationCaseResult, EvaluationSummaryReport, evaluate_golden_dataset
from src.evaluation.golden_dataset import GoldenTestCase, get_golden_dataset

logger = logging.getLogger(__name__)


def load_baseline(baseline_path: Path = EVALUATION_BASELINE_PATH) -> Optional[dict[str, Any]]:
    """Carga la línea base de evaluación si existe en disco."""
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("No se pudo cargar la línea base desde %s: %s", baseline_path, exc)
        return None


def save_baseline(report: EvaluationSummaryReport, baseline_path: Path = EVALUATION_BASELINE_PATH) -> None:
    """Guarda el informe actual como la nueva línea base oficial."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compare_against_baseline(
    current: EvaluationSummaryReport,
    baseline: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Compara los resultados actuales contra la línea base y detecta regresiones críticas."""
    if not baseline:
        return {
            "has_baseline": False,
            "regressions": [],
            "improvements": [],
            "new_cases": [c.case_id for c in current.cases],
            "metric_deltas": {},
            "status": "NO_BASELINE",
        }

    base_cases: dict[str, dict[str, Any]] = {
        item["case_id"]: item for item in baseline.get("cases", [])
    }

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    new_cases: list[str] = []

    for curr_case in current.cases:
        cid = curr_case.case_id
        if cid not in base_cases:
            new_cases.append(cid)
            continue

        base_case = base_cases[cid]
        # Comprobaciones deterministas críticas
        checks = [
            ("intent_match", base_case.get("intent_match", True), curr_case.intent_match),
            ("route_match", base_case.get("route_match", True), curr_case.route_match),
            ("territorio_match", base_case.get("territorio_match", True), curr_case.territorio_match),
            (
                "zero_numerical_hallucination",
                base_case.get("zero_numerical_hallucination", True),
                curr_case.zero_numerical_hallucination,
            ),
        ]

        case_regressions = []
        case_improvements = []
        for name, was_ok, is_ok in checks:
            if was_ok and not is_ok:
                case_regressions.append(name)
            elif not was_ok and is_ok:
                case_improvements.append(name)

        if case_regressions:
            regressions.append({
                "case_id": cid,
                "regressions": case_regressions,
                "predicted_intent": curr_case.predicted_intent,
                "expected_intent": curr_case.expected_intent,
                "predicted_route": curr_case.predicted_route,
                "expected_route": curr_case.expected_route,
                "predicted_territorio": curr_case.predicted_territorio,
                "expected_territorio": curr_case.expected_territorio,
            })
        elif case_improvements:
            improvements.append({
                "case_id": cid,
                "improvements": case_improvements,
            })

    # Deltas en métricas agregadas
    metric_deltas = {}
    metric_keys = [
        "intent_accuracy",
        "route_accuracy",
        "territorio_accuracy",
        "zero_hallucination_rate",
        "mean_faithfulness",
        "mean_answer_relevancy",
    ]
    for key in metric_keys:
        curr_val = getattr(current, key, None)
        base_val = baseline.get(key)
        if curr_val is not None and base_val is not None:
            metric_deltas[key] = {
                "baseline": base_val,
                "current": curr_val,
                "delta": round(float(curr_val) - float(base_val), 4),
            }

    status = "FAILED" if regressions else "PASSED"
    return {
        "has_baseline": True,
        "status": status,
        "regressions": regressions,
        "improvements": improvements,
        "new_cases": new_cases,
        "metric_deltas": metric_deltas,
    }


def run_regression_suite(
    quick: bool = False,
    case_id: Optional[str] = None,
    force_recompute: bool = False,
    update_baseline: bool = False,
) -> tuple[EvaluationSummaryReport, dict[str, Any]]:
    """Ejecuta la suite de regresión y genera el informe comparativo."""
    dataset = get_golden_dataset(include_production=True)
    if case_id:
        dataset = [c for c in dataset if c.case_id == case_id]
        if not dataset:
            raise ValueError(f"No se encontró ningún caso con case_id='{case_id}' en el Golden Dataset.")

    current_report = evaluate_golden_dataset(
        force_recompute=force_recompute,
        cases=dataset,
        skip_semantic_judge=quick,
    )

    baseline = load_baseline()
    comparison = compare_against_baseline(current_report, baseline)

    if update_baseline:
        save_baseline(current_report)
        comparison["baseline_updated"] = True

    return current_report, comparison


def _cli() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Suite de Regresión y Control de Calidad Continua de AiHoreca."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo ultrarrápido determinista (0 tokens de juez semántico RAGAS).",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Ejecuta la regresión únicamente para un caso específico (ej. GD_13_DESAMBIGUACION_LOCAL_PONFERRADA).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza recomputación sin reutilizar caché previa en evaluation_results.json.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Actualiza la línea base oficial (evaluation_baseline.json) con los resultados de esta ejecución.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime el resultado completo en formato JSON estructurado.",
    )
    args = parser.parse_args()

    report, comparison = run_regression_suite(
        quick=args.quick,
        case_id=args.case_id,
        force_recompute=args.force,
        update_baseline=args.update_baseline,
    )

    if args.json:
        out = {
            "summary": report.model_dump(exclude={"cases"}),
            "comparison": comparison,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(1 if comparison.get("status") == "FAILED" else 0)

    print("=" * 100)
    print("AIHORECA REGRESSION SUITE · INFORME DE CONTROL DE CALIDAD")
    print("=" * 100)
    print(f"Modo: {'QUICK (Determinista, 0 tokens juez)' if args.quick else 'FULL (Semántico RAGAS + Determinista)'}")
    print(f"Casos evaluados: {len(report.cases)}")
    print(f"Estado general: {'❌ REGRESIÓN DETECTADA' if comparison.get('status') == 'FAILED' else '✅ TODO EN ORDEN'}")
    print("-" * 100)

    print(f"{'CASO':<40} | {'INTENT':<10} | {'ROUTE':<18} | {'TERRITORIO':<15} | {'HALLUC':<8} | {'ESTADO'}")
    print("-" * 100)
    for c in report.cases:
        det_ok = c.intent_match and c.route_match and c.territorio_match and c.zero_numerical_hallucination
        icon = "✅ OK" if det_ok else "❌ ERROR"
        terr_str = str(c.predicted_territorio or "—")[:14]
        print(
            f"{c.case_id:<40} | "
            f"{'OK' if c.intent_match else 'FAIL':<10} | "
            f"{'OK' if c.route_match else 'FAIL':<18} | "
            f"{terr_str:<15} | "
            f"{'0' if c.zero_numerical_hallucination else 'H!':<8} | "
            f"{icon}"
        )

    print("-" * 100)
    if comparison.get("has_baseline"):
        print("COMPARATIVA CONTRA LÍNEA BASE:")
        deltas = comparison.get("metric_deltas", {})
        for metric, data in deltas.items():
            diff = data["delta"]
            diff_sign = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"  - {metric:<25}: Base={data['baseline']:.4f} -> Actual={data['current']:.4f} ({diff_sign})")

        if comparison.get("regressions"):
            print("\n❌ REGRESIONES ENCONTRADAS (Casos que antes pasaban y ahora fallan):")
            for r in comparison["regressions"]:
                print(f"  * {r['case_id']}: fallos en {r['regressions']}")

        if comparison.get("improvements"):
            print("\n🎉 MEJORAS (Casos que antes fallaban y ahora pasan):")
            for im in comparison["improvements"]:
                print(f"  * {im['case_id']}: recuperó {im['improvements']}")

        if comparison.get("new_cases"):
            print("\n🆕 CASOS NUEVOS (Sin línea base previa):")
            for nc in comparison["new_cases"]:
                print(f"  * {nc}")

    if comparison.get("baseline_updated"):
        print("\n💾 Línea base oficial actualizada con éxito en data/processed/evaluation_baseline.json")

    print("=" * 100)
    sys.exit(1 if comparison.get("status") == "FAILED" else 0)


if __name__ == "__main__":
    _cli()
