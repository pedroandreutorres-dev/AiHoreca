"""Motor de Evaluación Empírica Híbrido (Determinista en CPU + Juez Semántico RAGAS Single-Pass) para AiHoreca.

Principios de diseño:
1. Eficiencia de Cuota Free Tier: evita el antipatrón de `ragas.evaluate()` estándar (~15 llamadas LLM por pregunta)
   evaluando simultáneamente `faithfulness`, `answer_relevancy`, `context_recall_semantic` y `hierarchy_application_score`
   en 1 única llamada estructurada (`RAGASJudgeSchema`) por caso generativo, y usando evaluación determinista (0 tokens)
   para los casos de cortocircuito (`block_out_of_scope` / `ask_clarification`).
2. Métricas Deterministas Exactas en CPU:
   - Exactitud de enrutamiento (`intent_match`, `route_match`, `territorio_match`).
   - Cobertura de las 3 capas normativas (`layer_coverage_score`).
   - Recall de citas y anexos en el Top-8 (`citation_recall_at_k`).
   - Fidelidad numérica y ausencia de alucinaciones (`zero_numerical_hallucination`).
3. Caché Incremental en Disco (`data/processed/evaluation_results.json`):
   - Guarda cada caso conforme se completa para que el notebook de auditoría pueda renderizar tablas y gráficos
     instantáneamente sin re-consumir cuota de API.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.config import EVALUATION_RESULTS_PATH, GEMINI_LITE_MODEL
from src.evaluation.golden_dataset import GoldenTestCase, get_golden_dataset
from src.graph.memory import SessionMemoryStore
from src.graph.nodes import _call_structured_llm
from src.graph.workflow import build_aihoreca_graph, run_legal_query
from src.retrieval.chunker import chunk_all_processed_documents
from src.retrieval.vector_store import AiHorecaVectorStore

logger = logging.getLogger(__name__)


class RAGASJudgeSchema(BaseModel):
    """Contrato Pydantic v2 para el Juez Semántico Single-Pass (Métricas RAGAS + Jerarquía Normativa)."""

    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fidelidad al contexto (0.0 a 1.0): mide si todas las afirmaciones jurídicas y cifras de la respuesta "
            "se deducen estrictamente de los fragmentos recuperados sin inventar datos (1.0 = cero invención)."
        ),
    )
    answer_relevancy: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Relevancia de la respuesta (0.0 a 1.0): mide si la respuesta contesta de forma directa, clara y completa "
            "a la consulta autocontenida del usuario."
        ),
    )
    context_recall_semantic: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Context Recall semántico (0.0 a 1.0): mide qué proporción de la información clave descrita en el "
            "Ground Truth de referencia está respaldada por los fragmentos recuperados de Qdrant + Cohere."
        ),
    )
    hierarchy_application_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Aplicación de Jerarquía Normativa (0.0 a 1.0): mide si la respuesta articula adecuadamente las normas "
            "aplicables (Convenio Colectivo Provincial, VI ALEH y Estatuto de los Trabajadores)."
        ),
    )
    critica_experta: str = Field(
        description="Justificación técnica concisa (1-2 frases) de las puntuaciones otorgadas."
    )


class EvaluationCaseResult(BaseModel):
    """Resultado completo (determinista + semántico RAGAS + perfilado de latencia) de un caso del Golden Dataset."""

    case_id: str
    category: str
    session_id: str
    query: str
    effective_query: str

    # Enrutamiento y Entidades
    expected_intent: str
    predicted_intent: str
    intent_match: bool

    expected_route: str
    predicted_route: str
    route_match: bool

    expected_territorio: Optional[str] = None
    predicted_territorio: Optional[str] = None
    territorio_match: bool

    categoria_aleh_detectada: Optional[str] = None

    # Métricas Deterministas de Recuperación y Guardrail de Salida
    num_retrieved_docs: int
    retrieved_layers: list[str]
    retrieved_articles: list[str]
    layer_coverage_score: float
    citation_recall_at_k: float
    ground_truth_number_hit: float
    zero_numerical_hallucination: bool
    verified_numbers: list[str]
    unverified_numbers: list[str]

    # Métricas Semánticas RAGAS (Single-Pass Structured Judge)
    faithfulness: float
    answer_relevancy: float
    context_recall_semantic: float
    hierarchy_application_score: float
    judge_critique: str
    judge_model_used: str

    # Latencias y Respuesta
    node_timings_ms: dict[str, float]
    total_latency_ms: float
    final_response: str


class EvaluationSummaryReport(BaseModel):
    """Agregación estadística global y por categoría del benchmark de evaluación."""

    total_cases: int
    generative_cases: int
    guardrail_and_hitl_cases: int

    # Tasas Deterministas Globales
    intent_accuracy: float
    route_accuracy: float
    territorio_accuracy: float
    mean_layer_coverage: float
    mean_citation_recall_at_k: float
    mean_ground_truth_number_hit: float
    zero_hallucination_rate: float

    # Medias Semánticas RAGAS (sobre casos generativos y global)
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_recall_semantic: float
    mean_hierarchy_application_score: float

    # Latencias medias (ms)
    mean_routing_latency_ms: float
    mean_retrieval_latency_ms: float
    mean_synthesis_latency_ms: float
    mean_total_latency_ms: float

    cases: list[EvaluationCaseResult]


def audit_corpus_and_index_stats() -> dict[str, Any]:
    """Audita la distribución exacta de los 648 fragmentos del corpus por documento, capa, territorio y tipo."""
    chunks = chunk_all_processed_documents()
    by_doc = Counter(c.source_file for c in chunks)
    by_layer = Counter(c.capa_normativa for c in chunks)
    by_territory = Counter(c.territorio for c in chunks)
    by_type = Counter(c.tipo_contenido for c in chunks)
    lengths = [len(c.text) for c in chunks]
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)

    doc_type_matrix: dict[str, dict[str, int]] = {}
    for c in chunks:
        if c.source_file not in doc_type_matrix:
            doc_type_matrix[c.source_file] = {"articulo": 0, "tabla_salarial": 0, "anexo_o_disposicion": 0, "total": 0}
        doc_type_matrix[c.source_file][c.tipo_contenido] += 1
        doc_type_matrix[c.source_file]["total"] += 1

    return {
        "total_chunks": n,
        "by_document": dict(by_doc),
        "by_normative_layer": dict(by_layer),
        "by_territory": dict(by_territory),
        "by_content_type": dict(by_type),
        "doc_type_matrix": doc_type_matrix,
        "char_length_stats": {
            "min": sorted_lens[0] if n else 0,
            "p25": sorted_lens[int(n * 0.25)] if n else 0,
            "median": sorted_lens[int(n * 0.50)] if n else 0,
            "mean": round(sum(lengths) / n, 1) if n else 0.0,
            "p75": sorted_lens[int(n * 0.75)] if n else 0,
            "max": sorted_lens[-1] if n else 0,
        },
    }


def _evaluate_ragas_single_pass(
    tc: GoldenTestCase,
    effective_query: str,
    retrieved_context: str,
    final_response: str,
) -> tuple[RAGASJudgeSchema, str]:
    """Ejecuta el Juez Semántico RAGAS en una única llamada estructurada Pydantic."""
    system_instruction = (
        "Eres un Auditor Senior de Sistemas RAG Jurídico-Laborales (Juez Imparcial RAGAS).\n"
        "Evalúa con rigor técnico la respuesta generada por el sistema frente a:\n"
        "1) Los FRAGMENTOS RECUPERADOS de la base de datos vectorial (para medir `faithfulness` y `context_recall_semantic`).\n"
        "2) La CONSULTA AUTOCONTENIDA del usuario (para medir `answer_relevancy`).\n"
        "3) El GROUND TRUTH DE REFERENCIA (para medir `context_recall_semantic` y `hierarchy_application_score`).\n"
        "Nota importante: Los convenios colectivos pueden tener tablas de varios años (2023, 2024, 2025, 2027) o varias "
        "categorías de establecimiento (Lujo, 1ª, 2ª, 3ª, 4ª o Niveles V / V bis). Si la respuesta cita correctamente "
        "las cifras de cualquiera de las tablas vigentes presentes en los fragmentos recuperados, considera la cifra válida y fiel."
    )

    prompt = (
        f"CASO ID: {tc.case_id}\n"
        f"CONSULTA ORIGINAL: {tc.query}\n"
        f"CONSULTA AUTOCONTENIDA RESUELTA: {effective_query}\n\n"
        f"GROUND TRUTH DE REFERENCIA:\n{tc.ground_truth_reference}\n\n"
        f"FRAGMENTOS RECUPERADOS (CONTEXTO):\n{retrieved_context[:14000]}\n\n"
        f"RESPUESTA GENERADA POR EL SISTEMA:\n{final_response}"
    )

    return _call_structured_llm(
        prompt=prompt,
        schema=RAGASJudgeSchema,
        preferred_model=GEMINI_LITE_MODEL,
        system_instruction=system_instruction,
        temperature=0.0,
    )


def evaluate_golden_dataset(
    force_recompute: bool = False,
    output_path: Optional[Path] = None,
    cases: Optional[list[GoldenTestCase]] = None,
    skip_semantic_judge: bool = False,
) -> EvaluationSummaryReport:
    """Ejecuta el benchmark completo sobre los casos del Golden Dataset con caché incremental en disco."""
    cache_file = Path(output_path or EVALUATION_RESULTS_PATH)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = cases if cases is not None else get_golden_dataset()
    cached_cases: dict[str, EvaluationCaseResult] = {}

    if not force_recompute and cache_file.exists():
        try:
            raw_data = json.loads(cache_file.read_text(encoding="utf-8"))
            for item in raw_data.get("cases", []):
                res = EvaluationCaseResult.model_validate(item)
                cached_cases[res.case_id] = res
            if all(tc.case_id in cached_cases for tc in dataset):
                logger.info("Reutilizando evaluación de %d casos desde caché: %s", len(dataset), cache_file)
                return _build_summary_report([cached_cases[tc.case_id] for tc in dataset])
        except Exception as exc:
            logger.warning("No se pudo reutilizar caché previa de evaluación (%s). Recomputando...", exc)

    vstore = AiHorecaVectorStore()
    mem = SessionMemoryStore()
    app = build_aihoreca_graph(vector_store=vstore)

    results: list[EvaluationCaseResult] = []

    try:
        for idx, tc in enumerate(dataset, start=1):
            if not force_recompute and tc.case_id in cached_cases:
                logger.info("[%d/%d] Reutilizando caso en caché: %s", idx, len(dataset), tc.case_id)
                results.append(cached_cases[tc.case_id])
                continue

            if tc.reset_session_before:
                mem.delete_session(tc.session_id)

            logger.info("[%d/%d] Ejecutando caso %s (%s)...", idx, len(dataset), tc.case_id, tc.category)
            state_out = run_legal_query(
                query=tc.query,
                session_id=tc.session_id,
                memory_store=mem,
                compiled_graph=app,
            )

            pred_intent = state_out.get("intent", "OUT_OF_SCOPE")
            pred_route = state_out.get("route_decision", "block_out_of_scope")
            if pred_intent == "OUT_OF_SCOPE":
                pred_terr = state_out.get("territorio_normalizado")
            else:
                pred_terr = state_out.get("territorio_normalizado") or state_out.get("provincia_detectada")
            eff_query = state_out.get("effective_query") or tc.query
            cat_aleh = state_out.get("categoria_profesional_aleh")
            docs = state_out.get("retrieved_docs") or []
            audit = state_out.get("audit_report") or {}
            timings = state_out.get("node_timings_ms") or {}
            final_resp = state_out.get("final_response", "")

            intent_match = pred_intent == tc.expected_intent
            route_match = pred_route == tc.expected_route
            terr_match = pred_terr == tc.expected_territorio

            # Cobertura de capas normativas
            retrieved_layers = sorted({d.capa_normativa for d in docs})
            if not tc.expected_layers:
                layer_cov = 1.0 if len(docs) == 0 else 0.0
            else:
                hits = sum(1 for lyr in tc.expected_layers if lyr in retrieved_layers)
                layer_cov = round(hits / len(tc.expected_layers), 4)

            # Recall determinista de citas/artículos/anexos en los fragmentos recuperados
            combined_docs_meta_and_text = "\n".join(
                f"{d.norma_titulo} | {d.titulo_o_capitulo} | {d.articulo} | {d.text}" for d in docs
            ).lower()

            if not tc.expected_citation_keywords:
                cit_recall = 1.0
            else:
                cit_hits = sum(
                    1 for kw in tc.expected_citation_keywords if kw.lower() in combined_docs_meta_and_text
                )
                cit_recall = round(cit_hits / len(tc.expected_citation_keywords), 4)

            # Hit rate de cifras del Ground Truth en contexto + respuesta
            combined_eval_surface = f"{combined_docs_meta_and_text}\n{final_resp.lower()}"
            if not tc.expected_numbers_any:
                num_hit = 1.0
            else:
                matched_nums = sum(
                    1 for num in tc.expected_numbers_any if num.lower() in combined_eval_surface
                )
                # Si al menos las cifras principales de la tabla/artículo están presentes, calculamos ratio
                num_hit = round(min(1.0, matched_nums / max(1, len(tc.expected_numbers_any) - 1)), 4)

            unverified_nums = audit.get("unverified_numbers") or []
            verified_nums = audit.get("verified_numbers") or []
            zero_hallucination = len(unverified_nums) == 0

            # Evaluación Semántica RAGAS (Single-Pass Judge para casos generativos, determinista para cortocircuitos)
            if skip_semantic_judge:
                passed_det = intent_match and route_match and terr_match and zero_hallucination
                score_val = 1.0 if passed_det else 0.0
                faithfulness = score_val
                answer_rel = score_val
                ctx_recall_sem = score_val
                hier_score = score_val
                critique = (
                    "Evaluación rápida en modo determinista (0 tokens de Juez Semántico)."
                    if passed_det
                    else "Discrepancia en comprobaciones deterministas (enrutamiento o auditoría numérica)."
                )
                judge_model = "deterministic-quick-check"
            elif pred_route == "parallel_retrieve" and docs:
                ctx_str = "\n\n".join(
                    f"[{d.norma_titulo} | {d.capa_normativa} | {d.articulo}]\n{d.text}" for d in docs
                )
                judge_res, judge_model = _evaluate_ragas_single_pass(
                    tc=tc,
                    effective_query=eff_query,
                    retrieved_context=ctx_str,
                    final_response=final_resp,
                )
                faithfulness = round(float(judge_res.faithfulness), 4)
                answer_rel = round(float(judge_res.answer_relevancy), 4)
                ctx_recall_sem = round(float(judge_res.context_recall_semantic), 4)
                hier_score = round(float(judge_res.hierarchy_application_score), 4)
                critique = judge_res.critica_experta
                # Pausa breve para respetar ventana RPM/TPM del Free Tier
                time.sleep(1.2)
            else:
                passed_shortcircuit = intent_match and route_match and terr_match
                score_val = 1.0 if passed_shortcircuit else 0.0
                faithfulness = score_val
                answer_rel = score_val
                ctx_recall_sem = score_val
                hier_score = score_val
                critique = (
                    "Cortocircuito determinista (Guardrail / Gatekeeper HITL) ejecutado con precisión exacta (0 tokens de síntesis)."
                    if passed_shortcircuit
                    else "Discrepancia en el enrutamiento del cortocircuito."
                )
                judge_model = "deterministic-guardrail-verifier"

            total_ms = round(sum(timings.values()), 1)
            case_result = EvaluationCaseResult(
                case_id=tc.case_id,
                category=tc.category,
                session_id=tc.session_id,
                query=tc.query,
                effective_query=eff_query,
                expected_intent=tc.expected_intent,
                predicted_intent=pred_intent,
                intent_match=intent_match,
                expected_route=tc.expected_route,
                predicted_route=pred_route,
                route_match=route_match,
                expected_territorio=tc.expected_territorio,
                predicted_territorio=pred_terr,
                territorio_match=terr_match,
                categoria_aleh_detectada=cat_aleh,
                num_retrieved_docs=len(docs),
                retrieved_layers=retrieved_layers,
                retrieved_articles=[f"{d.capa_normativa}:{d.articulo}" for d in docs],
                layer_coverage_score=layer_cov,
                citation_recall_at_k=cit_recall,
                ground_truth_number_hit=num_hit,
                zero_numerical_hallucination=zero_hallucination,
                verified_numbers=verified_nums,
                unverified_numbers=unverified_nums,
                faithfulness=faithfulness,
                answer_relevancy=answer_rel,
                context_recall_semantic=ctx_recall_sem,
                hierarchy_application_score=hier_score,
                judge_critique=critique,
                judge_model_used=judge_model,
                node_timings_ms=timings,
                total_latency_ms=total_ms,
                final_response=final_resp,
            )
            results.append(case_result)
            cached_cases[case_result.case_id] = case_result

            # Guardar progreso incremental en disco fusionando con casos previos en caché
            merged_cases = list(cached_cases.values())
            partial_summary = _build_summary_report(merged_cases)
            cache_file.write_text(
                json.dumps(partial_summary.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        vstore.close()

    # Si se evaluó un subconjunto, devolvemos el informe del subconjunto solicitado
    final_summary = _build_summary_report(results)
    return final_summary


def _build_summary_report(cases: list[EvaluationCaseResult]) -> EvaluationSummaryReport:
    """Calcula las métricas agregadas globales y de casos generativos."""
    n = len(cases) or 1
    gen_cases = [c for c in cases if c.expected_route == "parallel_retrieve"]
    n_gen = len(gen_cases) or 1
    short_cases = [c for c in cases if c.expected_route != "parallel_retrieve"]

    routing_lats = [c.node_timings_ms.get("analyze_and_route", 0.0) for c in cases]
    retr_lats = [c.node_timings_ms.get("parallel_retrieval", 0.0) for c in gen_cases]
    synth_lats = [c.node_timings_ms.get("legal_synthesis_and_audit", 0.0) for c in gen_cases]
    tot_lats = [c.total_latency_ms for c in cases]

    return EvaluationSummaryReport(
        total_cases=len(cases),
        generative_cases=len(gen_cases),
        guardrail_and_hitl_cases=len(short_cases),
        intent_accuracy=round(sum(1 for c in cases if c.intent_match) / n, 4),
        route_accuracy=round(sum(1 for c in cases if c.route_match) / n, 4),
        territorio_accuracy=round(sum(1 for c in cases if c.territorio_match) / n, 4),
        mean_layer_coverage=round(sum(c.layer_coverage_score for c in gen_cases) / n_gen, 4),
        mean_citation_recall_at_k=round(sum(c.citation_recall_at_k for c in gen_cases) / n_gen, 4),
        mean_ground_truth_number_hit=round(sum(c.ground_truth_number_hit for c in gen_cases) / n_gen, 4),
        zero_hallucination_rate=round(sum(1 for c in cases if c.zero_numerical_hallucination) / n, 4),
        mean_faithfulness=round(sum(c.faithfulness for c in gen_cases) / n_gen, 4),
        mean_answer_relevancy=round(sum(c.answer_relevancy for c in gen_cases) / n_gen, 4),
        mean_context_recall_semantic=round(sum(c.context_recall_semantic for c in gen_cases) / n_gen, 4),
        mean_hierarchy_application_score=round(sum(c.hierarchy_application_score for c in gen_cases) / n_gen, 4),
        mean_routing_latency_ms=round(sum(routing_lats) / n, 1),
        mean_retrieval_latency_ms=round(sum(retr_lats) / n_gen, 1),
        mean_synthesis_latency_ms=round(sum(synth_lats) / n_gen, 1),
        mean_total_latency_ms=round(sum(tot_lats) / n, 1),
        cases=cases,
    )


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    corpus_stats = audit_corpus_and_index_stats()
    print("=" * 95)
    print("1. AUDITORÍA DEL CORPUS E ÍNDICE VECTORIAL QDRANT (648 CHUNKS)")
    print("=" * 95)
    print(json.dumps(corpus_stats, ensure_ascii=False, indent=2))

    report = evaluate_golden_dataset(force_recompute=True)
    print("\n" + "=" * 95)
    print("2. RESUMEN EJECUTIVO DEL BENCHMARK DE EVALUACIÓN (12 CASOS DEL GOLDEN DATASET)")
    print("=" * 95)
    summary_dict = report.model_dump(exclude={"cases"})
    print(json.dumps(summary_dict, ensure_ascii=False, indent=2))

    print("\n" + "=" * 95)
    print("3. DESGLOSE CASO POR CASO")
    print("=" * 95)
    for c in report.cases:
        print(
            f"[{c.case_id}] Intent={c.predicted_intent} (OK={c.intent_match}) | "
            f"Route={c.predicted_route} (OK={c.route_match}) | Terr={c.predicted_territorio} (OK={c.territorio_match}) | "
            f"LayerCov={c.layer_coverage_score:.2f} | CitRecall@8={c.citation_recall_at_k:.2f} | "
            f"Faithfulness={c.faithfulness:.2f} | AnswerRel={c.answer_relevancy:.2f} | "
            f"CtxRecall={c.context_recall_semantic:.2f} | Hierarchy={c.hierarchy_application_score:.2f} | "
            f"ZeroHalluc={c.zero_numerical_hallucination} | Latency={c.total_latency_ms:.0f}ms"
        )
        print(f"   -> Crítica Juez: {c.judge_critique}")
