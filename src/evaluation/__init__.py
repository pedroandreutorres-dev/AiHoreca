"""Paquete de evaluación cuantitativa y auditoría empírica (Fase 4) para AiHoreca."""

from src.evaluation.evaluator import (
    EvaluationCaseResult,
    EvaluationSummaryReport,
    RAGASJudgeSchema,
    audit_corpus_and_index_stats,
    evaluate_golden_dataset,
)
from src.evaluation.golden_dataset import GoldenTestCase, get_golden_dataset

__all__ = [
    "EvaluationCaseResult",
    "EvaluationSummaryReport",
    "GoldenTestCase",
    "RAGASJudgeSchema",
    "audit_corpus_and_index_stats",
    "evaluate_golden_dataset",
    "get_golden_dataset",
]
