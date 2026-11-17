"""Stage 4A: Gameplay Intelligence Foundation and Evaluation System."""

from .dataset import DEFAULT_MANIFEST_PATH, EvaluationDataset, compute_source_fingerprint, generate_sample_id
from .models import (
    STANDARD_CATEGORIES,
    AnnotationRecord,
    CandidateContext,
    CandidateMatch,
    EvaluationMetricsReport,
)
from .annotate import annotate_interval
from .context import FrameSamplerConfig, build_candidate_context
from .evaluate import evaluate_candidates, export_evaluation_artifacts
from .judge import judge_candidates
from .providers.qwen import QwenProviderConfig, QwenVisionLanguageProvider
from .schema import AIJudgmentRecord, EvidenceItem

__all__ = [
    "AIJudgmentRecord",
    "AnnotationRecord",
    "CandidateContext",
    "CandidateMatch",
    "DEFAULT_MANIFEST_PATH",
    "EvaluationDataset",
    "EvaluationMetricsReport",
    "EvidenceItem",
    "FrameSamplerConfig",
    "QwenProviderConfig",
    "QwenVisionLanguageProvider",
    "STANDARD_CATEGORIES",
    "annotate_interval",
    "build_candidate_context",
    "compute_source_fingerprint",
    "evaluate_candidates",
    "export_evaluation_artifacts",
    "generate_sample_id",
    "judge_candidates",
]
