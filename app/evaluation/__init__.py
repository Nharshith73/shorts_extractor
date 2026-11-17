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

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "STANDARD_CATEGORIES",
    "AnnotationRecord",
    "CandidateContext",
    "CandidateMatch",
    "EvaluationDataset",
    "EvaluationMetricsReport",
    "FrameSamplerConfig",
    "annotate_interval",
    "build_candidate_context",
    "compute_source_fingerprint",
    "evaluate_candidates",
    "export_evaluation_artifacts",
    "generate_sample_id",
]
