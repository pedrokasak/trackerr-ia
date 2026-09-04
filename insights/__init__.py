"""
Insights com profundidade (TRA-56).

Ver `insights/service.py` para a pipeline: evidencia deterministica ->
confianca calculada -> prompt curto pedindo apenas texto -> parser que
rejeita numero inventado -> fallback deterministico.
"""

from .service import (
    InsightsService,
    InsightCandidate,
    NumberHallucinationError,
    build_deterministic_evidence,
    compute_confidence,
    parse_llm_insight_response,
    exposure_by_category,
)

__all__ = [
    "InsightsService",
    "InsightCandidate",
    "NumberHallucinationError",
    "build_deterministic_evidence",
    "compute_confidence",
    "parse_llm_insight_response",
    "exposure_by_category",
]
