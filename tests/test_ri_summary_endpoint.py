from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app
from ri.summary_service import RiSummaryRejectedError, RiSummaryResult


client = TestClient(app)

VALID_PAYLOAD = {
    "document": {
        "ticker": "PETR4",
        "company": "Petrobras",
        "document_type": "earnings_release",
        "period": "2T26",
        "published_at": "2026-08-07T00:00:00Z",
    },
    "content": "Receita líquida cresceu 12% no trimestre. " * 20,
    "structured_signals": {
        "revenue": {"detected": True, "direction": "up", "evidence": ["receita"]}
    },
}


def test_ri_summarize_returns_highlights_and_narrative():
    with patch(
        "ri.summary_service.RiSummaryService.summarize",
        new=AsyncMock(
            return_value=RiSummaryResult(
                highlights=["Receita cresceu 12%."],
                narrative="Trimestre de crescimento de receita.",
                provider="gemini",
            )
        ),
    ):
        response = client.post("/api/ri/summarize", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {
        "highlights": ["Receita cresceu 12%."],
        "narrative": "Trimestre de crescimento de receita.",
        "provider": "gemini",
    }


def test_ri_summarize_returns_422_when_guard_rejects():
    with patch(
        "ri.summary_service.RiSummaryService.summarize",
        new=AsyncMock(side_effect=RiSummaryRejectedError("directed_recommendation")),
    ):
        response = client.post("/api/ri/summarize", json=VALID_PAYLOAD)

    assert response.status_code == 422
    assert response.json()["detail"] == "directed_recommendation"


def test_ri_summarize_returns_500_on_provider_error():
    with patch(
        "ri.summary_service.RiSummaryService.summarize",
        new=AsyncMock(side_effect=Exception("provider timeout")),
    ):
        response = client.post("/api/ri/summarize", json=VALID_PAYLOAD)

    assert response.status_code == 500


def test_ri_summarize_rejects_missing_content():
    payload = {**VALID_PAYLOAD, "content": ""}
    response = client.post("/api/ri/summarize", json=payload)
    assert response.status_code == 422


def test_ri_summarize_rejects_oversized_content():
    payload = {**VALID_PAYLOAD, "content": "a" * 400_001}
    response = client.post("/api/ri/summarize", json=payload)
    assert response.status_code == 422
