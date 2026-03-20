from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app


client = TestClient(app)


def test_chat_endpoint_returns_answer():
    with patch(
        "benchmark.benchmark.AIAnalysisService.chat_with_ai",
        new=AsyncMock(return_value={"answer": "Resposta baseada na carteira."}),
    ):
        response = client.post(
            "/api/chat",
            json={
                "question": "Minha carteira está arriscada?",
                "profile_plan": "pro",
                "context": {
                    "portfolioSummary": {"totalValue": 10000},
                    "assets": [{"symbol": "PETR4", "quantity": 10}],
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Resposta baseada na carteira."


def test_chat_endpoint_fallbacks_to_raw_response():
    with patch(
        "benchmark.benchmark.AIAnalysisService.chat_with_ai",
        new=AsyncMock(return_value={"raw_response": "Resposta sem JSON estruturado"}),
    ):
        response = client.post(
            "/api/chat",
            json={
                "question": "Quanto recebo de dividendos?",
                "profile_plan": "premium",
                "context": {},
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Resposta sem JSON estruturado"
