from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from rag.database import get_rag_session
from rag.query_service import RagQueryResult


async def _fake_session():
    yield object()


app.dependency_overrides[get_rag_session] = _fake_session
client = TestClient(app)


def test_rag_query_endpoint_returns_answer():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagQueryService"
    ) as mock_service_cls:
        mock_service_cls.return_value.query = AsyncMock(
            return_value=RagQueryResult(
                answer="PETR4 representa 15% da carteira.\n\nAviso.",
                source="ai",
                chunk_count=2,
            )
        )

        response = client.post(
            "/api/rag/query",
            json={"user_id": "user-1", "question": "Quanto tenho em PETR4?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "PETR4 representa 15% da carteira.\n\nAviso."
    assert body["source"] == "ai"
    assert body["chunk_count"] == 2


def test_rag_query_endpoint_returns_422_on_empty_question():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagQueryService"
    ) as mock_service_cls:
        mock_service_cls.return_value.query = AsyncMock(
            side_effect=ValueError("question obrigatória.")
        )

        response = client.post(
            "/api/rag/query", json={"user_id": "user-1", "question": "  "}
        )

    assert response.status_code == 422


def test_rag_query_endpoint_returns_500_on_provider_error():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagQueryService"
    ) as mock_service_cls:
        mock_service_cls.return_value.query = AsyncMock(
            side_effect=Exception("provider timeout")
        )

        response = client.post(
            "/api/rag/query",
            json={"user_id": "user-1", "question": "Quanto tenho em PETR4?"},
        )

    assert response.status_code == 500
