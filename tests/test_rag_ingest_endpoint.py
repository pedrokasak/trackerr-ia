from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from rag.database import get_rag_session
from rag.ingestion_service import RagIngestResult


async def _fake_session():
    yield object()


app.dependency_overrides[get_rag_session] = _fake_session
client = TestClient(app)


def test_rag_ingest_endpoint_returns_counts():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagIngestionService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(
            return_value=RagIngestResult(chunks_deleted=2, chunks_created=3)
        )

        response = client.post(
            "/api/rag/ingest",
            json={
                "user_id": "user-1",
                "items": [
                    {"source_type": "portfolio_position", "source_id": "PETR4", "content": "PETR4: 15%"},
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_deleted"] == 2
    assert body["chunks_created"] == 3


def test_rag_ingest_endpoint_returns_422_on_missing_user_id():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagIngestionService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(
            side_effect=ValueError("user_id obrigatorio — ingestao nunca roda sem escopo de usuario.")
        )

        response = client.post(
            "/api/rag/ingest",
            json={"user_id": "", "items": []},
        )

    assert response.status_code == 422


def test_rag_ingest_endpoint_returns_500_on_provider_error():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagIngestionService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(side_effect=Exception("provider timeout"))

        response = client.post(
            "/api/rag/ingest",
            json={
                "user_id": "user-1",
                "items": [
                    {"source_type": "portfolio_position", "source_id": "PETR4", "content": "PETR4: 15%"},
                ],
            },
        )

    assert response.status_code == 500
