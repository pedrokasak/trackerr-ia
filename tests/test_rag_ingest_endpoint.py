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
            return_value=RagIngestResult(
                chunks_deleted=2,
                chunks_created=3,
                chunks_unchanged=27,
                warnings=[],
            )
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
    assert body["chunks_unchanged"] == 27


def test_rag_ingest_endpoint_accepts_metadata_and_as_of():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagIngestionService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(
            return_value=RagIngestResult(chunks_deleted=0, chunks_created=1)
        )

        response = client.post(
            "/api/rag/ingest",
            json={
                "user_id": "user-1",
                "items": [
                    {
                        "source_type": "portfolio_position",
                        "source_id": "PETR4",
                        "content": "PETR4 representa 15% da carteira.",
                        "metadata": {"symbol": "PETR4", "sector": "Energia"},
                        "as_of": "2026-08-20",
                    }
                ],
            },
        )

    assert response.status_code == 200
    forwarded = mock_service_cls.return_value.ingest.call_args.args[1][0]
    assert forwarded.metadata == {"symbol": "PETR4", "sector": "Energia"}
    assert forwarded.as_of.isoformat() == "2026-08-20"


def test_rag_ingest_endpoint_still_accepts_payload_without_metadata():
    """Compatibilidade com o payload de TRA-72, que nao tem metadata/as_of."""
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RagIngestionService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(
            return_value=RagIngestResult(chunks_deleted=0, chunks_created=1)
        )

        response = client.post(
            "/api/rag/ingest",
            json={
                "user_id": "user-1",
                "items": [
                    {
                        "source_type": "portfolio_position",
                        "source_id": "PETR4",
                        "content": "PETR4 representa 15% da carteira.",
                    }
                ],
            },
        )

    assert response.status_code == 200
    forwarded = mock_service_cls.return_value.ingest.call_args.args[1][0]
    assert forwarded.metadata is None
    assert forwarded.as_of is None


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
