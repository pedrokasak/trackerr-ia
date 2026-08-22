from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from rag.shared_knowledge_service import SharedKnowledgeIngestResult

client = TestClient(app)


def test_knowledge_ingest_endpoint_reports_counts():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.SharedKnowledgeService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(
            return_value=SharedKnowledgeIngestResult(
                chunks_deleted=1, chunks_created=2, chunks_unchanged=3, warnings=[]
            )
        )
        response = client.post(
            "/api/rag/knowledge/ingest",
            json={
                "knowledge_base": "fiscal",
                "items": [
                    {
                        "source_id": "fiscal:acoes:isencao-20k",
                        "content": "Vendas ate 20k sao isentas.",
                        "version": "v1",
                    }
                ],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_created"] == 2
    assert body["chunks_unchanged"] == 3


def test_knowledge_ingest_endpoint_422_on_empty_base():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.SharedKnowledgeService"
    ) as mock_service_cls:
        mock_service_cls.return_value.ingest = AsyncMock(
            side_effect=ValueError("knowledge_base obrigatorio.")
        )
        response = client.post(
            "/api/rag/knowledge/ingest",
            json={"knowledge_base": "", "items": []},
        )
    assert response.status_code == 422
