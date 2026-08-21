from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from rag.erasure_service import RagErasureResult

client = TestClient(app)


def test_rag_erase_endpoint_apaga_e_reporta_contagens():
    with patch("main.RagErasureService") as mock_service_cls:
        mock_service_cls.return_value.erase = AsyncMock(
            return_value=RagErasureResult(chunks_deleted=12, audit_rows_anonymized=4)
        )
        response = client.post("/api/rag/erase", json={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.json() == {"chunks_deleted": 12, "audit_rows_anonymized": 4}


def test_rag_erase_endpoint_e_idempotente():
    with patch("main.RagErasureService") as mock_service_cls:
        mock_service_cls.return_value.erase = AsyncMock(
            return_value=RagErasureResult(chunks_deleted=0, audit_rows_anonymized=0)
        )
        response = client.post("/api/rag/erase", json={"user_id": "ja-apagado"})

    assert response.status_code == 200
    assert response.json()["chunks_deleted"] == 0


def test_rag_erase_endpoint_422_sem_user_id():
    with patch("main.RagErasureService") as mock_service_cls:
        mock_service_cls.return_value.erase = AsyncMock(
            side_effect=ValueError("user_id obrigatorio")
        )
        response = client.post("/api/rag/erase", json={"user_id": ""})

    assert response.status_code == 422


def test_health_reporta_provider_da_config_nao_hardcoded(monkeypatch):
    # Regressao: a versao anterior devolvia "Groq/Llama-3.3" fixo, que virou
    # mentira quando o provider mudou.
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    body = client.get("/api/health").json()
    assert body["llm_provider"] == "nvidia"
    assert "Llama" not in str(body)
