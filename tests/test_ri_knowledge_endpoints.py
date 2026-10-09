"""Endpoints do acervo de RI (TRA-264)."""

from datetime import date
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from rag.database import get_rag_session
from rag.models import RiDocumentChunk
from ri.knowledge_service import (
    RiAnswerItem,
    RiAnswerRejectedError,
    RiAnswerResult,
    RiIndexResult,
)


async def _fake_session():
    yield object()


app.dependency_overrides[get_rag_session] = _fake_session
client = TestClient(app)

INDEX_PAYLOAD = {
    "document": {
        "key": "cvm:1571942:1096648:1",
        "issuer": "PETR",
        "ticker": "PETR4",
        "company": "Petrobras",
        "title": "Fato Relevante - Aquisição",
        "category": "Fato Relevante",
        "published_at": "2026-09-28",
        "source_url": "https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?numProtocolo=1571942",
    },
    "content": "A Petrobras informa a aquisição de 30% do ativo X. -- 1 of 1 --",
}


def test_index_devolve_o_que_foi_feito():
    with patch("main.GeminiEmbeddingProvider"), patch(
        "main.RiDocumentIndexer"
    ) as indexer_cls:
        indexer_cls.return_value.index = AsyncMock(
            return_value=RiIndexResult(status="indexed", chunks=1)
        )

        response = client.post("/api/ri/index", json=INDEX_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "chunks": 1}
    meta, content = indexer_cls.return_value.index.call_args.args
    assert meta.key == "cvm:1571942:1096648:1"
    assert meta.published_at == date(2026, 9, 28)
    assert content.startswith("A Petrobras informa")


def test_index_recusa_documento_sem_data():
    payload = {**INDEX_PAYLOAD, "document": {**INDEX_PAYLOAD["document"]}}
    del payload["document"]["published_at"]

    response = client.post("/api/ri/index", json=payload)

    assert response.status_code == 422


def test_ask_devolve_afirmacoes_com_citacao():
    chunk = RiDocumentChunk(
        document_key="cvm:1:1:1",
        issuer="PETR",
        ticker="PETR4",
        company="Petrobras",
        title="ITR 2T26",
        category="Dados Econômico-Financeiros",
        period="2T26",
        published_at=date(2026, 8, 7),
        source_url="https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?numProtocolo=1",
        page=3,
        chunk_index=0,
        content="aprovou dividendos de R$ 0,45 por ação",
        embedding=[0.1] * 768,
        document_hash="h",
    )
    with patch("main.GeminiEmbeddingProvider"), patch("main.LLMFactory"), patch(
        "main.RiDocumentAnswerer"
    ) as answerer_cls:
        answerer_cls.return_value.ask = AsyncMock(
            return_value=RiAnswerResult(
                items=[
                    RiAnswerItem(
                        text="Dividendos de R$ 0,45 por ação aprovados.",
                        excerpt="aprovou dividendos de R$ 0,45 por ação",
                        page=3,
                        chunk=chunk,
                    )
                ],
                provider="gemini",
                dropped_reasons=["claim_evidence_not_found"],
            )
        )

        response = client.post(
            "/api/ri/ask",
            json={"issuer": "PETR", "question": "o que disse sobre dividendos?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": [
            {
                "text": "Dividendos de R$ 0,45 por ação aprovados.",
                "citation": {
                    "document_key": "cvm:1:1:1",
                    "title": "ITR 2T26",
                    "category": "Dados Econômico-Financeiros",
                    "period": "2T26",
                    "published_at": "2026-08-07",
                    "source_url": "https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?numProtocolo=1",
                    "page": 3,
                    "excerpt": "aprovou dividendos de R$ 0,45 por ação",
                },
            }
        ],
        "not_found": False,
        "provider": "gemini",
        "dropped_claims": 1,
    }


def test_ask_sem_base_nos_documentos_e_not_found():
    with patch("main.GeminiEmbeddingProvider"), patch("main.LLMFactory"), patch(
        "main.RiDocumentAnswerer"
    ) as answerer_cls:
        answerer_cls.return_value.ask = AsyncMock(return_value=RiAnswerResult())

        response = client.post(
            "/api/ri/ask", json={"issuer": "PETR", "question": "guidance?"}
        )

    assert response.status_code == 200
    assert response.json()["not_found"] is True
    assert response.json()["answer"] == []


def test_ask_barrada_pelo_guardrail_e_422_sem_texto_do_modelo():
    with patch("main.GeminiEmbeddingProvider"), patch("main.LLMFactory"), patch(
        "main.RiDocumentAnswerer"
    ) as answerer_cls:
        answerer_cls.return_value.ask = AsyncMock(
            side_effect=RiAnswerRejectedError("directed_recommendation")
        )

        response = client.post(
            "/api/ri/ask", json={"issuer": "PETR", "question": "devo comprar?"}
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "directed_recommendation"


def test_ask_recusa_pergunta_vazia():
    response = client.post("/api/ri/ask", json={"issuer": "PETR", "question": ""})

    assert response.status_code == 422
