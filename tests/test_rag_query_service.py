from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.query_service import (
    DISCLAIMER,
    GUARD_REJECTED_ANSWER,
    NO_CONTEXT_ANSWER,
    RagQueryService,
)


def make_chunk(id_: int, content: str):
    chunk = MagicMock()
    chunk.id = id_
    chunk.content = content
    return chunk


@pytest.fixture
def embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 768
    return provider


@pytest.fixture
def llm_provider():
    return AsyncMock()


@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_sem_chunks_nao_chama_llm_e_audita_no_context(
    mock_repo_cls, mock_audit_cls, embedding_provider, llm_provider
):
    mock_repo = mock_repo_cls.return_value
    mock_repo.search = AsyncMock(return_value=[])
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()

    service = RagQueryService(session=MagicMock(), embedding_provider=embedding_provider, llm_provider=llm_provider)
    result = await service.query("user-1", "Quanto tenho em PETR4?")

    assert result.source == "no_context"
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.chunk_count == 0
    llm_provider.analyze.assert_not_called()
    mock_audit.record.assert_awaited_once_with(
        "user-1", "Quanto tenho em PETR4?", [], NO_CONTEXT_ANSWER, "no_context"
    )


@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_resposta_valida_recebe_disclaimer_e_audita_ok(
    mock_repo_cls, mock_audit_cls, embedding_provider, llm_provider
):
    mock_repo = mock_repo_cls.return_value
    mock_repo.search = AsyncMock(
        return_value=[make_chunk(1, "PETR4: 15% da carteira")]
    )
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    llm_provider.analyze.return_value = {"answer": "PETR4 representa 15% da sua carteira."}

    service = RagQueryService(session=MagicMock(), embedding_provider=embedding_provider, llm_provider=llm_provider)
    result = await service.query("user-1", "Quanto tenho em PETR4?")

    assert result.source == "ai"
    assert "PETR4 representa 15%" in result.answer
    assert DISCLAIMER in result.answer
    assert result.chunk_count == 1
    mock_audit.record.assert_awaited_once()
    args = mock_audit.record.call_args.args
    assert args[0] == "user-1"
    assert args[2] == [1]
    assert args[4] == "ok"


@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_resposta_com_recomendacao_e_descartada_e_audita_motivo(
    mock_repo_cls, mock_audit_cls, embedding_provider, llm_provider
):
    mock_repo = mock_repo_cls.return_value
    mock_repo.search = AsyncMock(return_value=[make_chunk(1, "PETR4: 15% da carteira")])
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    llm_provider.analyze.return_value = {"answer": "Recomendo vender PETR4 agora."}

    service = RagQueryService(session=MagicMock(), embedding_provider=embedding_provider, llm_provider=llm_provider)
    result = await service.query("user-1", "Devo vender PETR4?")

    assert result.source == "guard_rejected"
    assert result.answer == GUARD_REJECTED_ANSWER
    assert "Recomendo vender" not in result.answer
    mock_audit.record.assert_awaited_once_with(
        "user-1", "Devo vender PETR4?", [1], GUARD_REJECTED_ANSWER, "recommendation_language"
    )


@pytest.mark.asyncio
async def test_rejeita_pergunta_vazia(embedding_provider, llm_provider):
    service = RagQueryService(session=MagicMock(), embedding_provider=embedding_provider, llm_provider=llm_provider)
    with pytest.raises(ValueError):
        await service.query("user-1", "   ")
