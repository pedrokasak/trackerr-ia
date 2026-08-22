from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.models import DocumentChunk
from rag.query_service import (
    DISCLAIMER,
    GUARD_REJECTED_ANSWER,
    NO_CONTEXT_ANSWER,
    RagQueryService,
)


def make_chunk(id_: int, content: str, as_of=None, source_type="portfolio_position"):
    # DocumentChunk real (nao MagicMock): o retrieval usa isinstance() pra
    # separar chunk pessoal de compartilhado no audit (TRA-87), entao o tipo
    # precisa ser o de verdade.
    chunk = DocumentChunk(
        id=id_,
        user_id="user-1",
        source_type=source_type,
        source_id=f"s{id_}",
        content=content,
        as_of=as_of,
    )
    return chunk


def personal(chunks):
    """Formato de search_with_distance: (chunk, distancia)."""
    return [(c, 0.1) for c in chunks]


@pytest.fixture
def embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 768
    return provider


@pytest.fixture
def llm_provider():
    return AsyncMock()


def build_service(mock_repo_cls, mock_shared_cls, embedding_provider, llm_provider,
                  personal_chunks=None, shared_chunks=None):
    mock_repo = mock_repo_cls.return_value
    mock_repo.search_with_distance = AsyncMock(
        return_value=personal(personal_chunks or [])
    )
    mock_shared = mock_shared_cls.return_value
    mock_shared.search_with_distance = AsyncMock(return_value=shared_chunks or [])
    return RagQueryService(
        session=MagicMock(),
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
    )


@patch("rag.query_service.SharedKnowledgeRepository")
@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_sem_chunks_nao_chama_llm_e_audita_no_context(
    mock_repo_cls, mock_audit_cls, mock_shared_cls, embedding_provider, llm_provider
):
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    service = build_service(mock_repo_cls, mock_shared_cls, embedding_provider, llm_provider)

    result = await service.query("user-1", "Quanto tenho em PETR4?")

    assert result.source == "no_context"
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.chunk_count == 0
    llm_provider.analyze.assert_not_called()
    mock_audit.record.assert_awaited_once_with(
        "user-1", "Quanto tenho em PETR4?", [], NO_CONTEXT_ANSWER, "no_context"
    )


@patch("rag.query_service.SharedKnowledgeRepository")
@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_resposta_valida_recebe_disclaimer_e_audita_ok(
    mock_repo_cls, mock_audit_cls, mock_shared_cls, embedding_provider, llm_provider
):
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    llm_provider.analyze.return_value = {"answer": "PETR4 representa 15% da sua carteira."}
    service = build_service(
        mock_repo_cls, mock_shared_cls, embedding_provider, llm_provider,
        personal_chunks=[make_chunk(1, "PETR4: 15% da carteira")],
    )

    result = await service.query("user-1", "Quanto tenho em PETR4?")

    assert result.source == "ai"
    assert "PETR4 representa 15%" in result.answer
    assert DISCLAIMER in result.answer
    assert result.chunk_count == 1
    args = mock_audit.record.call_args.args
    assert args[0] == "user-1"
    assert args[2] == [1]
    assert args[4] == "ok"
    assert mock_audit.record.call_args.kwargs.get("data_max_age_days") is None
    assert result.data_max_age_days is None


@patch("rag.query_service.SharedKnowledgeRepository")
@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_resposta_com_recomendacao_e_descartada_e_audita_motivo(
    mock_repo_cls, mock_audit_cls, mock_shared_cls, embedding_provider, llm_provider
):
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    llm_provider.analyze.return_value = {"answer": "Recomendo vender PETR4 agora."}
    service = build_service(
        mock_repo_cls, mock_shared_cls, embedding_provider, llm_provider,
        personal_chunks=[make_chunk(1, "PETR4: 15% da carteira")],
    )

    result = await service.query("user-1", "Devo vender PETR4?")

    assert result.source == "guard_rejected"
    assert result.answer == GUARD_REJECTED_ANSWER
    assert "Recomendo vender" not in result.answer
    mock_audit.record.assert_awaited_once_with(
        "user-1", "Devo vender PETR4?", [1], GUARD_REJECTED_ANSWER, "recommendation_language"
    )


@pytest.mark.asyncio
async def test_rejeita_pergunta_vazia(embedding_provider, llm_provider):
    service = RagQueryService(
        session=MagicMock(), embedding_provider=embedding_provider, llm_provider=llm_provider
    )
    with pytest.raises(ValueError):
        await service.query("user-1", "   ")


@patch("rag.query_service.SharedKnowledgeRepository")
@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_dado_velho_anota_resposta_e_registra_frescor(
    mock_repo_cls, mock_audit_cls, mock_shared_cls, embedding_provider, llm_provider
):
    from datetime import timedelta, datetime, timezone

    stale_date = datetime.now(timezone.utc).date() - timedelta(days=6)
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    llm_provider.analyze.return_value = {"answer": "PETR4 representa 15% da sua carteira."}
    service = build_service(
        mock_repo_cls, mock_shared_cls, embedding_provider, llm_provider,
        personal_chunks=[make_chunk(1, "PETR4: 15% da carteira", as_of=stale_date)],
    )

    result = await service.query("user-1", "Quanto tenho em PETR4?")

    assert "dias" in result.answer
    assert result.answer.index("dias") < result.answer.index("PETR4 representa")
    assert DISCLAIMER in result.answer
    assert result.data_max_age_days == 6
    assert mock_audit.record.call_args.kwargs.get("data_max_age_days") == 6


@patch("rag.query_service.SharedKnowledgeRepository")
@patch("rag.query_service.AuditLogRepository")
@patch("rag.query_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_mescla_chunk_compartilhado_com_pessoal(
    mock_repo_cls, mock_audit_cls, mock_shared_cls, embedding_provider, llm_provider
):
    # TRA-87: um chunk pessoal e um compartilhado, o compartilhado MAIS
    # proximo (distancia menor) — ambos entram no contexto, ordenados por
    # relevancia, e o audit registra so o id do pessoal.
    from rag.models import SharedKnowledgeChunk

    shared = SharedKnowledgeChunk(
        id=99, knowledge_base="fiscal", source_id="fiscal:x",
        content="Regra geral de exemplo.", content_hash="h",
    )
    mock_audit = mock_audit_cls.return_value
    mock_audit.record = AsyncMock()
    llm_provider.analyze.return_value = {"answer": "Resposta combinando os dois contextos."}

    mock_repo = mock_repo_cls.return_value
    mock_repo.search_with_distance = AsyncMock(
        return_value=[(make_chunk(1, "PETR4: 15% da carteira"), 0.4)]
    )
    mock_shared = mock_shared_cls.return_value
    mock_shared.search_with_distance = AsyncMock(return_value=[(shared, 0.1)])
    service = RagQueryService(
        session=MagicMock(), embedding_provider=embedding_provider, llm_provider=llm_provider
    )

    result = await service.query("user-1", "Como funciona a regra e minha carteira?")

    assert result.source == "ai"
    assert result.chunk_count == 2
    # audit registra apenas o chunk pessoal (id 1), nao o compartilhado (99):
    # ids de tabelas diferentes nao podem se misturar no mesmo array.
    assert mock_audit.record.call_args.args[2] == [1]
