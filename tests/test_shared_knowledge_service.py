from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.shared_knowledge_service import (
    SharedKnowledgeService,
    SharedKnowledgeItem,
)
from rag.models import compute_content_hash


@pytest.fixture
def embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 768
    return provider


def make_repo(mock_repo_cls, existing=None):
    repo = mock_repo_cls.return_value
    repo.get_hashes = AsyncMock(return_value=existing or {})
    repo.delete_by_source_ids = AsyncMock(return_value=0)
    repo.add_chunks = AsyncMock()
    return repo


@patch("rag.shared_knowledge_service.SharedKnowledgeRepository")
@pytest.mark.asyncio
async def test_grava_chunk_novo(mock_repo_cls, embedding_provider):
    repo = make_repo(mock_repo_cls)
    service = SharedKnowledgeService(MagicMock(), embedding_provider)

    result = await service.ingest(
        "fiscal",
        [SharedKnowledgeItem("fiscal:acoes:isencao-20k", "Vendas ate 20k sao isentas.", version="v1")],
    )

    created = repo.add_chunks.call_args.args[0]
    assert len(created) == 1
    assert created[0].knowledge_base == "fiscal"
    assert created[0].source_id == "fiscal:acoes:isencao-20k"
    assert created[0].version == "v1"
    assert result.chunks_created == 1


@patch("rag.shared_knowledge_service.SharedKnowledgeRepository")
@pytest.mark.asyncio
async def test_chunk_inalterado_nao_paga_embedding(mock_repo_cls, embedding_provider):
    content = "Vendas ate 20k sao isentas."
    repo = make_repo(mock_repo_cls, {"fiscal:acoes:isencao-20k": compute_content_hash(content)})
    service = SharedKnowledgeService(MagicMock(), embedding_provider)

    result = await service.ingest(
        "fiscal", [SharedKnowledgeItem("fiscal:acoes:isencao-20k", content)]
    )

    embedding_provider.embed.assert_not_called()
    repo.add_chunks.assert_not_called()
    assert result.chunks_unchanged == 1
    assert result.chunks_created == 0


@patch("rag.shared_knowledge_service.SharedKnowledgeRepository")
@pytest.mark.asyncio
async def test_chunk_removido_da_base_sai_do_store(mock_repo_cls, embedding_provider):
    repo = make_repo(
        mock_repo_cls,
        {
            "fiscal:acoes:a": compute_content_hash("A"),
            "fiscal:acoes:b": compute_content_hash("B"),
        },
    )
    service = SharedKnowledgeService(MagicMock(), embedding_provider)

    # so 'a' vem no payload — 'b' foi removido da base curada
    await service.ingest("fiscal", [SharedKnowledgeItem("fiscal:acoes:a", "A")])

    repo.delete_by_source_ids.assert_awaited_once_with("fiscal", ["fiscal:acoes:b"])
    embedding_provider.embed.assert_not_called()


@patch("rag.shared_knowledge_service.SharedKnowledgeRepository")
@pytest.mark.asyncio
async def test_source_id_duplicado_gera_warning(mock_repo_cls, embedding_provider):
    repo = make_repo(mock_repo_cls)
    service = SharedKnowledgeService(MagicMock(), embedding_provider)

    result = await service.ingest(
        "fiscal",
        [
            SharedKnowledgeItem("fiscal:x", "A"),
            SharedKnowledgeItem("fiscal:x", "B"),
        ],
    )

    created = repo.add_chunks.call_args.args[0]
    assert len(created) == 1
    assert created[0].content == "A"
    assert any("fiscal:x" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_rejeita_knowledge_base_vazia(embedding_provider):
    service = SharedKnowledgeService(MagicMock(), embedding_provider)
    with pytest.raises(ValueError):
        await service.ingest("", [SharedKnowledgeItem("x", "y")])
