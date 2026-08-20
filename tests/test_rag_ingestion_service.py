from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.ingestion_service import RagIngestionService, RagIngestItem
from rag.repository import MissingUserIdError


@pytest.fixture
def embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 768
    return provider


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_ingest_apaga_chunks_antigos_antes_de_gravar(mock_repo_cls, embedding_provider):
    mock_repo = mock_repo_cls.return_value
    mock_repo.delete_for_user = AsyncMock(return_value=3)
    mock_repo.add_chunks = AsyncMock()

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    items = [RagIngestItem(source_type="portfolio_position", source_id="PETR4", content="PETR4: 15% da carteira")]
    result = await service.ingest("user-1", items)

    mock_repo.delete_for_user.assert_awaited_once_with("user-1")
    mock_repo.add_chunks.assert_awaited_once()
    created_chunks = mock_repo.add_chunks.call_args.args[0]
    assert len(created_chunks) == 1
    assert created_chunks[0].user_id == "user-1"
    assert created_chunks[0].source_type == "portfolio_position"
    assert created_chunks[0].source_id == "PETR4"
    assert created_chunks[0].content == "PETR4: 15% da carteira"
    assert created_chunks[0].embedding == [0.1] * 768
    assert result.chunks_deleted == 3
    assert result.chunks_created == 1


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_ingest_com_lista_vazia_so_apaga_nao_grava_nada(mock_repo_cls, embedding_provider):
    mock_repo = mock_repo_cls.return_value
    mock_repo.delete_for_user = AsyncMock(return_value=5)
    mock_repo.add_chunks = AsyncMock()

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest("user-1", [])

    mock_repo.delete_for_user.assert_awaited_once_with("user-1")
    mock_repo.add_chunks.assert_not_called()
    embedding_provider.embed.assert_not_called()
    assert result.chunks_deleted == 5
    assert result.chunks_created == 0


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_ingest_ignora_itens_com_content_vazio(mock_repo_cls, embedding_provider):
    mock_repo = mock_repo_cls.return_value
    mock_repo.delete_for_user = AsyncMock(return_value=0)
    mock_repo.add_chunks = AsyncMock()

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    items = [
        RagIngestItem(source_type="portfolio_position", source_id="PETR4", content="   "),
        RagIngestItem(source_type="portfolio_position", source_id="VALE3", content="VALE3: 10% da carteira"),
    ]
    result = await service.ingest("user-1", items)

    created_chunks = mock_repo.add_chunks.call_args.args[0]
    assert len(created_chunks) == 1
    assert created_chunks[0].source_id == "VALE3"
    assert result.chunks_created == 1


@pytest.mark.asyncio
async def test_ingest_rejeita_user_id_vazio(embedding_provider):
    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    with pytest.raises(MissingUserIdError):
        await service.ingest("", [RagIngestItem(source_type="x", source_id="1", content="texto")])
