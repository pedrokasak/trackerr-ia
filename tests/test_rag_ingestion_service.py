from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.ingestion_service import RagIngestionService, RagIngestItem
from rag.models import compute_content_hash
from rag.repository import MissingUserIdError


@pytest.fixture
def embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 768
    return provider


def make_repo(mock_repo_cls, existing_hashes=None):
    mock_repo = mock_repo_cls.return_value
    mock_repo.get_hashes_for_user = AsyncMock(return_value=existing_hashes or {})
    mock_repo.delete_by_source_ids = AsyncMock(return_value=0)
    mock_repo.add_chunks = AsyncMock()
    return mock_repo


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_grava_chunk_novo_com_metadata_e_as_of(mock_repo_cls, embedding_provider):
    mock_repo = make_repo(mock_repo_cls)

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    items = [
        RagIngestItem(
            source_type="portfolio_position",
            source_id="PETR4",
            content="PETR4 representa 15% da carteira.",
            metadata={"symbol": "PETR4", "sector": "Energia"},
            as_of=date(2026, 8, 20),
        )
    ]
    result = await service.ingest("user-1", items)

    created = mock_repo.add_chunks.call_args.args[0]
    assert len(created) == 1
    assert created[0].chunk_metadata == {"symbol": "PETR4", "sector": "Energia"}
    assert created[0].as_of == date(2026, 8, 20)
    assert created[0].content_hash == compute_content_hash(
        "PETR4 representa 15% da carteira."
    )
    assert result.chunks_created == 1
    assert result.chunks_unchanged == 0


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_chunk_inalterado_nao_paga_embedding(mock_repo_cls, embedding_provider):
    content = "PETR4 representa 15% da carteira."
    mock_repo = make_repo(mock_repo_cls, {"PETR4": compute_content_hash(content)})

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest(
        "user-1",
        [RagIngestItem(source_type="portfolio_position", source_id="PETR4", content=content)],
    )

    # O ponto inteiro de TRA-74: nada de embedding, nada de escrita.
    embedding_provider.embed.assert_not_called()
    mock_repo.add_chunks.assert_not_called()
    assert result.chunks_unchanged == 1
    assert result.chunks_created == 0


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_chunk_alterado_e_reembedado_e_substitui_o_antigo(
    mock_repo_cls, embedding_provider
):
    mock_repo = make_repo(
        mock_repo_cls, {"PETR4": compute_content_hash("PETR4 representa 15% da carteira.")}
    )

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest(
        "user-1",
        [
            RagIngestItem(
                source_type="portfolio_position",
                source_id="PETR4",
                content="PETR4 representa 22% da carteira.",
            )
        ],
    )

    embedding_provider.embed.assert_awaited_once_with("PETR4 representa 22% da carteira.")
    mock_repo.delete_by_source_ids.assert_awaited_once_with("user-1", ["PETR4"])
    assert result.chunks_created == 1
    assert result.chunks_unchanged == 0


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_ativo_que_sumiu_da_carteira_e_removido(mock_repo_cls, embedding_provider):
    content = "VALE3 representa 10% da carteira."
    mock_repo = make_repo(
        mock_repo_cls,
        {
            "VALE3": compute_content_hash(content),
            "PETR4": compute_content_hash("PETR4 representa 15% da carteira."),
        },
    )

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    # PETR4 nao vem mais no payload — foi vendido.
    await service.ingest(
        "user-1",
        [RagIngestItem(source_type="portfolio_position", source_id="VALE3", content=content)],
    )

    mock_repo.delete_by_source_ids.assert_awaited_once_with("user-1", ["PETR4"])
    embedding_provider.embed.assert_not_called()


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_chunk_sem_hash_antigo_e_reprocessado_uma_vez(
    mock_repo_cls, embedding_provider
):
    # Chunk gravado antes de TRA-74 nao tem content_hash; o repositorio nao
    # o devolve no mapa, entao ele conta como novo e e reprocessado.
    mock_repo = make_repo(mock_repo_cls, {})

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest(
        "user-1",
        [
            RagIngestItem(
                source_type="portfolio_position",
                source_id="PETR4",
                content="PETR4 representa 15% da carteira.",
            )
        ],
    )

    assert result.chunks_created == 1
    mock_repo.add_chunks.assert_awaited_once()


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_source_id_duplicado_gera_warning_e_grava_uma_vez(
    mock_repo_cls, embedding_provider
):
    mock_repo = make_repo(mock_repo_cls)

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest(
        "user-1",
        [
            RagIngestItem(source_type="portfolio_position", source_id="PETR4", content="A"),
            RagIngestItem(source_type="portfolio_position", source_id="PETR4", content="B"),
        ],
    )

    created = mock_repo.add_chunks.call_args.args[0]
    assert len(created) == 1
    assert created[0].content == "A"
    assert result.chunks_created == 1
    assert any("PETR4" in warning for warning in result.warnings)


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_ingest_ignora_itens_com_content_vazio(mock_repo_cls, embedding_provider):
    mock_repo = make_repo(mock_repo_cls)

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest(
        "user-1",
        [
            RagIngestItem(source_type="portfolio_position", source_id="PETR4", content="   "),
            RagIngestItem(
                source_type="portfolio_position",
                source_id="VALE3",
                content="VALE3: 10% da carteira",
            ),
        ],
    )

    created = mock_repo.add_chunks.call_args.args[0]
    assert len(created) == 1
    assert created[0].source_id == "VALE3"
    assert result.chunks_created == 1


@patch("rag.ingestion_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_payload_vazio_remove_tudo_que_existia(mock_repo_cls, embedding_provider):
    mock_repo = make_repo(mock_repo_cls, {"PETR4": "abc", "VALE3": "def"})
    mock_repo.delete_by_source_ids = AsyncMock(return_value=2)

    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    result = await service.ingest("user-1", [])

    assert sorted(mock_repo.delete_by_source_ids.call_args.args[1]) == ["PETR4", "VALE3"]
    assert result.chunks_deleted == 2
    assert result.chunks_created == 0


@pytest.mark.asyncio
async def test_ingest_rejeita_user_id_vazio(embedding_provider):
    service = RagIngestionService(session=MagicMock(), embedding_provider=embedding_provider)
    with pytest.raises(MissingUserIdError):
        await service.ingest("", [RagIngestItem(source_type="x", source_id="1", content="texto")])


def test_hash_e_estavel_e_sensivel_a_mudanca():
    assert compute_content_hash("PETR4: 15%") == compute_content_hash("PETR4: 15%")
    assert compute_content_hash("PETR4: 15%") != compute_content_hash("PETR4: 22%")
