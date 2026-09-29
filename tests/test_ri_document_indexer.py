"""Indexacao de documento de RI no acervo (TRA-264)."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.models import compute_content_hash
from ri.knowledge_service import RiDocumentIndexer, RiDocumentMeta

META = RiDocumentMeta(
    key="cvm:1571942:1096648:1",
    issuer="petr",
    ticker="petr4",
    company="Petrobras",
    title="Fato Relevante - Aquisição",
    published_at=date(2026, 9, 28),
    source_url="https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?numProtocolo=1571942",
    category="Fato Relevante",
    period="2026",
)

CONTENT = (
    "A Petrobras informa a aquisição de 30% do ativo X por US$ 1,2 bilhão. -- 1 of 2 -- "
    "O pagamento será feito em duas parcelas iguais ao longo de 2027."
)


@pytest.fixture
def embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 768
    return provider


def make_repo(mock_repo_cls, existing_hash=None):
    repo = mock_repo_cls.return_value
    repo.get_document_hash = AsyncMock(return_value=existing_hash)
    repo.replace_document = AsyncMock()
    return repo


@patch("ri.knowledge_service.RiDocumentChunkRepository")
@pytest.mark.asyncio
async def test_indexa_por_pagina_com_metadados_do_documento(mock_repo_cls, embedding_provider):
    repo = make_repo(mock_repo_cls)
    indexer = RiDocumentIndexer(MagicMock(), embedding_provider)

    result = await indexer.index(META, CONTENT)

    key, chunks = repo.replace_document.call_args.args
    assert key == META.key
    assert result.status == "indexed" and result.chunks == 2
    assert [chunk.page for chunk in chunks] == [1, 2]
    first = chunks[0]
    assert first.issuer == "PETR" and first.ticker == "PETR4"
    assert first.source_url == META.source_url
    assert first.document_hash == compute_content_hash(CONTENT)
    # O texto guardado e so o do PDF; titulo e periodo vao so no embedding.
    assert first.content.startswith("A Petrobras informa")
    embedded = embedding_provider.embed.call_args_list[0].args[0]
    assert embedded.startswith("Fato Relevante - Aquisição (2026):")


@patch("ri.knowledge_service.RiDocumentChunkRepository")
@pytest.mark.asyncio
async def test_mesmo_documento_mesmo_texto_nao_faz_nada(mock_repo_cls, embedding_provider):
    repo = make_repo(mock_repo_cls, existing_hash=compute_content_hash(CONTENT))
    indexer = RiDocumentIndexer(MagicMock(), embedding_provider)

    result = await indexer.index(META, CONTENT)

    assert result.status == "unchanged"
    embedding_provider.embed.assert_not_called()
    repo.replace_document.assert_not_called()


@patch("ri.knowledge_service.RiDocumentChunkRepository")
@pytest.mark.asyncio
async def test_texto_novo_troca_os_chunks_do_documento(mock_repo_cls, embedding_provider):
    repo = make_repo(mock_repo_cls, existing_hash="hash-antigo")
    indexer = RiDocumentIndexer(MagicMock(), embedding_provider)

    await indexer.index(META, CONTENT)

    repo.replace_document.assert_awaited_once()


@patch("ri.knowledge_service.RiDocumentChunkRepository")
@pytest.mark.asyncio
async def test_documento_sem_texto_util_apaga_o_que_havia(mock_repo_cls, embedding_provider):
    repo = make_repo(mock_repo_cls, existing_hash="hash-antigo")
    indexer = RiDocumentIndexer(MagicMock(), embedding_provider)

    result = await indexer.index(META, "12 -- 1 of 1 --")

    assert result.status == "empty"
    repo.replace_document.assert_awaited_once_with(META.key, [])
