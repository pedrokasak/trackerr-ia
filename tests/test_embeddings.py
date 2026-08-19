from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.embeddings import EMBEDDING_MODEL, GeminiEmbeddingProvider


def test_levanta_erro_sem_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiEmbeddingProvider()


@pytest.mark.asyncio
async def test_embed_devolve_os_768_valores(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    fake_embedding = MagicMock()
    fake_embedding.values = [0.1] * 768
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]

    with patch("rag.embeddings.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.aio.models.embed_content = AsyncMock(return_value=fake_response)

        provider = GeminiEmbeddingProvider()
        result = await provider.embed("Quanto tenho em PETR4?")

        assert len(result) == 768
        mock_client.aio.models.embed_content.assert_awaited_once_with(
            model=EMBEDDING_MODEL, contents=["Quanto tenho em PETR4?"]
        )


@pytest.mark.asyncio
async def test_embed_levanta_erro_em_dimensao_inesperada(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    fake_embedding = MagicMock()
    fake_embedding.values = [0.1] * 42  # dimensao errada de proposito
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]

    with patch("rag.embeddings.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.aio.models.embed_content = AsyncMock(return_value=fake_response)

        provider = GeminiEmbeddingProvider()
        with pytest.raises(RuntimeError, match="dimensão inesperada"):
            await provider.embed("texto qualquer")


@pytest.mark.asyncio
async def test_embed_levanta_erro_quando_gemini_nao_devolve_embedding(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    fake_response = MagicMock()
    fake_response.embeddings = None

    with patch("rag.embeddings.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.aio.models.embed_content = AsyncMock(return_value=fake_response)

        provider = GeminiEmbeddingProvider()
        with pytest.raises(RuntimeError):
            await provider.embed("texto qualquer")
