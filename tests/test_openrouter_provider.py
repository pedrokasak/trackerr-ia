import pytest
from unittest.mock import MagicMock, patch

from benchmark.providers.openrouter_provider import OpenRouterProvider


@pytest.fixture
def mock_openai_client():
    with patch("benchmark.providers.openrouter_provider.OpenAI") as mock:
        yield mock


@pytest.mark.asyncio
async def test_openrouter_provider_analyze(mock_openai_client):
    mock_instance = mock_openai_client.return_value
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content='{"portfolio_assessment": "Bom", "key_insights": []}'))
    ]
    mock_instance.chat.completions.create.return_value = mock_completion

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}, clear=False):
        provider = OpenRouterProvider()
        result = await provider.analyze("teste prompt")

        assert result["portfolio_assessment"] == "Bom"
        assert provider.provider_name == "openrouter"
        mock_instance.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_openrouter_provider_parse_error(mock_openai_client):
    mock_instance = mock_openai_client.return_value
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Texto sem json"))]
    mock_instance.chat.completions.create.return_value = mock_completion

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}, clear=False):
        provider = OpenRouterProvider()
        result = await provider.analyze("teste prompt")

        assert "raw_response" in result
        assert result["raw_response"] == "Texto sem json"


def test_openrouter_provider_levanta_erro_sem_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()


def test_openrouter_provider_usa_model_default(mock_openai_client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    provider = OpenRouterProvider()
    assert provider._model == OpenRouterProvider.DEFAULT_MODEL
    assert provider._model == "anthropic/claude-sonnet-4.5"


def test_openrouter_provider_respeita_openrouter_model_env(mock_openai_client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    provider = OpenRouterProvider()
    assert provider._model == "openai/gpt-5"


def test_openrouter_provider_aceita_model_explicito_no_construtor(
    mock_openai_client, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    # Parametro explicito vence a env var.
    provider = OpenRouterProvider(model="algum-outro-modelo")
    assert provider._model == "algum-outro-modelo"


def test_openrouter_provider_usa_base_url_correta_sem_headers_opcionais(
    mock_openai_client, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_NAME", raising=False)
    OpenRouterProvider()
    mock_openai_client.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="test_key",
        # OPENROUTER_APP_NAME tem default próprio ("Trackerr IA"), então o
        # header X-Title sempre existe mesmo sem a env var setada.
        default_headers={"X-Title": "Trackerr IA"},
    )


def test_openrouter_provider_inclui_referer_quando_site_url_configurada(
    mock_openai_client, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://trackerr.com.br")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "Trackerr Prod")
    OpenRouterProvider()
    mock_openai_client.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="test_key",
        default_headers={
            "HTTP-Referer": "https://trackerr.com.br",
            "X-Title": "Trackerr Prod",
        },
    )
