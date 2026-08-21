import pytest
from unittest.mock import MagicMock, patch

from benchmark.providers.nvidia_provider import NvidiaProvider


@pytest.fixture
def mock_openai_client():
    with patch("benchmark.providers.nvidia_provider.OpenAI") as mock:
        yield mock


@pytest.mark.asyncio
async def test_nvidia_provider_analyze(mock_openai_client):
    mock_instance = mock_openai_client.return_value
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content='{"portfolio_assessment": "Bom", "key_insights": []}'))
    ]
    mock_instance.chat.completions.create.return_value = mock_completion

    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test_key"}, clear=False):
        provider = NvidiaProvider()
        result = await provider.analyze("teste prompt")

        assert result["portfolio_assessment"] == "Bom"
        assert provider.provider_name == "nvidia"
        mock_instance.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_nvidia_provider_parse_error(mock_openai_client):
    mock_instance = mock_openai_client.return_value
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Texto sem json"))]
    mock_instance.chat.completions.create.return_value = mock_completion

    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test_key"}, clear=False):
        provider = NvidiaProvider()
        result = await provider.analyze("teste prompt")

        assert "raw_response" in result
        assert result["raw_response"] == "Texto sem json"


def test_nvidia_provider_levanta_erro_sem_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NvidiaProvider()


def test_nvidia_provider_usa_model_default(mock_openai_client, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test_key")
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    provider = NvidiaProvider()
    assert provider._model == NvidiaProvider.DEFAULT_MODEL
    assert provider._model == "deepseek-ai/deepseek-v4-flash-0731"


def test_nvidia_provider_respeita_nvidia_model_env(mock_openai_client, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test_key")
    monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.2")
    provider = NvidiaProvider()
    assert provider._model == "z-ai/glm-5.2"


def test_nvidia_provider_aceita_model_explicito_no_construtor(
    mock_openai_client, monkeypatch
):
    monkeypatch.setenv("NVIDIA_API_KEY", "test_key")
    monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.2")
    # Parametro explicito vence a env var.
    provider = NvidiaProvider(model="algum-outro-modelo")
    assert provider._model == "algum-outro-modelo"


def test_nvidia_provider_usa_base_url_correta(mock_openai_client, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test_key")
    NvidiaProvider()
    mock_openai_client.assert_called_once_with(
        base_url="https://integrate.api.nvidia.com/v1", api_key="test_key"
    )
