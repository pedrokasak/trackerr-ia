"""
Testes unitários para o sistema de providers de LLM.
Usa mocks para evitar chamadas reais às APIs.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LLMFactory
# ---------------------------------------------------------------------------

class TestLLMFactory:
    def test_factory_returns_perplexity_by_default(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "perplexity", "PERPLEXITY_API_KEY": "fake-key"}):
            from benchmark.providers.factory import LLMFactory
            with patch("benchmark.providers.perplexity_provider.Perplexity"):
                provider = LLMFactory.get_provider()
                assert provider.provider_name == "perplexity"

    def test_factory_returns_claude(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "claude", "ANTHROPIC_API_KEY": "fake-key"}):
            from benchmark.providers.factory import LLMFactory
            with patch("benchmark.providers.claude_provider.anthropic.Anthropic"):
                provider = LLMFactory.get_provider()
                assert provider.provider_name == "claude"

    def test_factory_returns_groq(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "fake-key"}):
            from benchmark.providers.factory import LLMFactory
            with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: MagicMock() if name == "groq" else __import__(name, *args, **kwargs)):
                with patch("benchmark.providers.groq_provider.GroqProvider.__init__", return_value=None):
                    provider = LLMFactory.get_provider()
                    assert provider.provider_name == "groq"

    def test_factory_raises_on_unknown_provider(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}):
            from benchmark.providers.factory import LLMFactory
            with pytest.raises(ValueError, match="não suportado"):
                LLMFactory.get_provider()


# ---------------------------------------------------------------------------
# PerplexityProvider
# ---------------------------------------------------------------------------

class TestPerplexityProvider:
    @pytest.fixture
    def provider(self):
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "fake-key"}):
            with patch("benchmark.providers.perplexity_provider.Perplexity"):
                from benchmark.providers.perplexity_provider import PerplexityProvider
                return PerplexityProvider()

    def test_provider_name(self, provider):
        assert provider.provider_name == "perplexity"

    @pytest.mark.asyncio
    async def test_analyze_returns_parsed_json(self, provider):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"key": "value"}'
        provider._client.chat.completions.create = MagicMock(return_value=mock_response)

        result = await provider.analyze("prompt de teste")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_analyze_returns_raw_on_invalid_json(self, provider):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "resposta sem json"
        provider._client.chat.completions.create = MagicMock(return_value=mock_response)

        result = await provider.analyze("prompt de teste")
        assert "raw_response" in result

    def test_raises_on_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            from benchmark.providers.perplexity_provider import PerplexityProvider
            with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
                PerplexityProvider()


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------

class TestClaudeProvider:
    @pytest.fixture
    def provider(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("benchmark.providers.claude_provider.anthropic.Anthropic"):
                from benchmark.providers.claude_provider import ClaudeProvider
                return ClaudeProvider()

    def test_provider_name(self, provider):
        assert provider.provider_name == "claude"

    @pytest.mark.asyncio
    async def test_analyze_returns_parsed_json(self, provider):
        mock_response = MagicMock()
        mock_response.content[0].text = '{"portfolio_assessment": "bom"}'
        provider._client.messages.create = MagicMock(return_value=mock_response)

        result = await provider.analyze("prompt de teste")
        assert result == {"portfolio_assessment": "bom"}

    def test_raises_on_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            from benchmark.providers.claude_provider import ClaudeProvider
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ClaudeProvider()
