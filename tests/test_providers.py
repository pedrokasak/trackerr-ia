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
    def test_factory_returns_gemini_by_default(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "fake-key"}):
            from benchmark.providers.factory import LLMFactory
            with patch("benchmark.providers.gemini_provider.genai.Client"):
                provider = LLMFactory.get_provider()
                assert provider.provider_name == "gemini"

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
# GeminiProvider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    @pytest.fixture
    def provider(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch("benchmark.providers.gemini_provider.genai.Client"):
                from benchmark.providers.gemini_provider import GeminiProvider
                return GeminiProvider()

    def test_provider_name(self, provider):
        assert provider.provider_name == "gemini"

    @pytest.mark.asyncio
    async def test_analyze_returns_parsed_json(self, provider):
        mock_response = MagicMock()
        mock_response.text = '{"key": "value"}'
        provider._client.models.generate_content = MagicMock(return_value=mock_response)

        result = await provider.analyze("prompt de teste")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_analyze_returns_raw_on_invalid_json(self, provider):
        mock_response = MagicMock()
        mock_response.text = "resposta sem json"
        provider._client.models.generate_content = MagicMock(return_value=mock_response)

        result = await provider.analyze("prompt de teste")
        assert "raw_response" in result

    def test_raises_on_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            from benchmark.providers.gemini_provider import GeminiProvider
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider()


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
