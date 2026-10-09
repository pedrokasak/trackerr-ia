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
            with patch("benchmark.providers.groq_provider.GroqProvider.__init__", return_value=None):
                provider = LLMFactory.get_provider()
                assert provider.provider_name == "groq"

    def test_factory_returns_nvidia(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "nvidia", "NVIDIA_API_KEY": "fake-key"}):
            from benchmark.providers.factory import LLMFactory
            with patch("benchmark.providers.nvidia_provider.OpenAI"):
                provider = LLMFactory.get_provider()
                assert provider.provider_name == "nvidia"

    def test_factory_returns_openrouter(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "fake-key"}):
            from benchmark.providers.factory import LLMFactory
            with patch("benchmark.providers.openrouter_provider.OpenAI"):
                provider = LLMFactory.get_provider()
                assert provider.provider_name == "openrouter"

    # ------------------------------------------------------------------
    # LLM_PROVIDER_FALLBACK (TRA-203)
    # ------------------------------------------------------------------

    def test_factory_sem_fallback_devolve_provider_puro(self):
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "nvidia", "NVIDIA_API_KEY": "fake-key"},
            clear=True,
        ):
            from benchmark.providers.factory import LLMFactory
            from benchmark.providers.nvidia_provider import NvidiaProvider

            with patch("benchmark.providers.nvidia_provider.OpenAI"):
                provider = LLMFactory.get_provider()
                assert isinstance(provider, NvidiaProvider)

    def test_factory_com_fallback_devolve_wrapper(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "nvidia",
                "NVIDIA_API_KEY": "fake-key",
                "LLM_PROVIDER_FALLBACK": "openrouter",
                "OPENROUTER_API_KEY": "fake-key",
            },
            clear=True,
        ):
            from benchmark.providers.factory import LLMFactory
            from benchmark.providers.fallback_provider import FallbackLLMProvider

            with patch("benchmark.providers.nvidia_provider.OpenAI"), patch(
                "benchmark.providers.openrouter_provider.OpenAI"
            ):
                provider = LLMFactory.get_provider()
                assert isinstance(provider, FallbackLLMProvider)
                assert provider.provider_name == "nvidia"

    def test_factory_fallback_igual_ao_primario_e_ignorado(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "nvidia",
                "NVIDIA_API_KEY": "fake-key",
                "LLM_PROVIDER_FALLBACK": "nvidia",
            },
            clear=True,
        ):
            from benchmark.providers.factory import LLMFactory
            from benchmark.providers.nvidia_provider import NvidiaProvider

            with patch("benchmark.providers.nvidia_provider.OpenAI"):
                provider = LLMFactory.get_provider()
                # Sem wrapper: fallback igual ao primario nao protege nada.
                assert isinstance(provider, NvidiaProvider)

    def test_factory_fallback_invalido_levanta_erro_no_boot(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "nvidia",
                "NVIDIA_API_KEY": "fake-key",
                "LLM_PROVIDER_FALLBACK": "bing",
            },
            clear=True,
        ):
            from benchmark.providers.factory import LLMFactory

            with patch("benchmark.providers.nvidia_provider.OpenAI"):
                with pytest.raises(ValueError, match="não suportado"):
                    LLMFactory.get_provider()

    def test_factory_aceita_cadeia_de_fallbacks_em_ordem(self):
        # TRA-250: um fallback só não bastou — OpenRouter grátis 429 e NVIDIA 504.
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "fake-key",
                "LLM_PROVIDER_FALLBACK": "gemini, groq ,nvidia,gemini,openrouter",
                "GEMINI_API_KEY": "fake-key",
                "GROQ_API_KEY": "fake-key",
                "NVIDIA_API_KEY": "fake-key",
            },
            clear=True,
        ):
            from benchmark.providers.factory import LLMFactory

            assert LLMFactory._fallback_chain("openrouter") == [
                "gemini",
                "groq",
                "nvidia",
            ]

    @pytest.mark.asyncio
    async def test_cadeia_tenta_cada_fallback_ate_um_responder(self):
        from fastapi import HTTPException

        from benchmark.providers.factory import LLMFactory

        def fake(name, behavior):
            provider = MagicMock()
            provider.provider_name = name
            provider.analyze = AsyncMock(side_effect=behavior)
            return provider

        built = {
            "openrouter": fake("openrouter", HTTPException(status_code=500, detail="429 upstream")),
            "nvidia": fake("nvidia", HTTPException(status_code=500, detail="504")),
            "gemini": fake("gemini", [{"ok": True}]),
            "groq": fake("groq", [{"never": True}]),
        }
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "openrouter", "LLM_PROVIDER_FALLBACK": "nvidia,gemini,groq"},
            clear=True,
        ), patch.object(LLMFactory, "_build", side_effect=lambda name: built[name]):
            provider = LLMFactory.get_provider()
            result = await provider.analyze("prompt")

        assert result == {"ok": True}
        assert provider.provider_name == "openrouter"
        built["nvidia"].analyze.assert_awaited_once()
        built["groq"].analyze.assert_not_awaited()

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
