"""
Factory de LLM Provider.
Lê a variável LLM_PROVIDER do .env e retorna a instância correta.

Valores aceitos para LLM_PROVIDER:
  - perplexity (padrão)
  - claude
  - groq
"""

import os
from fastapi.logger import logger

from .base import LLMProvider


class LLMFactory:
    """Factory que resolve qual provider de LLM usar com base no .env."""

    _SUPPORTED = ("perplexity", "claude", "groq")

    @staticmethod
    def get_provider() -> LLMProvider:
        """
        Instancia e retorna o provider configurado em LLM_PROVIDER.

        Returns:
            LLMProvider: Instância do provider selecionado.

        Raises:
            ValueError: Se LLM_PROVIDER não for um valor suportado.
        """
        provider_name = os.getenv("LLM_PROVIDER", "perplexity").lower().strip()
        logger.info(f"[LLMFactory] Usando provider: {provider_name}")

        if provider_name == "perplexity":
            from .perplexity_provider import PerplexityProvider
            return PerplexityProvider()

        if provider_name == "claude":
            from .claude_provider import ClaudeProvider
            return ClaudeProvider()

        if provider_name == "groq":
            from .groq_provider import GroqProvider
            return GroqProvider()

        raise ValueError(
            f"LLM_PROVIDER='{provider_name}' não suportado. "
            f"Valores aceitos: {', '.join(LLMFactory._SUPPORTED)}"
        )
