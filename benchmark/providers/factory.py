"""
Factory de LLM Provider.
Lê a variável LLM_PROVIDER do .env e retorna a instância correta.

Valores aceitos para LLM_PROVIDER (e para LLM_PROVIDER_FALLBACK):
  - gemini (padrão)
  - claude
  - groq
  - nvidia
  - openrouter

LLM_PROVIDER_FALLBACK (opcional, TRA-203): se configurada, o provider
retornado tenta o primário e, em caso de falha (erro de API, timeout, chave
ausente/inválida), cai automaticamente para o fallback antes de propagar o
erro. Resolve o cenário do TRA-71 (Groq bloqueou o projeto inteiro a nível
de conta) sem precisar de redeploy — ver especificação em TRA-203.

Aceita uma LISTA separada por vírgula (TRA-250), tentada em ordem:
`LLM_PROVIDER_FALLBACK=gemini,groq,nvidia`. Um fallback só não bastou em
produção: o modelo gratuito do OpenRouter voltou 429 e o único fallback
(NVIDIA) deu 504, com as chaves do Gemini e do Groq configuradas e sem uso.
"""

import os
from fastapi.logger import logger

from .base import LLMProvider
from .fallback_provider import FallbackLLMProvider


class LLMFactory:
    """Factory que resolve qual provider de LLM usar com base no .env."""

    _SUPPORTED = ("gemini", "claude", "groq", "nvidia", "openrouter")

    @staticmethod
    def get_provider() -> LLMProvider:
        """
        Instancia e retorna o provider configurado em LLM_PROVIDER, com
        fallback automático para LLM_PROVIDER_FALLBACK quando configurada.

        Returns:
            LLMProvider: Instância do provider selecionado (ou um
            FallbackLLMProvider envolvendo os dois, se houver fallback).

        Raises:
            ValueError: Se LLM_PROVIDER ou LLM_PROVIDER_FALLBACK não forem
                valores suportados.
        """
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        primary = LLMFactory._build(provider_name)

        fallback_names = LLMFactory._fallback_chain(provider_name)
        if not fallback_names:
            logger.info(f"[LLMFactory] Usando provider: {provider_name}")
            return primary

        # Encadeia da última para a primeira: cada wrapper tenta o seu
        # provider e, se ele estiver indisponível, passa para o resto da
        # cadeia. `provider_name` continua sendo o do primário.
        chain: LLMProvider = LLMFactory._build(fallback_names[-1])
        for name in reversed(fallback_names[:-1]):
            chain = FallbackLLMProvider(LLMFactory._build(name), chain)

        logger.info(
            f"[LLMFactory] Usando provider: {provider_name} "
            f"(fallback: {', '.join(fallback_names)})"
        )
        return FallbackLLMProvider(primary, chain)

    @staticmethod
    def get_named_provider(provider_name: str) -> LLMProvider:
        """
        Um provider pelo nome, sem fallback (TRA-242): o juiz da avaliação
        precisa ser um modelo específico, diferente do gerador — cair para
        outro provider em silêncio mudaria quem dá a nota.
        """
        return LLMFactory._build(provider_name.lower().strip())

    @staticmethod
    def _fallback_chain(provider_name: str) -> list:
        """Nomes de fallback, em ordem, sem repetir nem incluir o primário."""
        raw = os.getenv("LLM_PROVIDER_FALLBACK", "")
        names = []
        for item in raw.split(","):
            name = item.lower().strip()
            if not name or name in names:
                continue
            if name == provider_name:
                # Fallback igual ao primário não protege contra nada — a mesma
                # conta/chave que falhou falharia de novo. Ignora em vez de
                # lançar: um .env mal configurado não pode derrubar o boot.
                logger.warning(
                    "[LLMFactory] LLM_PROVIDER_FALLBACK contém o próprio "
                    "LLM_PROVIDER ('%s') — ignorado, não protege contra nada.",
                    provider_name,
                )
                continue
            names.append(name)
        return names

    @staticmethod
    def _build(provider_name: str) -> LLMProvider:
        """Instancia o provider pelo nome, sem aplicar fallback nenhum."""
        if provider_name == "gemini":
            from .gemini_provider import GeminiProvider
            return GeminiProvider()

        if provider_name == "claude":
            from .claude_provider import ClaudeProvider
            return ClaudeProvider()

        if provider_name == "groq":
            from .groq_provider import GroqProvider
            return GroqProvider()

        if provider_name == "nvidia":
            from .nvidia_provider import NvidiaProvider
            return NvidiaProvider()

        if provider_name == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            return OpenRouterProvider()

        raise ValueError(
            f"LLM_PROVIDER='{provider_name}' não suportado. "
            f"Valores aceitos: {', '.join(LLMFactory._SUPPORTED)}"
        )
