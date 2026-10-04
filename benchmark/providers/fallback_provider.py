"""
Provider de fallback (TRA-203).

Especificação: https://linear.app/tracker-invest/issue/TRA-203 — Fase 1 da
recomendação (não adotar orquestrador externo agora; resolver o cenário real
do TRA-71 com fallback simples dentro da própria factory).

TRA-71: a conta Groq usada neste projeto bloqueou TODO modelo de chat a
nível de projeto, e a única correção foi trocar `LLM_PROVIDER` no .env e
fazer redeploy. Este wrapper resolve isso automaticamente: se o provider
primário falhar de um jeito que sugere "provider fora do ar" (erro de API,
timeout, chave ausente/invalida), tenta uma vez no provider de fallback
antes de propagar o erro.

Não tenta cobrir erro de CONTEUDO (ex.: resposta que não é JSON válido) —
isso já vira `{"raw_response": ...}` dentro do próprio `analyze()` de cada
provider, nunca uma exceção. O fallback só existe para "o provider não
respondeu", não para "o provider respondeu algo inesperado".
"""

from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.logger import logger

from .base import LLMProvider, ToolCallingNotSupported, ToolCallsResult, ToolSpec

# ValueError: API key ausente/invalida (config errada = provider indisponivel
# tanto quanto um 500 seria). HTTPException: todo erro de rede/API dos
# providers atuais (ver claude/gemini/groq/nvidia/openrouter_provider.py) —
# nenhum deles diferencia o status code em subclasses proprias, entao a
# unica forma de identificar "provider indisponivel" hoje e por este tipo.
_UNAVAILABLE_ERRORS = (HTTPException, ValueError)


class FallbackLLMProvider(LLMProvider):
    """
    Encapsula um provider primário e um de fallback. `provider_name` sempre
    reflete o primário — o fallback é um detalhe interno, não uma escolha do
    operador visível fora deste wrapper.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def provider_name(self) -> str:
        return self._primary.provider_name

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            result = await self._primary.analyze(prompt)
            return result
        except _UNAVAILABLE_ERRORS as primary_error:
            logger.warning(
                "[LLMFactory] Provider primário '%s' indisponível (%s); "
                "tentando fallback '%s'.",
                self._primary.provider_name,
                primary_error,
                self._fallback.provider_name,
            )
            result = await self._fallback.analyze(prompt)
            logger.info(
                "[LLMFactory] Resposta veio do fallback '%s'.",
                self._fallback.provider_name,
            )
            return result

    async def call_tools(
        self, system: str, prompt: str, tools: List[ToolSpec]
    ) -> ToolCallsResult:
        """
        Mesma cadeia do `analyze` (TRA-241). Primário sem tool-calling nativo
        também passa para o próximo: o primeiro da cadeia que souber chamar
        ferramentas responde.
        """
        try:
            return await self._primary.call_tools(system, prompt, tools)
        except (*_UNAVAILABLE_ERRORS, ToolCallingNotSupported) as primary_error:
            logger.warning(
                "[LLMFactory] Provider '%s' sem tool calling agora (%s); "
                "tentando fallback '%s'.",
                self._primary.provider_name,
                type(primary_error).__name__,
                self._fallback.provider_name,
            )
            return await self._fallback.call_tools(system, prompt, tools)
