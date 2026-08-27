"""
Provider Gemini (Google).
Requer GEMINI_API_KEY no .env e dependência `google-genai` instalada.
"""

import asyncio
import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.logger import logger
from google import genai

from .base import LLMProvider

_RETRYABLE_CODES = (503, 429, 500)
_MAX_RETRIES = 3


class GeminiProvider(LLMProvider):
    """Provider usando Google Gemini."""

    # "gemini-2.5-flash" descontinuado pro Google — 404 NOT_FOUND
    # confirmado contra a API real, com a propria resposta de erro
    # apontando o substituto: "models/gemini-3.6-flash". Verificado
    # funcionando com chamada real antes de trocar aqui.
    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não configurada no .env")
        self._client = genai.Client(api_key=api_key)
        # Argumento explicito vence; depois GEMINI_MODEL; por ultimo o
        # padrao da classe. Trocar de modelo passa a ser variavel de
        # ambiente, sem alterar codigo.
        self._model = model or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        logger.info(
        	"[%s] Modelo resolvido: %s (origem: %s)",
        	self.provider_name,
        	self._model,
        	"argumento" if model else ("GEMINI_MODEL" if os.getenv("GEMINI_MODEL") else "padrao da classe"),
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama Gemini e retorna JSON parseado. Retry com backoff em 503/429."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                response_text = response.text
                logger.info(f"[{self.provider_name}] Resposta recebida. Modelo: {self._model}")
                return self._parse_json(response_text)

            except Exception as e:
                err_str = str(e)
                is_retryable = any(str(code) in err_str for code in _RETRYABLE_CODES)
                if is_retryable and attempt < _MAX_RETRIES:
                    wait = 2 ** (attempt - 1)  # 1s, 2s
                    logger.warning(
                        f"[{self.provider_name}] Erro transitório (tentativa {attempt}/{_MAX_RETRIES}), "
                        f"retry em {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
                    last_exc = e
                else:
                    logger.error(f"[{self.provider_name}] Erro: {e}")
                    raise HTTPException(status_code=500, detail=err_str)

        raise HTTPException(status_code=503, detail=str(last_exc))

    def _parse_json(self, response_text: str) -> Dict[str, Any]:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": response_text}
