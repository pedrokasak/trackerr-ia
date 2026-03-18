"""
Provider Claude (Anthropic).
Usa a lib anthropic já presente no projeto.
"""

import json
import os
from typing import Any, Dict

import anthropic
from fastapi import HTTPException
from fastapi.logger import logger

from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    """Provider usando Claude (Anthropic)."""

    DEFAULT_MODEL = "claude-opus-4-5"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY não configurada no .env")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "claude"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama Claude e retorna JSON parseado."""
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            logger.info(f"[{self.provider_name}] Resposta recebida. Modelo: {self._model}")

            return self._parse_json(response_text)

        except anthropic.APIError as e:
            logger.error(f"[{self.provider_name}] Erro na API: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"[{self.provider_name}] Erro inesperado: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _parse_json(self, response_text: str) -> Dict[str, Any]:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": response_text}
