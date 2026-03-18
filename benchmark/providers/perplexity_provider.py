"""
Provider Perplexity (sonar-pro).
Migração do código original do benchmark.py.
"""

import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.logger import logger
from perplexity import Perplexity

from .base import LLMProvider


class PerplexityProvider(LLMProvider):
    """Provider usando Perplexity sonar-pro com busca web integrada."""

    def __init__(self) -> None:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY não configurada no .env")
        self._client = Perplexity(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "perplexity"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama Perplexity sonar-pro e retorna JSON parseado."""
        try:
            message = self._client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.choices[0].message.content
            logger.info(f"[{self.provider_name}] Resposta recebida.")

            return self._parse_json(response_text)

        except Exception as e:
            logger.error(f"[{self.provider_name}] Erro: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _parse_json(self, response_text: str) -> Dict[str, Any]:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": response_text}
