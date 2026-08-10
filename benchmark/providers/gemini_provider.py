"""
Provider Gemini (Google).
Requer GEMINI_API_KEY no .env e dependência `google-genai` instalada.
"""

import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.logger import logger
from google import genai

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Provider usando Google Gemini."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não configurada no .env")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama Gemini e retorna JSON parseado."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            response_text = response.text
            logger.info(f"[{self.provider_name}] Resposta recebida. Modelo: {self._model}")

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
