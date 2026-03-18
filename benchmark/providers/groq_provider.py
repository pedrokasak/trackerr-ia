"""
Provider Groq.
Requer GROQ_API_KEY no .env e dependência `groq` instalada.
"""

import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.logger import logger

from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Provider usando Groq (modelos rápidos e de baixo custo)."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY não configurada no .env")
        # Import lazy para evitar erro se a lib não estiver instalada
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Lib 'groq' não instalada. Execute: poetry add groq"
            )
        self._model = model

    @property
    def provider_name(self) -> str:
        return "groq"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama Groq e retorna JSON parseado."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            response_text = completion.choices[0].message.content
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
