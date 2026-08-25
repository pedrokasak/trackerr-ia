"""
Provider NVIDIA NIM (API OpenAI-compatible).
Requer NVIDIA_API_KEY no .env e dependência `openai` instalada.

Existe como alternativa ao Groq: TRA-71 documenta que a conta Groq usada
neste projeto bloqueia todo modelo de chat a nível de projeto (403
model_permission_blocked_project), incluindo o default atual
(openai/gpt-oss-120b) — verificado contra a API real em 2026-08-21. Trocar
o modelo de novo não resolveria: o bloqueio é da conta, não do nome do
modelo, então esta é uma via alternativa até alguém liberar acesso no
console da Groq (ou em vez dela).

API da NVIDIA (https://integrate.api.nvidia.com/v1) segue o mesmo formato
de request/response do OpenAI Chat Completions, então usa o SDK `openai`
oficial só trocando `base_url` — não é integração própria.
"""

import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.logger import logger
from openai import OpenAI

from .base import LLMProvider

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider(LLMProvider):
    """Provider usando a API OpenAI-compatible da NVIDIA (NIM)."""

    # Rápido e barato — bom pra volume alto de chamadas de chat/análise.
    DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash-0731"
    # Alternativa maior, mais forte em raciocínio/instrução complexa —
    # trocar via NVIDIA_MODEL no .env, sem precisar editar código.
    ALT_MODEL = "z-ai/glm-5.2"

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY não configurada no .env")
        self._client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
        self._model = model or os.getenv("NVIDIA_MODEL") or self.DEFAULT_MODEL
        logger.info(
        	"[%s] Modelo resolvido: %s (origem: %s)",
        	self.provider_name,
        	self._model,
        	"argumento" if model else ("NVIDIA_MODEL" if os.getenv("NVIDIA_MODEL") else "padrao da classe"),
        )

    @property
    def provider_name(self) -> str:
        return "nvidia"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama a API da NVIDIA e retorna JSON parseado."""
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
