"""
Provider Groq.
Requer GROQ_API_KEY no .env e dependência `groq` instalada.
"""

import asyncio
import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.logger import logger

from .base import LLMProvider

_RETRYABLE_CODES = (503, 429, 500)
_MAX_RETRIES = 3


class GroqProvider(LLMProvider):
    """Provider usando Groq (modelos rápidos e de baixo custo)."""

    # "llama-3.3-70b-versatile" foi descontinuado pela Groq e nem aparece
    # mais em client.models.list() — 404 model_not_found confirmado contra
    # a API real. Groq descontinuou a linha Llama; openai/gpt-oss-120b é o
    # modelo de maior capacidade atualmente listado, mais próximo do 70B
    # antigo em porte. Precisa estar habilitado no console da Groq
    # (Settings > Project > Limits) — contas novas vêm com a maioria dos
    # modelos bloqueados a nível de projeto por padrão, confirmado com 403
    # model_permission_blocked_project ao testar.
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, model: str | None = None) -> None:
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
        # Argumento explicito vence; depois GROQ_MODEL; por ultimo o
        # padrao da classe. Trocar de modelo passa a ser variavel de
        # ambiente, sem alterar codigo.
        self._model = model or os.getenv("GROQ_MODEL") or self.DEFAULT_MODEL
        logger.info(
        	"[%s] Modelo resolvido: %s (origem: %s)",
        	self.provider_name,
        	self._model,
        	"argumento" if model else ("GROQ_MODEL" if os.getenv("GROQ_MODEL") else "padrao da classe"),
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama Groq e retorna JSON parseado. Retry com backoff em 503/429."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
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
