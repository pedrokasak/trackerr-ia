"""
Provider OpenRouter (API OpenAI-compatible).
Requer OPENROUTER_API_KEY no .env e dependência `openai` instalada.

OpenRouter (https://openrouter.ai/api/v1) é um roteador que expõe dezenas de
modelos de provedores diferentes (Anthropic, OpenAI, Google, Meta, etc.) sob
um único endpoint e formato — o mesmo Chat Completions do OpenAI, então usa
o SDK `openai` oficial só trocando `base_url`, igual ao NvidiaProvider.

Existe como via de acesso a modelos que os providers diretos deste projeto
não cobrem (ex.: GPT ou modelos Meta) sem integrar cada um separadamente, e
como caminho alternativo caso outro provider fique bloqueado (mesma situação
do TRA-71 com a Groq).
"""

import json
import os
from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.logger import logger
from openai import OpenAI

from .base import LLMProvider, ToolCall, ToolCallsResult, ToolSpec

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(LLMProvider):
    """Provider usando a API OpenAI-compatible do OpenRouter."""

    # Bom equilíbrio custo/qualidade para análise estruturada. Trocar via
    # OPENROUTER_MODEL no .env, sem editar código — ver catálogo completo em
    # https://openrouter.ai/models.
    DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY não configurada no .env")
        # Headers opcionais recomendados pela OpenRouter para identificar a
        # aplicação nos logs/ranking deles — não afetam a resposta.
        default_headers = {}
        referer = os.getenv("OPENROUTER_SITE_URL")
        if referer:
            default_headers["HTTP-Referer"] = referer
        app_title = os.getenv("OPENROUTER_APP_NAME", "Trackerr IA")
        if app_title:
            default_headers["X-Title"] = app_title

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers=default_headers or None,
        )
        self._model = model or os.getenv("OPENROUTER_MODEL") or self.DEFAULT_MODEL
        logger.info(
        	"[%s] Modelo resolvido: %s (origem: %s)",
        	self.provider_name,
        	self._model,
        	"argumento" if model else ("OPENROUTER_MODEL" if os.getenv("OPENROUTER_MODEL") else "padrao da classe"),
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """Chama a API da OpenRouter e retorna JSON parseado."""
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

    async def call_tools(
        self, system: str, prompt: str, tools: List[ToolSpec]
    ) -> ToolCallsResult:
        """
        Tool calling no formato Chat Completions (TRA-241), que a OpenRouter
        repassa ao modelo. É o provider de produção: sem este método, o
        roteador do chat nunca rodaria lá. Modelo sem suporte a tools volta
        erro da API, e a cadeia de fallback segue.
        """
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                    for tool in tools
                ],
                tool_choice="auto",
                temperature=0,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"[{self.provider_name}] Erro no tool calling: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        message = completion.choices[0].message
        calls: List[ToolCall] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if not function or not function.name:
                continue
            try:
                arguments = json.loads(function.arguments or "{}")
            except json.JSONDecodeError:
                # Argumento quebrado não vira chute: a chamada sai sem eles, e
                # o server decide se a ferramenta roda assim.
                arguments = {}
            calls.append(
                ToolCall(
                    name=function.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        usage = getattr(completion, "usage", None)
        return ToolCallsResult(
            calls=calls,
            provider=self.provider_name,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _parse_json(self, response_text: str) -> Dict[str, Any]:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": response_text}
