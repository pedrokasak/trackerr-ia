"""
Tool calling nativo nos providers (TRA-241): cada SDK devolve as chamadas
num formato, e todos viram `ToolCall(name, arguments)`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from benchmark.providers.base import (
    LLMProvider,
    ToolCall,
    ToolCallingNotSupported,
    ToolCallsResult,
    ToolSpec,
)
from benchmark.providers.fallback_provider import FallbackLLMProvider

TOOLS = [
    ToolSpec(
        name="portfolio_risk",
        description="Risco e concentração da carteira.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="asset_comparison",
        description="Compara ativos.",
        parameters={
            "type": "object",
            "properties": {"tickers": {"type": "array", "items": {"type": "string"}}},
        },
    ),
]


@pytest.mark.asyncio
async def test_claude_reads_tool_use_blocks():
    with patch("benchmark.providers.claude_provider.anthropic.Anthropic") as anthropic_cls, patch.dict(
        "os.environ", {"ANTHROPIC_API_KEY": "k"}
    ):
        from benchmark.providers.claude_provider import ClaudeProvider

        client = anthropic_cls.return_value
        client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="vou chamar"),
                SimpleNamespace(type="tool_use", name="asset_comparison", input={"tickers": ["PETR4", "VALE3"]}),
                SimpleNamespace(type="tool_use", name="portfolio_risk", input={}),
            ],
            usage=SimpleNamespace(input_tokens=320, output_tokens=40),
        )

        result = await ClaudeProvider().call_tools("sistema", "pergunta", TOOLS)

    assert result.calls == [
        ToolCall("asset_comparison", {"tickers": ["PETR4", "VALE3"]}),
        ToolCall("portfolio_risk", {}),
    ]
    assert (result.input_tokens, result.output_tokens) == (320, 40)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "sistema"
    assert kwargs["tool_choice"] == {"type": "auto"}
    assert kwargs["tools"][1]["input_schema"] == TOOLS[1].parameters


@pytest.mark.asyncio
async def test_gemini_reads_function_calls():
    with patch("benchmark.providers.gemini_provider.genai.Client") as client_cls, patch.dict(
        "os.environ", {"GEMINI_API_KEY": "k"}
    ):
        from benchmark.providers.gemini_provider import GeminiProvider

        client = client_cls.return_value
        client.models.generate_content.return_value = SimpleNamespace(
            function_calls=[SimpleNamespace(name="portfolio_risk", args={})],
            usage_metadata=SimpleNamespace(prompt_token_count=210, candidates_token_count=12),
        )

        result = await GeminiProvider().call_tools("sistema", "pergunta", TOOLS)

    assert result.calls == [ToolCall("portfolio_risk", {})]
    assert result.provider == "gemini"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.system_instruction == "sistema"
    declarations = config.tools[0].function_declarations
    assert [d.name for d in declarations] == ["portfolio_risk", "asset_comparison"]
    assert config.tool_config.function_calling_config.mode == "AUTO"


@pytest.mark.asyncio
async def test_gemini_without_function_calls_returns_none():
    with patch("benchmark.providers.gemini_provider.genai.Client") as client_cls, patch.dict(
        "os.environ", {"GEMINI_API_KEY": "k"}
    ):
        from benchmark.providers.gemini_provider import GeminiProvider

        client_cls.return_value.models.generate_content.return_value = SimpleNamespace(
            function_calls=None, usage_metadata=None
        )

        result = await GeminiProvider().call_tools("s", "oi", TOOLS)

    assert result.calls == []


def _openrouter_completion(tool_calls):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=30),
    )


@pytest.mark.asyncio
async def test_openrouter_reads_tool_calls_and_drops_broken_arguments():
    with patch("benchmark.providers.openrouter_provider.OpenAI") as openai_cls, patch.dict(
        "os.environ", {"OPENROUTER_API_KEY": "k"}
    ):
        from benchmark.providers.openrouter_provider import OpenRouterProvider

        client = openai_cls.return_value
        client.chat.completions.create.return_value = _openrouter_completion(
            [
                SimpleNamespace(function=SimpleNamespace(name="asset_comparison", arguments='{"tickers": ["ITUB4"]}')),
                SimpleNamespace(function=SimpleNamespace(name="portfolio_risk", arguments="{quebrado")),
            ]
        )

        result = await OpenRouterProvider().call_tools("sistema", "pergunta", TOOLS)

    assert result.calls == [
        ToolCall("asset_comparison", {"tickers": ["ITUB4"]}),
        ToolCall("portfolio_risk", {}),
    ]
    assert (result.input_tokens, result.output_tokens) == (150, 30)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "sistema"}
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openrouter_api_error_becomes_http_exception():
    with patch("benchmark.providers.openrouter_provider.OpenAI") as openai_cls, patch.dict(
        "os.environ", {"OPENROUTER_API_KEY": "k"}
    ):
        from benchmark.providers.openrouter_provider import OpenRouterProvider

        openai_cls.return_value.chat.completions.create.side_effect = RuntimeError(
            "No endpoints found that support tool use"
        )

        with pytest.raises(HTTPException):
            await OpenRouterProvider().call_tools("s", "p", TOOLS)


class _NoTools(LLMProvider):
    async def analyze(self, prompt):
        return {}

    @property
    def provider_name(self):
        return "sem-tools"


class _WithTools(_NoTools):
    async def call_tools(self, system, prompt, tools):
        return ToolCallsResult(calls=[ToolCall("portfolio_risk", {})], provider="com-tools")

    @property
    def provider_name(self):
        return "com-tools"


class _Down(_NoTools):
    async def call_tools(self, system, prompt, tools):
        raise HTTPException(status_code=500, detail="fora")


@pytest.mark.asyncio
async def test_provider_without_native_tools_says_so():
    with pytest.raises(ToolCallingNotSupported):
        await _NoTools().call_tools("s", "p", TOOLS)


@pytest.mark.asyncio
async def test_fallback_skips_a_provider_without_tool_calling():
    result = await FallbackLLMProvider(_NoTools(), _WithTools()).call_tools("s", "p", TOOLS)

    assert result.provider == "com-tools"


@pytest.mark.asyncio
async def test_fallback_skips_a_provider_that_is_down():
    result = await FallbackLLMProvider(_Down(), _WithTools()).call_tools("s", "p", TOOLS)

    assert result.calls == [ToolCall("portfolio_risk", {})]


@pytest.mark.asyncio
async def test_whole_chain_without_tool_calling_propagates():
    with pytest.raises(ToolCallingNotSupported):
        await FallbackLLMProvider(_NoTools(), _NoTools()).call_tools("s", "p", TOOLS)
