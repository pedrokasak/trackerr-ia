"""Roteador do chat (TRA-241): o modelo escolhe, a saída é conferida."""

import pytest

from benchmark.providers.base import (
    LLMProvider,
    ToolCall,
    ToolCallingNotSupported,
    ToolCallsResult,
    ToolSpec,
)
from chat.tool_planner import MAX_TOOL_CALLS, SingleStepToolRuntime, sanitize_calls

TICKERS_SCHEMA = {
    "type": "object",
    "properties": {
        "tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
}

TOOLS = [
    ToolSpec("portfolio_risk", "Risco da carteira.", {"type": "object", "properties": {}}),
    ToolSpec("asset_comparison", "Compara ativos.", TICKERS_SCHEMA),
    ToolSpec("dividends_received", "Proventos recebidos.", {"type": "object", "properties": {}}),
    ToolSpec("ri_question", "Pergunta aos documentos de RI.", TICKERS_SCHEMA),
]


class _FakeProvider(LLMProvider):
    def __init__(self, calls=None, error=None):
        self.calls = calls or []
        self.error = error
        self.received = None

    async def analyze(self, prompt):
        return {}

    async def call_tools(self, system, prompt, tools):
        self.received = (system, prompt, tools)
        if self.error:
            raise self.error
        return ToolCallsResult(calls=self.calls, provider="fake", input_tokens=90, output_tokens=10)

    @property
    def provider_name(self):
        return "fake"


def test_drops_tools_outside_the_catalog():
    calls = sanitize_calls([ToolCall("delete_user", {}), ToolCall("portfolio_risk", {})], TOOLS, 3)

    assert calls == [ToolCall("portfolio_risk", {})]


def test_keeps_only_declared_arguments_of_the_declared_type():
    calls = sanitize_calls(
        [
            ToolCall(
                "asset_comparison",
                {"tickers": [" PETR4 ", 42, "VALE3", "ITUB4"], "senha": "x"},
            )
        ],
        TOOLS,
        3,
    )

    # maxItems do schema: 2. Número não é ticker. Argumento extra sai.
    assert calls == [ToolCall("asset_comparison", {"tickers": ["PETR4", "VALE3"]})]


def test_tool_without_parameters_gets_no_arguments():
    calls = sanitize_calls([ToolCall("portfolio_risk", {"tickers": ["PETR4"]})], TOOLS, 3)

    assert calls == [ToolCall("portfolio_risk", {})]


def test_caps_the_calls_and_counts_repeats_once():
    calls = sanitize_calls(
        [
            ToolCall("portfolio_risk", {}),
            ToolCall("portfolio_risk", {}),
            ToolCall("asset_comparison", {"tickers": ["PETR4"]}),
            ToolCall("dividends_received", {}),
            ToolCall("ri_question", {"tickers": ["PETR4"]}),
        ],
        TOOLS,
        MAX_TOOL_CALLS,
    )

    assert [call.name for call in calls] == [
        "portfolio_risk",
        "asset_comparison",
        "dividends_received",
    ]


@pytest.mark.asyncio
async def test_plan_returns_the_sanitized_calls_and_the_cost():
    provider = _FakeProvider(calls=[ToolCall("portfolio_risk", {})])

    plan = await SingleStepToolRuntime(provider).plan("minha carteira está arriscada?", TOOLS, 3)

    assert plan.calls == [ToolCall("portfolio_risk", {})]
    assert (plan.provider, plan.input_tokens, plan.output_tokens) == ("fake", 90, 10)
    assert plan.reason is None


@pytest.mark.asyncio
async def test_only_the_question_reaches_the_model():
    provider = _FakeProvider()

    await SingleStepToolRuntime(provider).plan("  compare PETR4 e VALE3  ", TOOLS, 2)

    system, prompt, tools = provider.received
    assert prompt == "Pergunta do usuário:\ncompare PETR4 e VALE3"
    assert "1 a 2 ferramentas" in system
    assert tools == TOOLS


@pytest.mark.asyncio
async def test_no_tool_chosen_is_a_valid_answer():
    plan = await SingleStepToolRuntime(_FakeProvider(calls=[])).plan("oi, tudo bem?", TOOLS, 3)

    assert plan.calls == []
    assert plan.reason == "no_tool"


@pytest.mark.asyncio
async def test_chain_without_tool_calling_is_reported():
    provider = _FakeProvider(error=ToolCallingNotSupported("groq"))

    plan = await SingleStepToolRuntime(provider).plan("risco?", TOOLS, 3)

    assert plan.reason == "not_supported"
    assert plan.calls == []


@pytest.mark.asyncio
async def test_max_calls_never_goes_above_three():
    provider = _FakeProvider()

    await SingleStepToolRuntime(provider).plan("tudo", TOOLS, 10)

    assert "1 a 3 ferramentas" in provider.received[0]
