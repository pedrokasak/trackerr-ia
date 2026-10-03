"""POST /api/chat/plan (TRA-241)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from benchmark.providers.base import LLMProvider, ToolCall, ToolCallsResult
from main import app

client = TestClient(app)

TOOLS = [
    {
        "name": "portfolio_risk",
        "description": "Risco e concentração da carteira.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "asset_comparison",
        "description": "Compara dois ou mais ativos.",
        "parameters": {
            "type": "object",
            "properties": {"tickers": {"type": "array", "items": {"type": "string"}}},
        },
    },
]


class _Provider(LLMProvider):
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def analyze(self, prompt):
        return {}

    async def call_tools(self, system, prompt, tools):
        if self.error:
            raise self.error
        return self.result

    @property
    def provider_name(self):
        return "fake"


def test_returns_the_chosen_calls():
    provider = _Provider(
        ToolCallsResult(
            calls=[
                ToolCall("asset_comparison", {"tickers": ["PETR4", "VALE3"]}),
                ToolCall("portfolio_risk", {}),
            ],
            provider="openrouter",
            input_tokens=400,
            output_tokens=35,
        )
    )
    with patch("main.LLMFactory.get_provider", return_value=provider):
        response = client.post(
            "/api/chat/plan",
            json={"question": "compare PETR4 e VALE3 e diga o impacto no meu risco", "tools": TOOLS},
        )

    assert response.status_code == 200
    assert response.json() == {
        "calls": [
            {"name": "asset_comparison", "arguments": {"tickers": ["PETR4", "VALE3"]}},
            {"name": "portfolio_risk", "arguments": {}},
        ],
        "provider": "openrouter",
        "input_tokens": 400,
        "output_tokens": 35,
        "reason": None,
    }


def test_no_tool_is_a_200_with_the_reason():
    provider = _Provider(ToolCallsResult(calls=[], provider="gemini"))
    with patch("main.LLMFactory.get_provider", return_value=provider):
        response = client.post("/api/chat/plan", json={"question": "oi", "tools": TOOLS})

    assert response.status_code == 200
    assert response.json()["reason"] == "no_tool"


def test_provider_failure_is_a_502_without_details():
    provider = _Provider(error=RuntimeError("chave invalida sk-123"))
    with patch("main.LLMFactory.get_provider", return_value=provider):
        response = client.post("/api/chat/plan", json={"question": "risco?", "tools": TOOLS})

    assert response.status_code == 502
    assert "sk-123" not in response.text


def test_rejects_a_malformed_catalog():
    too_many = [dict(TOOLS[0], name=f"tool_{i}") for i in range(41)]
    for body in (
        {"question": "risco?", "tools": []},
        {"question": "risco?", "tools": [dict(TOOLS[0], name="Drop Table")]},
        {"question": "risco?", "tools": too_many},
        {"question": "risco?", "tools": TOOLS, "max_calls": 4},
        {"question": "", "tools": TOOLS},
    ):
        assert client.post("/api/chat/plan", json=body).status_code == 422
