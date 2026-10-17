"""OpenTelemetry do trackerr-ia (TRA-222, F1c)."""

import asyncio

import pytest
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from benchmark.providers.base import (
    LLMProvider,
    ToolCall,
    ToolCallsResult,
    ToolSpec,
)
from benchmark.providers.fallback_provider import FallbackLLMProvider
from fastapi import HTTPException
from telemetry.config import excluded_urls, telemetry_config_from
from telemetry.setup import setup_telemetry

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
# O provider global só pode ser definido uma vez por processo.
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def clear_spans():
    _exporter.clear()
    yield


class FakeProvider(LLMProvider):
    def __init__(self, fail: bool = False) -> None:
        self._model = "modelo-x"
        self._fail = fail

    @property
    def provider_name(self) -> str:
        return "fake"

    async def analyze(self, prompt: str):
        if self._fail:
            raise HTTPException(status_code=503, detail=f"eco do pedido: {prompt}")
        return {"ok": True}

    async def call_tools(self, system, prompt, tools):
        return ToolCallsResult(
            calls=[ToolCall(name="carteira", arguments={})],
            provider="fake",
            input_tokens=120,
            output_tokens=8,
        )


def run(coro):
    return asyncio.run(coro)


class TestConfig:
    def test_desligado_sem_endpoint(self):
        assert telemetry_config_from({}).enabled is False

    def test_liga_com_endpoint_e_normaliza(self):
        config = telemetry_config_from(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318/",
                "SOURCE_COMMIT": "0123456789abcdef",
                "OTEL_DEPLOYMENT_ENVIRONMENT": "production",
                "OTEL_METRIC_EXPORT_INTERVAL": "abc",
            }
        )
        assert config.enabled is True
        assert config.endpoint == "http://collector:4318"
        assert config.service_name == "trackerr-ia"
        assert config.service_version == "0123456789ab"
        assert config.environment == "production"
        assert config.metric_interval_ms == 30_000

    def test_sdk_disabled_vence_o_endpoint(self):
        config = telemetry_config_from(
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://c:4318", "OTEL_SDK_DISABLED": "TRUE"}
        )
        assert config.enabled is False

    def test_health_fica_sem_trace(self):
        assert excluded_urls() == "/health$"

    def test_setup_sem_endpoint_nao_instrumenta(self):
        app = FastAPI()
        config = setup_telemetry(app, env={})
        assert config.enabled is False
        assert not getattr(app, "_is_instrumented_by_opentelemetry", False)


class TestLlmSpans:
    def test_analyze_vira_span_gen_ai_sem_prompt(self):
        run(FakeProvider().analyze("minha carteira tem PETR4 e CPF 123"))
        (span,) = _exporter.get_finished_spans()
        assert span.name == "chat modelo-x"
        assert span.kind == trace.SpanKind.CLIENT
        assert dict(span.attributes) == {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "fake",
            "gen_ai.request.model": "modelo-x",
        }
        assert all("PETR4" not in str(value) for value in span.attributes.values())

    def test_call_tools_registra_tokens_e_ferramentas(self):
        tools = [ToolSpec(name="carteira", description="d", parameters={})]
        run(FakeProvider().call_tools("sys", "pergunta", tools))
        (span,) = _exporter.get_finished_spans()
        assert span.attributes["gen_ai.usage.input_tokens"] == 120
        assert span.attributes["gen_ai.usage.output_tokens"] == 8
        assert span.attributes["gen_ai.request.tool_count"] == 1
        assert span.attributes["gen_ai.response.tool_call_count"] == 1

    def test_erro_grava_so_o_tipo(self):
        with pytest.raises(HTTPException):
            run(FakeProvider(fail=True).analyze("dado sensível"))
        (span,) = _exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "HTTPException"
        assert not span.events  # sem exceção gravada: a mensagem ecoa o pedido
        assert "sensível" not in (span.status.description or "")

    def test_fallback_nao_duplica_span(self):
        provider = FallbackLLMProvider(FakeProvider(fail=True), FakeProvider())
        assert run(provider.analyze("p")) == {"ok": True}
        spans = _exporter.get_finished_spans()
        assert [s.status.status_code for s in spans] == [StatusCode.ERROR, StatusCode.UNSET]
        assert isinstance(provider, FallbackLLMProvider)
