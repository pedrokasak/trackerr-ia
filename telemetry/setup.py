"""
Bootstrap do OpenTelemetry no trackerr-ia (TRA-222, F1c).

Liga traces e métricas por OTLP/HTTP para o Collector da stack em
`infra/observability` (repositório do server). Instrumenta:
  - FastAPI: um span por request de entrada (menos `/health`);
  - httpx: chamadas de saída, que é por onde os SDKs de LLM (OpenAI,
    Anthropic, Groq, google-genai) falam com os provedores.
As chamadas de LLM ganham ainda um span `gen_ai` próprio (ver `llm.py`).

Nada de corpo de request, prompt ou resposta entra nos spans. Chave de API
em query string é apagada pelo Collector antes de gravar.

Sem `OTEL_EXPORTER_OTLP_ENDPOINT`, `setup_telemetry` não faz nada.
"""

import logging
import os
from typing import Mapping, Optional

from fastapi import FastAPI

from .config import TelemetryConfig, excluded_urls, telemetry_config_from

logger = logging.getLogger(__name__)

_providers: list = []


def setup_telemetry(
    app: FastAPI, env: Optional[Mapping[str, str]] = None
) -> TelemetryConfig:
    config = telemetry_config_from(os.environ if env is None else env)
    if not config.enabled:
        return config

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": config.service_version,
            # Ainda incubando na semconv; o nome é o do padrão.
            "deployment.environment.name": config.environment,
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{config.endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{config.endpoint}/v1/metrics"),
                export_interval_millis=config.metric_interval_ms,
            )
        ],
    )
    metrics.set_meter_provider(meter_provider)

    FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded_urls())
    HTTPXClientInstrumentor().instrument()
    _providers.extend([tracer_provider, meter_provider])

    logger.info(
        "OpenTelemetry ligado: %s %s → %s",
        config.service_name,
        config.service_version,
        config.endpoint,
    )
    return config


def shutdown_telemetry() -> None:
    """Esvazia o buffer antes de o processo terminar (SIGTERM do Coolify)."""
    while _providers:
        provider = _providers.pop()
        try:
            provider.shutdown()
        except Exception:  # noqa: BLE001 — desligar nunca pode derrubar o shutdown
            logger.warning("Falha ao desligar o OpenTelemetry", exc_info=True)
