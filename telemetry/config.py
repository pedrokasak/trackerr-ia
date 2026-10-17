"""
Configuração do OpenTelemetry a partir do ambiente (TRA-222, F1c).

Mesmas variáveis e mesmas decisões do server (`src/observability/
telemetry-config.ts`): a telemetria só liga com um endpoint OTLP explícito.
Em dev, teste e CI nada é exportado e o serviço sobe igual a antes.

Variáveis (nomes do padrão OpenTelemetry):
  OTEL_EXPORTER_OTLP_ENDPOINT   ex.: http://trackerr-otel-collector:4318
  OTEL_SDK_DISABLED=true        desliga mesmo com endpoint
  OTEL_SERVICE_NAME             padrão: trackerr-ia
  OTEL_DEPLOYMENT_ENVIRONMENT   padrão: ENVIRONMENT, ou development
  SOURCE_COMMIT                 versão (o Coolify injeta o commit)
  OTEL_METRIC_EXPORT_INTERVAL   ms entre exportações de métrica (30000)
"""

from dataclasses import dataclass
from typing import Mapping

DEFAULT_SERVICE_NAME = "trackerr-ia"
DEFAULT_METRIC_INTERVAL_MS = 30_000

# Health check do Coolify e do monitor externo: não vira trace.
UNTRACED_PATHS = ("/health",)


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool
    endpoint: str
    service_name: str
    service_version: str
    environment: str
    metric_interval_ms: int


def telemetry_config_from(env: Mapping[str, str]) -> TelemetryConfig:
    endpoint = str(env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip().rstrip("/")
    disabled = str(env.get("OTEL_SDK_DISABLED") or "").lower() == "true"
    return TelemetryConfig(
        enabled=bool(endpoint) and not disabled,
        endpoint=endpoint,
        service_name=str(env.get("OTEL_SERVICE_NAME") or "").strip()
        or DEFAULT_SERVICE_NAME,
        service_version=str(env.get("SOURCE_COMMIT") or "").strip()[:12] or "unknown",
        environment=str(env.get("OTEL_DEPLOYMENT_ENVIRONMENT") or "").strip()
        or str(env.get("ENVIRONMENT") or "").strip()
        or "development",
        metric_interval_ms=_positive_int(
            env.get("OTEL_METRIC_EXPORT_INTERVAL"), DEFAULT_METRIC_INTERVAL_MS
        ),
    )


def excluded_urls() -> str:
    """Formato do `excluded_urls` do instrumentador do FastAPI (regex, vírgula)."""
    return ",".join(f"{path}$" for path in UNTRACED_PATHS)


def _positive_int(raw: object, fallback: int) -> int:
    try:
        value = int(str(raw))
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback
