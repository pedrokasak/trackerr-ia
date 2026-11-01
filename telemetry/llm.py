"""
Span `gen_ai` em volta de cada chamada de LLM (TRA-222, F1c).

Segue as convenções semânticas de GenAI do OpenTelemetry: operação, sistema
(provedor), modelo pedido e tokens quando o provider informa. Prompt e
resposta NÃO entram: carregam carteira, perguntas e dados da pessoa.

`LLMProvider.__init_subclass__` aplica `trace_llm_methods` em cada provider
concreto. Assim o tipo continua o mesmo (`isinstance(p, ClaudeProvider)`
segue valendo) e provider novo já nasce instrumentado.

Usa só a API do OpenTelemetry. Sem SDK configurado o tracer é no-op, então
custa praticamente nada em dev e teste.
"""

import functools
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

_tracer = trace.get_tracer("trackerr-ia.llm")

# Nomes das convenções de GenAI (ainda "incubating" na semconv).
OPERATION = "gen_ai.operation.name"
SYSTEM = "gen_ai.system"
REQUEST_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ERROR_TYPE = "error.type"

_TRACED_FLAG = "__trackerr_traced__"


def trace_llm_methods(cls: type) -> type:
    """Instrumenta `analyze` e `call_tools` DEFINIDOS na própria classe."""
    for name in ("analyze", "call_tools"):
        method = cls.__dict__.get(name)
        if method is None or getattr(method, _TRACED_FLAG, False):
            continue
        setattr(cls, name, _traced(method))
    return cls


def _attributes(provider: Any) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        OPERATION: "chat",
        SYSTEM: str(getattr(provider, "provider_name", "unknown")),
    }
    model = getattr(provider, "_model", None)
    if isinstance(model, str) and model:
        attributes[REQUEST_MODEL] = model
    return attributes


def _traced(method):
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        attributes = _attributes(self)
        if method.__name__ == "call_tools":
            tools = args[2] if len(args) > 2 else kwargs.get("tools") or []
            attributes["gen_ai.request.tool_count"] = len(tools)
        name = f"chat {attributes.get(REQUEST_MODEL, attributes[SYSTEM])}"
        with _tracer.start_as_current_span(
            name,
            kind=SpanKind.CLIENT,
            attributes=attributes,
            # O erro é marcado abaixo só com o tipo: a mensagem de alguns
            # SDKs ecoa trecho do pedido, e o padrão gravaria o traceback.
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await method(self, *args, **kwargs)
            except Exception as error:
                span.set_attribute(ERROR_TYPE, type(error).__name__)
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                raise
            _record_usage(span, result)
            return result

    setattr(wrapper, _TRACED_FLAG, True)
    return wrapper


def _record_usage(span: trace.Span, result: Any) -> None:
    input_tokens = getattr(result, "input_tokens", 0)
    output_tokens = getattr(result, "output_tokens", 0)
    if input_tokens:
        span.set_attribute(INPUT_TOKENS, int(input_tokens))
    if output_tokens:
        span.set_attribute(OUTPUT_TOKENS, int(output_tokens))
    calls = getattr(result, "calls", None)
    if isinstance(calls, list):
        span.set_attribute("gen_ai.response.tool_call_count", len(calls))
