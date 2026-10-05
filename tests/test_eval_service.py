"""Serviço da avaliação offline (TRA-242): juiz, agregados e LGPD."""

import pytest

from benchmark.providers.base import LLMProvider
from evals.judge import JudgeVerdict, LlmJudge, judge_provider_name, parse_verdict
from evals.service import EvalItem, EvalService, GuardStats
from rag.query_service import DISCLAIMER


class _JudgeProvider(LLMProvider):
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.prompts = []

    async def analyze(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.responses.pop(0) if self.responses else {}

    @property
    def provider_name(self):
        return "juiz-fake"


GOOD_RAG = {
    "fidelity": 0.9,
    "numeric_hallucination": False,
    "recommendation_language": False,
    "usefulness": 0.8,
    "level_fit": 1.0,
}
GOOD_CHAT = {
    "fidelity": None,
    "numeric_hallucination": None,
    "recommendation_language": False,
    "usefulness": 0.6,
    "level_fit": 0.9,
}


def _rag(item_id="rag-1", answer=None):
    return EvalItem(
        id=item_id,
        route="rag",
        intent="rag_query",
        question="Quanto recebi de proventos?",
        answer=answer or f"Você recebeu R$ 3.284,50.\n\n{DISCLAIMER}",
        context="- Proventos recebidos nos últimos 12 meses: R$ 3.284,50.",
    )


def _chat(intent="portfolio_risk", route="regex"):
    return EvalItem(
        id=f"chat-{intent}",
        route=route,
        intent=intent,
        question="Minha carteira está arriscada?",
        answer="Sua carteira apresenta um Score de Risco de 64/100.",
    )


def test_parse_verdict_requires_the_rag_fields_only_with_context():
    assert parse_verdict(GOOD_RAG, has_context=True) == JudgeVerdict(0.9, False, False, 0.8, 1.0)
    assert parse_verdict({**GOOD_RAG, "fidelity": None}, has_context=True) is None
    assert parse_verdict(GOOD_CHAT, has_context=False) == JudgeVerdict(None, None, False, 0.6, 0.9)
    assert parse_verdict({"usefulness": 2, "level_fit": -1, "recommendation_language": False}, False) == (
        JudgeVerdict(None, None, False, 1.0, 0.0)
    )
    assert parse_verdict({"usefulness": "boa"}, has_context=False) is None


def test_judge_is_a_different_provider_than_the_generator(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    assert judge_provider_name() == "gemini"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    for env in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert judge_provider_name() is None

    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "Claude")
    assert judge_provider_name() == "claude"


@pytest.mark.asyncio
async def test_reports_only_aggregates_by_route_and_intent():
    provider = _JudgeProvider([GOOD_RAG, GOOD_CHAT, {**GOOD_CHAT, "usefulness": 1.0}])
    service = EvalService(LlmJudge(provider))

    report = await service.evaluate(
        [_rag(), _chat("portfolio_risk"), _chat("asset_comparison", route="tool_calling")],
        GuardStats(answered=40, rejected=2),
    )

    assert report.totals == {"received": 3, "evaluated": 3, "judged": 3, "judge_failed": 0, "pii_blocked": 0}
    assert report.by_route["rag"]["fidelity"] == 0.9
    assert report.by_route["rag"]["disclaimer_rate"] == 1.0
    assert report.by_route["rag"]["numeric_hallucination_rate"] == 0.0
    assert report.by_route["regex"]["usefulness"] == 0.6
    assert report.by_route["regex"]["fidelity"] is None
    assert report.by_route["tool_calling"]["usefulness"] == 1.0
    assert set(report.by_intent) == {"rag_query", "portfolio_risk", "asset_comparison"}
    assert report.guard == {"answered": 40, "rejected": 2, "rejection_rate": 0.05}
    assert report.judge_provider == "juiz-fake"
    # Nenhum texto no relatório.
    assert "Score de Risco" not in repr(report)


@pytest.mark.asyncio
async def test_deterministic_checks_catch_what_the_judge_missed():
    provider = _JudgeProvider([GOOD_RAG])  # o juiz diz que está tudo certo
    report = await EvalService(LlmJudge(provider)).evaluate(
        [_rag(answer=f"Você recebeu R$ 9.999,00.\n\n{DISCLAIMER}")]
    )

    assert report.by_route["rag"]["numeric_hallucination_rate"] == 1.0


@pytest.mark.asyncio
async def test_personal_data_never_reaches_the_judge():
    provider = _JudgeProvider([GOOD_CHAT])
    item = EvalItem(
        id="chat-1",
        route="regex",
        intent="portfolio_summary",
        question="Sou a Ana, cpf 123.456.789-09, email ana@exemplo.com. Quanto tenho?",
        answer="O patrimônio total estimado é de R$ 61.420,00.",
    )

    await EvalService(LlmJudge(provider)).evaluate([item])

    prompt = provider.prompts[0]
    assert "123.456.789-09" not in prompt
    assert "ana@exemplo.com" not in prompt
    assert "[CPF]" in prompt and "[EMAIL]" in prompt


@pytest.mark.asyncio
async def test_a_failing_judge_keeps_the_deterministic_checks():
    provider = _JudgeProvider(error=RuntimeError("429"))

    report = await EvalService(LlmJudge(provider)).evaluate([_rag()])

    assert report.totals["judge_failed"] == 1
    assert report.by_route["rag"]["disclaimer_rate"] == 1.0
    assert report.by_route["rag"]["fidelity"] is None


@pytest.mark.asyncio
async def test_without_a_judge_only_deterministic_checks_run():
    report = await EvalService(None).evaluate([_rag(), _chat()])

    assert report.judge_provider is None
    assert report.totals["judged"] == 0
    assert report.by_route["rag"]["recommendation_rate"] == 0.0
    assert report.by_route["regex"]["recommendation_rate"] is None


@pytest.mark.asyncio
async def test_ignores_unknown_routes():
    item = EvalItem(id="x", route="email", intent="i", question="q", answer="a")

    report = await EvalService(None).evaluate([item])

    assert report.totals["evaluated"] == 0


def test_a_judge_without_api_key_degrades_to_deterministic_only(monkeypatch):
    from evals.judge import build_judge_provider

    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert build_judge_provider() is None


@pytest.mark.asyncio
async def test_a_full_server_batch_never_cuts_the_rag_samples():
    items = [_chat(f"intent_{i}") for i in range(120)] + [_rag(f"rag-{i}") for i in range(30)]

    report = await EvalService(None).evaluate(items)

    assert report.by_route["rag"]["count"] == 30
