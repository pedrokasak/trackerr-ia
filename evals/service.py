"""
Avaliação offline das respostas de IA (TRA-242).

Recebe amostras já sem `user_id` — do chat (rota regex ou roteador com
tools, mandadas pelo server) e do RAG (auditoria deste serviço) —, tira a
PII que restar, roda as checagens determinísticas e o juiz, e devolve SÓ
agregados: nenhuma pergunta ou resposta sai daqui no relatório.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evals.checks import DeterministicChecks, check_rag_answer
from evals.judge import RUBRIC_VERSION, JudgeVerdict, LlmJudge
from evals.pii import contains_pii, scrub

logger = logging.getLogger(__name__)

ROUTES = ("regex", "tool_calling", "rag")


@dataclass(frozen=True)
class EvalItem:
    id: str
    route: str
    intent: str
    question: str
    answer: str
    context: Optional[str] = None
    level: str = "intermediate"


@dataclass(frozen=True)
class GuardStats:
    """Respostas do RAG na janela: quantas o guardrail barrou."""

    answered: int
    rejected: int


@dataclass
class _Result:
    route: str
    intent: str
    verdict: Optional[JudgeVerdict] = None
    checks: Optional[DeterministicChecks] = None


@dataclass
class EvalReport:
    rubric_version: str
    judge_provider: Optional[str]
    totals: Dict[str, int]
    by_route: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    by_intent: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    guard: Dict[str, Optional[float]] = field(default_factory=dict)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 3) if present else None


def _rate(flags: List[Optional[bool]]) -> Optional[float]:
    present = [flag for flag in flags if flag is not None]
    return round(sum(1 for flag in present if flag) / len(present), 3) if present else None


def _recommendation(result: _Result) -> Optional[bool]:
    flags = [
        result.verdict.recommendation_language if result.verdict else None,
        result.checks.recommendation_language if result.checks else None,
    ]
    known = [flag for flag in flags if flag is not None]
    return any(known) if known else None


def _hallucination(result: _Result) -> Optional[bool]:
    flags = [
        result.verdict.numeric_hallucination if result.verdict else None,
        result.checks.numeric_hallucination if result.checks else None,
    ]
    known = [flag for flag in flags if flag is not None]
    return any(known) if known else None


def _summary(results: List[_Result]) -> Dict[str, Optional[float]]:
    verdicts = [result.verdict for result in results if result.verdict]
    return {
        "count": len(results),
        "judged": len(verdicts),
        "fidelity": _mean([verdict.fidelity for verdict in verdicts]),
        "numeric_hallucination_rate": _rate([_hallucination(r) for r in results]),
        "recommendation_rate": _rate([_recommendation(r) for r in results]),
        "usefulness": _mean([verdict.usefulness for verdict in verdicts]),
        "level_fit": _mean([verdict.level_fit for verdict in verdicts]),
        "disclaimer_rate": _rate(
            [result.checks.disclaimer_present if result.checks else None for result in results]
        ),
    }


class EvalService:
    # Teto do lote do server (120) mais o da amostra do RAG (100): nenhuma
    # fonte pode sumir do relatório por causa do corte.
    MAX_ITEMS = 220

    def __init__(self, judge: Optional[LlmJudge]) -> None:
        self._judge = judge

    async def evaluate(
        self, items: List[EvalItem], guard_stats: Optional[GuardStats] = None
    ) -> EvalReport:
        totals = {"received": len(items), "evaluated": 0, "judged": 0, "judge_failed": 0, "pii_blocked": 0}
        results: List[_Result] = []

        for item in items[: self.MAX_ITEMS]:
            if item.route not in ROUTES:
                continue
            question = scrub(item.question)
            answer = scrub(item.answer)
            context = scrub(item.context) if item.context else None
            # Trava final: o que ainda parecer dado pessoal não sai daqui.
            if any(contains_pii(text) for text in (question, answer, context or "")):
                totals["pii_blocked"] += 1
                continue

            result = _Result(route=item.route, intent=item.intent or "unknown")
            if item.route == "rag":
                result.checks = check_rag_answer(question, context or "", answer)
            if self._judge:
                try:
                    result.verdict = await self._judge.judge(
                        item.route, question, answer, context, item.level
                    )
                except Exception as error:  # juiz fora: a amostra fica sem nota
                    logger.warning("Juiz da avaliação falhou: %s", type(error).__name__)
                if result.verdict:
                    totals["judged"] += 1
                else:
                    totals["judge_failed"] += 1
            totals["evaluated"] += 1
            results.append(result)

        by_route = {
            route: _summary([r for r in results if r.route == route])
            for route in ROUTES
            if any(r.route == route for r in results)
        }
        intents = sorted({result.intent for result in results})
        by_intent = {
            intent: _summary([r for r in results if r.intent == intent]) for intent in intents
        }
        guard: Dict[str, Optional[float]] = {}
        if guard_stats:
            guard = {
                "answered": guard_stats.answered,
                "rejected": guard_stats.rejected,
                "rejection_rate": round(guard_stats.rejected / guard_stats.answered, 3)
                if guard_stats.answered
                else None,
            }
        return EvalReport(
            rubric_version=RUBRIC_VERSION,
            judge_provider=self._judge.provider_name if self._judge else None,
            totals=totals,
            by_route=by_route,
            by_intent=by_intent,
            guard=guard,
        )
