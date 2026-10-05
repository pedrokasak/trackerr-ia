"""POST /api/evals/run e o gate de CI do dataset dourado (TRA-242)."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from evals.fingerprint import prompt_fingerprint
from evals.golden import check_golden, load_golden
from evals.service import EvalItem, GuardStats
from main import app
from rag.database import get_rag_session

client = TestClient(app)


async def _fake_session():
    yield object()


class _SessionOverride:
    """Troca a sessão do banco e devolve a de antes: outros módulos de teste
    deixam a deles registrada no app."""

    def __enter__(self):
        self._previous = app.dependency_overrides.get(get_rag_session)
        app.dependency_overrides[get_rag_session] = _fake_session

    def __exit__(self, *exc):
        if self._previous is None:
            app.dependency_overrides.pop(get_rag_session, None)
        else:
            app.dependency_overrides[get_rag_session] = self._previous


def _run(body, rag_items=None, stats=None, judge=None):
    with _SessionOverride(), patch(
        "main.sample_rag_items",
        AsyncMock(return_value=(rag_items or [], stats or GuardStats(0, 0))),
    ), patch("main.build_judge_provider", return_value=judge):
        return client.post("/api/evals/run", json=body)


def test_runs_over_chat_and_rag_samples_and_returns_only_aggregates():
    rag = EvalItem(
        id="rag-7",
        route="rag",
        intent="rag_query",
        question="Quanto recebi?",
        answer="Você recebeu R$ 3.284,50.",
        context="- Proventos: R$ 3.284,50.",
    )
    response = _run(
        {
            "items": [
                {
                    "id": "chat-1",
                    "route": "tool_calling",
                    "intent": "portfolio_risk",
                    "question": "Estou exposto demais a bancos?",
                    "answer": "Sua carteira apresenta um Score de Risco de 64/100.",
                }
            ]
        },
        rag_items=[rag],
        stats=GuardStats(answered=10, rejected=1),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["evaluated"] == 2
    assert set(body["by_route"]) == {"tool_calling", "rag"}
    assert body["guard"]["rejection_rate"] == 0.1
    assert body["prompt_fingerprint"] == prompt_fingerprint()
    assert "Score de Risco" not in response.text


def test_rejects_an_unknown_route_or_too_many_items():
    item = {"id": "1", "route": "email", "intent": "x", "question": "q", "answer": "a"}
    assert _run({"items": [item]}).status_code == 422
    ok = {**item, "route": "regex"}
    assert _run({"items": [ok] * 121}).status_code == 422


def test_failure_is_a_500_without_details():
    with _SessionOverride(), patch(
        "main.sample_rag_items", AsyncMock(side_effect=RuntimeError("db senha=xyz"))
    ):
        response = client.post("/api/evals/run", json={})

    assert response.status_code == 500
    assert "senha" not in response.text


# Gate de CI (aceite da TRA-242): prompt mudou sem nova avaliação, ou a
# fidelidade caiu abaixo do limiar, e este teste falha.
def test_golden_dataset_passes_the_gate():
    problems = check_golden(load_golden())

    assert problems == [], "\n".join(problems)


def test_gate_fails_when_a_prompt_changes():
    golden = load_golden()
    golden["fingerprint"] = "0000000000000000"

    assert any("prompt mudou" in problem for problem in check_golden(golden))


def test_gate_fails_when_fidelity_drops_below_the_threshold():
    golden = load_golden()
    for case in golden["cases"]:
        case["judge"]["fidelity"] = 0.5

    assert any("abaixo do limiar" in problem for problem in check_golden(golden))
