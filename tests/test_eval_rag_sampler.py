"""Amostra do RAG para a avaliação (TRA-242): sem user_id, com o contexto usado."""

from types import SimpleNamespace

import pytest

from evals.rag_sampler import sample_rag_items


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _Session:
    """Responde as consultas na ordem: contagens, linhas da auditoria, chunks."""

    def __init__(self, counts, audit_rows, chunks):
        self._counts = list(counts)
        self._results = [_Result(audit_rows), _Result(chunks)]
        self.statements = []

    async def scalar(self, statement):
        self.statements.append(statement)
        return self._counts.pop(0)

    async def execute(self, statement):
        self.statements.append(statement)
        return self._results.pop(0)


AUDIT_ROW = SimpleNamespace(
    id=42,
    user_id="user-123",
    question="Quanto recebi de proventos?",
    response_text="Você recebeu R$ 3.284,50.",
    retrieved_chunk_ids=[7, 9, 99],
    guard_result="ok",
)


@pytest.mark.asyncio
async def test_samples_answers_with_their_context_and_without_the_user():
    session = _Session(
        counts=[40, 2],
        audit_rows=[AUDIT_ROW],
        chunks=[
            SimpleNamespace(id=9, content="Maior pagador: BBAS3."),
            SimpleNamespace(id=7, content="Proventos: R$ 3.284,50."),
        ],
    )

    items, stats = await sample_rag_items(session, window_days=7, limit=30)

    assert (stats.answered, stats.rejected) == (40, 2)
    [item] = items
    assert item.id == "rag-42"
    assert item.route == "rag"
    # Na ordem em que a resposta usou; chunk que sumiu (99) não quebra nada.
    assert item.context == "- Proventos: R$ 3.284,50.\n- Maior pagador: BBAS3."
    assert "user-123" not in repr(item)


@pytest.mark.asyncio
async def test_zero_samples_still_reports_the_guard():
    session = _Session(counts=[10, 1], audit_rows=[], chunks=[])

    items, stats = await sample_rag_items(session, window_days=7, limit=0)

    assert items == []
    assert stats.rejected == 1
    assert len(session.statements) == 2
