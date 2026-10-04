"""
Amostra das respostas do RAG para a avaliação semanal (TRA-242), da
auditoria (`rag_query_audit_log`). O contexto é o dos chunks pessoais que a
resposta usou; os compartilhados (base fiscal) não ficam na auditoria por id
e não entram — limite conhecido: a fidelidade dessas respostas sai
subestimada.

O `user_id` da auditoria não sai daqui: o item leva só o id da linha.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evals.service import EvalItem, GuardStats
from rag.models import DocumentChunk, RagQueryAuditLog

# Resultados do guardrail que não são recusa: respondeu, ou não havia contexto.
_NOT_REJECTED = ("ok", "no_context")


async def guard_stats(session: AsyncSession, since: datetime) -> GuardStats:
    """Respostas que passaram pelo LLM na janela e quantas o guardrail barrou."""
    answered = await session.scalar(
        select(func.count())
        .select_from(RagQueryAuditLog)
        .where(RagQueryAuditLog.created_at >= since)
        .where(RagQueryAuditLog.guard_result != "no_context")
    )
    rejected = await session.scalar(
        select(func.count())
        .select_from(RagQueryAuditLog)
        .where(RagQueryAuditLog.created_at >= since)
        .where(RagQueryAuditLog.guard_result.notin_(_NOT_REJECTED))
    )
    return GuardStats(answered=int(answered or 0), rejected=int(rejected or 0))


async def sample_rag_items(
    session: AsyncSession, window_days: int, limit: int
) -> Tuple[List[EvalItem], GuardStats]:
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    stats = await guard_stats(session, since)
    if limit <= 0:
        return [], stats

    rows = (
        await session.execute(
            select(RagQueryAuditLog)
            .where(RagQueryAuditLog.created_at >= since)
            .where(RagQueryAuditLog.guard_result == "ok")
            .order_by(func.random())
            .limit(limit)
        )
    ).scalars().all()

    chunk_ids = sorted({chunk_id for row in rows for chunk_id in (row.retrieved_chunk_ids or [])})
    contents = {}
    if chunk_ids:
        for chunk in (
            await session.execute(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))
        ).scalars():
            contents[chunk.id] = chunk.content

    items = [
        EvalItem(
            id=f"rag-{row.id}",
            route="rag",
            intent="rag_query",
            question=row.question,
            answer=row.response_text,
            context="\n".join(
                f"- {contents[chunk_id]}"
                for chunk_id in (row.retrieved_chunk_ids or [])
                if chunk_id in contents
            )
            or None,
        )
        for row in rows
    ]
    return items, stats
