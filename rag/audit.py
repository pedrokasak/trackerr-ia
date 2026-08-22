"""Registro de auditoria do RAG — ver RagQueryAuditLog em rag/models.py."""

from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import RagQueryAuditLog


class AuditLogRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(
        self,
        user_id: str,
        question: str,
        retrieved_chunk_ids: list[int],
        response_text: str,
        guard_result: str,
        data_max_age_days: int | None = None,
    ) -> None:
        self._session.add(
            RagQueryAuditLog(
                user_id=user_id,
                question=question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                response_text=response_text,
                guard_result=guard_result,
                data_max_age_days=data_max_age_days,
            )
        )
        await self._session.commit()
