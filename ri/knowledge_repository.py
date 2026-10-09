"""
Acesso ao acervo de documentos de RI (TRA-264).

Toda busca filtra por EMISSOR — o filtro nao e opcional, pelo mesmo motivo
do `user_id` em rag/repository.py: sem ele, a resposta sobre a PETR4 viria
com trecho da VALE3. Nao e isolamento de seguranca (documento publico), e
correcao: a citacao tem de ser da empresa perguntada.
"""

from datetime import date
from typing import List, Optional, Sequence, Tuple

from sqlalchemy import Select, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import RiDocumentChunk


class MissingIssuerError(ValueError):
    """Busca no acervo de RI sem emissor."""


class RiDocumentChunkRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    @staticmethod
    def build_search_statement(
        issuer: str,
        query_embedding: Sequence[float],
        top_k: int,
        published_after: Optional[date] = None,
    ) -> Select:
        normalized = (issuer or "").strip().upper()
        if not normalized:
            raise MissingIssuerError("issuer obrigatorio na busca do acervo de RI.")
        if top_k <= 0:
            raise ValueError("top_k precisa ser positivo.")

        distance = RiDocumentChunk.embedding.cosine_distance(list(query_embedding))
        statement = select(RiDocumentChunk, distance.label("distance")).where(
            RiDocumentChunk.issuer == normalized
        )
        if published_after:
            statement = statement.where(RiDocumentChunk.published_at >= published_after)
        return statement.order_by(distance).limit(top_k)

    async def search(
        self,
        issuer: str,
        query_embedding: Sequence[float],
        top_k: int,
        published_after: Optional[date] = None,
    ) -> List[Tuple[RiDocumentChunk, float]]:
        statement = self.build_search_statement(
            issuer, query_embedding, top_k, published_after
        )
        result = await self._session.execute(statement)
        return [(row[0], float(row[1])) for row in result.all()]

    async def get_document_hash(self, document_key: str) -> Optional[str]:
        result = await self._session.execute(
            select(RiDocumentChunk.document_hash)
            .where(RiDocumentChunk.document_key == document_key)
            .limit(1)
        )
        row = result.first()
        return row[0] if row else None

    async def replace_document(
        self, document_key: str, chunks: List[RiDocumentChunk]
    ) -> None:
        """Troca os chunks do documento numa transacao so: nunca meio a meio."""
        await self._session.execute(
            delete(RiDocumentChunk).where(RiDocumentChunk.document_key == document_key)
        )
        if chunks:
            self._session.add_all(chunks)
        await self._session.commit()
