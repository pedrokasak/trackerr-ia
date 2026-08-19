"""
Acesso ao vector store, com isolamento por usuario como invariante — nao
opcional, nao configuravel por chamador.

Risco P0 do doc (docs/rag-trackerr-ia.md, secao 4): vazamento de dado entre
usuarios no retrieval. Mitigacao aqui: `user_id` e parametro obrigatorio,
validado antes de montar a query, e sempre vira clausula WHERE — nunca um
filtro "se fornecido". `build_search_statement` fica separado de `search`
de proposito: permite testar que a clausula existe sem precisar de um
Postgres real rodando (ver tests/test_rag_repository.py).
"""

from sqlalchemy import Select, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import DocumentChunk


class MissingUserIdError(ValueError):
    """Retrieval sem user_id e sempre erro de programacao, nunca 'busca geral'."""


class DocumentChunkRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self._session.add_all(chunks)
        await self._session.commit()

    @staticmethod
    def build_search_statement(
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: str | None = None,
    ) -> Select:
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — retrieval nunca roda sem escopo de usuario."
            )
        if top_k <= 0:
            raise ValueError("top_k precisa ser positivo.")

        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.user_id == user_id)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        if source_type:
            statement = statement.where(DocumentChunk.source_type == source_type)
        return statement

    async def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[DocumentChunk]:
        statement = self.build_search_statement(
            user_id, query_embedding, top_k, source_type
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def delete_for_user(self, user_id: str) -> int:
        """Usado por reprocessamento de ingestao — apaga os chunks antigos
        de um usuario antes de gravar a versao atualizada."""
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — delete nunca roda sem escopo de usuario."
            )
        result = await self._session.execute(
            delete(DocumentChunk).where(DocumentChunk.user_id == user_id)
        )
        await self._session.commit()
        return result.rowcount or 0
