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

from rag.models import DocumentChunk, SharedKnowledgeChunk, compute_content_hash


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

    async def search_with_distance(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Igual a `search`, mas devolve a distancia de cosseno junto — necessario
        pra mesclar chunks pessoais com os compartilhados (TRA-87) numa unica
        ordem de relevancia. Isolamento por usuario intocado: mesma clausula
        `WHERE user_id = X` de `build_search_statement`.
        """
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — retrieval nunca roda sem escopo de usuario."
            )
        if top_k <= 0:
            raise ValueError("top_k precisa ser positivo.")
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        result = await self._session.execute(
            select(DocumentChunk, distance.label("distance"))
            .where(DocumentChunk.user_id == user_id)
            .order_by(distance)
            .limit(top_k)
        )
        return [(row[0], float(row[1])) for row in result.all()]

    async def get_hashes_for_user(self, user_id: str) -> dict[str, str]:
        """
        Mapa source_id -> content_hash dos chunks que o usuario ja tem.

        Base da ingestao incremental (TRA-74): so re-embeda o que mudou de
        fato. Chunk gravado antes de TRA-74 tem content_hash NULL e por isso
        nunca casa com o hash novo — ou seja, e reprocessado uma vez e
        depois passa a ser pulado. Comportamento correto, sem backfill.
        """
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — leitura de hash nunca roda sem escopo de usuario."
            )
        result = await self._session.execute(
            select(DocumentChunk.source_id, DocumentChunk.content_hash).where(
                DocumentChunk.user_id == user_id
            )
        )
        return {
            source_id: content_hash
            for source_id, content_hash in result.all()
            if content_hash is not None
        }

    async def delete_by_source_ids(self, user_id: str, source_ids: list[str]) -> int:
        """Apaga chunks especificos de um usuario — usado pra substituir os
        que mudaram e pra remover os que sumiram da carteira."""
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — delete nunca roda sem escopo de usuario."
            )
        if not source_ids:
            return 0
        result = await self._session.execute(
            delete(DocumentChunk)
            .where(DocumentChunk.user_id == user_id)
            .where(DocumentChunk.source_id.in_(source_ids))
        )
        await self._session.commit()
        return result.rowcount or 0

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


class SharedKnowledgeRepository:
    """
    Acesso ao conhecimento curado e compartilhado (TRA-87).

    Sem `user_id` de proposito: e conteudo pra todos os usuarios (base fiscal
    revisada, TRA-36). A ausencia de filtro por usuario NAO enfraquece o
    isolamento de `document_chunks` — sao tabelas separadas; esta nunca
    guarda dado pessoal.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def search_with_distance(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        knowledge_base: str | None = None,
    ) -> list[tuple[SharedKnowledgeChunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k precisa ser positivo.")
        distance = SharedKnowledgeChunk.embedding.cosine_distance(query_embedding)
        statement = (
            select(SharedKnowledgeChunk, distance.label("distance"))
            .order_by(distance)
            .limit(top_k)
        )
        if knowledge_base:
            statement = statement.where(
                SharedKnowledgeChunk.knowledge_base == knowledge_base
            )
        result = await self._session.execute(statement)
        return [(row[0], float(row[1])) for row in result.all()]

    async def get_hashes(self, knowledge_base: str) -> dict[str, str]:
        """Mapa source_id -> content_hash da base, pra upsert incremental."""
        result = await self._session.execute(
            select(
                SharedKnowledgeChunk.source_id, SharedKnowledgeChunk.content_hash
            ).where(SharedKnowledgeChunk.knowledge_base == knowledge_base)
        )
        return {source_id: content_hash for source_id, content_hash in result.all()}

    async def delete_by_source_ids(
        self, knowledge_base: str, source_ids: list[str]
    ) -> int:
        if not source_ids:
            return 0
        result = await self._session.execute(
            delete(SharedKnowledgeChunk)
            .where(SharedKnowledgeChunk.knowledge_base == knowledge_base)
            .where(SharedKnowledgeChunk.source_id.in_(source_ids))
        )
        await self._session.commit()
        return result.rowcount or 0

    async def add_chunks(self, chunks: list[SharedKnowledgeChunk]) -> None:
        if not chunks:
            return
        self._session.add_all(chunks)
        await self._session.commit()


def compute_shared_content_hash(content: str) -> str:
    """Alias explicito — mesmo hash de conteudo do resto do RAG (TRA-74)."""
    return compute_content_hash(content)
