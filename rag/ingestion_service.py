"""
Ingestao de chunks pro RAG (TRA-72).

trackerr-ia nao decide O QUE vira chunk — isso e regra de negocio do
server (quais fatos de carteira, quando recalcular), fora do escopo deste
modulo. Aqui so recebe texto ja pronto, embeda e grava. Substituicao
completa por usuario a cada chamada (delete_for_user + add_chunks):
mais simples que merge incremental, e fatos de carteira mudam todo dia
mesmo — nao ha valor em preservar chunk antigo.
"""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from rag.embeddings import EmbeddingProvider
from rag.models import DocumentChunk
from rag.repository import DocumentChunkRepository, MissingUserIdError


@dataclass
class RagIngestItem:
    source_type: str
    source_id: str
    content: str


@dataclass
class RagIngestResult:
    chunks_deleted: int
    chunks_created: int


class RagIngestionService:
    def __init__(self, session: AsyncSession, embedding_provider: EmbeddingProvider) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._chunks_repo = DocumentChunkRepository(session)

    async def ingest(self, user_id: str, items: list[RagIngestItem]) -> RagIngestResult:
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — ingestao nunca roda sem escopo de usuario."
            )

        cleaned = [item for item in items if item.content and item.content.strip()]

        deleted = await self._chunks_repo.delete_for_user(user_id)

        if not cleaned:
            return RagIngestResult(chunks_deleted=deleted, chunks_created=0)

        chunks = []
        for item in cleaned:
            embedding = await self._embedding_provider.embed(item.content)
            chunks.append(
                DocumentChunk(
                    user_id=user_id,
                    source_type=item.source_type,
                    source_id=item.source_id,
                    content=item.content,
                    embedding=embedding,
                )
            )

        await self._chunks_repo.add_chunks(chunks)
        return RagIngestResult(chunks_deleted=deleted, chunks_created=len(chunks))
