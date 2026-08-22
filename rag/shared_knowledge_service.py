"""
Ingestao do conhecimento curado e compartilhado (TRA-87).

Diferente da ingestao de carteira (TRA-72/74), que roda por usuario todo dia:
este conteudo (base fiscal revisada, TRA-36) e embedado UMA VEZ, versionado,
e compartilhado entre todos os usuarios. Roda sob demanda — quando uma nova
revisao do conteudo curado e aprovada — nao em cron por usuario.

Mesmo diff incremental por content_hash: so re-embeda o chunk cujo texto
mudou, remove o que saiu da base curada, pula o resto.
"""

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from rag.embeddings import EmbeddingProvider
from rag.models import SharedKnowledgeChunk, compute_content_hash
from rag.repository import SharedKnowledgeRepository


@dataclass
class SharedKnowledgeItem:
    source_id: str
    content: str
    version: Optional[str] = None


@dataclass
class SharedKnowledgeIngestResult:
    chunks_deleted: int
    chunks_created: int
    chunks_unchanged: int = 0
    warnings: list[str] = field(default_factory=list)


class SharedKnowledgeService:
    def __init__(
        self, session: AsyncSession, embedding_provider: EmbeddingProvider
    ) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._repo = SharedKnowledgeRepository(session)

    async def ingest(
        self, knowledge_base: str, items: list[SharedKnowledgeItem]
    ) -> SharedKnowledgeIngestResult:
        if not knowledge_base or not knowledge_base.strip():
            raise ValueError("knowledge_base obrigatorio.")

        cleaned = [i for i in items if i.content and i.content.strip()]
        warnings: list[str] = []

        # source_id duplicado no payload e erro do chamador — avisa em vez de
        # deixar um sumir por sorte de ordenacao.
        seen: set[str] = set()
        deduped: list[SharedKnowledgeItem] = []
        for item in cleaned:
            if item.source_id in seen:
                warnings.append(f"source_id duplicado ignorado: {item.source_id}")
                continue
            seen.add(item.source_id)
            deduped.append(item)

        existing = await self._repo.get_hashes(knowledge_base)
        incoming_ids = {item.source_id for item in deduped}

        # Saiu da base curada -> sai do store.
        stale_ids = [sid for sid in existing if sid not in incoming_ids]

        to_write: list[SharedKnowledgeItem] = []
        unchanged = 0
        for item in deduped:
            if existing.get(item.source_id) == compute_content_hash(item.content):
                unchanged += 1
                continue
            to_write.append(item)

        replaced_ids = [i.source_id for i in to_write if i.source_id in existing]
        deleted = await self._repo.delete_by_source_ids(
            knowledge_base, stale_ids + replaced_ids
        )

        if not to_write:
            return SharedKnowledgeIngestResult(
                chunks_deleted=deleted,
                chunks_created=0,
                chunks_unchanged=unchanged,
                warnings=warnings,
            )

        chunks = []
        for item in to_write:
            embedding = await self._embedding_provider.embed(item.content)
            chunks.append(
                SharedKnowledgeChunk(
                    knowledge_base=knowledge_base,
                    source_id=item.source_id,
                    content=item.content,
                    embedding=embedding,
                    content_hash=compute_content_hash(item.content),
                    version=item.version,
                )
            )
        await self._repo.add_chunks(chunks)

        return SharedKnowledgeIngestResult(
            chunks_deleted=deleted,
            chunks_created=len(chunks),
            chunks_unchanged=unchanged,
            warnings=warnings,
        )
