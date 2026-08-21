"""
Ingestao de chunks pro RAG (TRA-72, incremental desde TRA-74).

trackerr-ia nao decide O QUE vira chunk — isso e regra de negocio do
server (quais fatos de carteira, quando recalcular), fora do escopo deste
modulo. Aqui so recebe texto ja pronto, embeda e grava.

A v1 (TRA-72) fazia substituicao completa por usuario a cada chamada. Isso
funciona, mas re-embeda tudo todo ciclo, e embedding e o custo dominante em
escala: na ordem de 1M usuarios x ~30 chunks sao 30M chamadas por ciclo.
Desde TRA-74 a ingestao e um diff por content_hash — so o que mudou de fato
paga embedding. Ver `compute_content_hash` em rag/models.py pro detalhe de
arredondamento que faz esta otimizacao valer alguma coisa.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from rag.embeddings import EmbeddingProvider
from rag.models import DocumentChunk, compute_content_hash
from rag.repository import DocumentChunkRepository, MissingUserIdError


@dataclass
class RagIngestItem:
    source_type: str
    source_id: str
    content: str
    metadata: dict[str, Any] | None = None
    as_of: date | None = None


@dataclass
class RagIngestResult:
    chunks_deleted: int
    chunks_created: int
    # Quantos chunks foram pulados por nao terem mudado desde o ciclo
    # anterior. E a metrica que mostra a otimizacao funcionando — se vier
    # sempre 0 em producao, o `content` esta carregando ruido (numero nao
    # arredondado, timestamp) e o hash nunca casa.
    chunks_unchanged: int = 0
    warnings: list[str] = field(default_factory=list)


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
        warnings: list[str] = []

        # source_id duplicado no mesmo payload e erro do chamador: os dois
        # chunks disputariam a mesma chave no diff e um sumiria em silencio.
        # Melhor avisar do que resolver por sorte de ordenacao.
        seen: set[str] = set()
        deduped: list[RagIngestItem] = []
        for item in cleaned:
            if item.source_id in seen:
                warnings.append(f"source_id duplicado ignorado: {item.source_id}")
                continue
            seen.add(item.source_id)
            deduped.append(item)

        existing_hashes = await self._chunks_repo.get_hashes_for_user(user_id)
        incoming_ids = {item.source_id for item in deduped}

        # Some da carteira -> some do vector store. Sem isso, um ativo
        # vendido continuaria aparecendo como contexto pra sempre.
        stale_ids = [
            source_id for source_id in existing_hashes if source_id not in incoming_ids
        ]

        to_write: list[RagIngestItem] = []
        unchanged = 0
        for item in deduped:
            content_hash = compute_content_hash(item.content)
            if existing_hashes.get(item.source_id) == content_hash:
                unchanged += 1
                continue
            to_write.append(item)

        # Chunk que mudou e substituido: apaga a versao antiga junto com os
        # que sumiram, numa unica passada.
        replaced_ids = [
            item.source_id for item in to_write if item.source_id in existing_hashes
        ]
        deleted = await self._chunks_repo.delete_by_source_ids(
            user_id, stale_ids + replaced_ids
        )

        if not to_write:
            return RagIngestResult(
                chunks_deleted=deleted,
                chunks_created=0,
                chunks_unchanged=unchanged,
                warnings=warnings,
            )

        chunks = []
        for item in to_write:
            embedding = await self._embedding_provider.embed(item.content)
            chunks.append(
                DocumentChunk(
                    user_id=user_id,
                    source_type=item.source_type,
                    source_id=item.source_id,
                    content=item.content,
                    embedding=embedding,
                    chunk_metadata=item.metadata,
                    as_of=item.as_of,
                    content_hash=compute_content_hash(item.content),
                )
            )

        await self._chunks_repo.add_chunks(chunks)
        return RagIngestResult(
            chunks_deleted=deleted,
            chunks_created=len(chunks),
            chunks_unchanged=unchanged,
            warnings=warnings,
        )
