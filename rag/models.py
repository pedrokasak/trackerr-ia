"""
Schema do vector store (TRA-35).

Um unico modelo por enquanto: chunks de documento do usuario, com
embedding. `user_id` e NOT NULL e obrigatorio em toda query de retrieval —
ver repository.py, onde o filtro nunca e opcional (risco P0 do doc: vazamento
de dado entre usuarios).

Dimensao do embedding fixa em 768. Modelo real: gemini-embedding-001, com
output_dimensionality forcado pra 768 (default do modelo e 3072) — ver
rag/embeddings.py. Trocar de modelo ou de dimensao no futuro exige migracao
(coluna vector tem dimensao fixa no Postgres) — documentado aqui de
proposito pra nao ser surpresa.
"""

import hashlib
from datetime import date, datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Date, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768


def compute_content_hash(content: str) -> str:
    """
    Hash do texto do chunk, usado pra decidir se precisa re-embedar (TRA-74).

    Embedding e o custo dominante da ingestao em escala, e a maioria dos
    fatos de carteira nao muda de um dia pro outro. Quem monta o `content`
    deve arredondar os numeros (`15%`, nao `15,03%`) — senao oscilacao de
    centavo muda o texto, muda o hash, e dispara re-embed diario de tudo,
    anulando a otimizacao inteira.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    """
    Um pedaco de texto (nota de corretagem, dado fiscal, resumo de ativo)
    pronto pra retrieval. `content` e o texto original do chunk — guardado
    pra devolver como contexto na resposta do RAG, nao so pra busca.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    # Metadata estruturada do fato (symbol, sector, portfolio_weight, ...).
    # So vale a pena manter enquanto o retrieval de fato filtrar por ela —
    # metadata que ninguem consulta e peso morto. Nullable porque chunks
    # gravados antes de TRA-74 nao tem.
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    # Data de referencia do FATO, nao da gravacao — coluna de primeira classe
    # (nao chave dentro do JSON) porque o caminho de resposta filtra e checa
    # frescor por ela (TRA-77).
    as_of: Mapped[date | None] = mapped_column(Date, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        # Toda query real filtra por user_id primeiro — indice cobre o caso
        # comum (usuario + tipo de fonte) sem exigir segundo index scan.
        Index("ix_document_chunks_user_id_source_type", "user_id", "source_type"),
        # Ingestao incremental busca os chunks existentes do usuario por
        # source_id pra comparar hash — sem isso, cada ciclo faz seq scan.
        Index("ix_document_chunks_user_id_source_id", "user_id", "source_id"),
    )


class SharedKnowledgeChunk(Base):
    """
    Conhecimento CURADO e COMPARTILHADO entre todos os usuarios (TRA-87).

    Tabela separada de `document_chunks` de proposito. `document_chunks` tem
    a propriedade de seguranca mais critica do sistema — isolamento por
    usuario (TRA-35), `WHERE user_id = X` que nunca pode errar. Conteudo
    compartilhado (base fiscal curada, TRA-36) e o oposto: nao tem dono, e
    pra todo mundo. Misturar os dois numa tabela so, com um sentinel de
    user_id, acoplaria dois ciclos de vida diferentes na tabela cujo filtro
    nao pode falhar. Manter separado deixa a query de isolamento intocada.

    Sem `user_id`: e compartilhado por definicao. `knowledge_base` agrupa por
    dominio (ex.: 'fiscal'); `source_id` e o chunk_id estavel do documento
    curado (ex.: 'fiscal:acoes:isencao-20k'), usado pra upsert incremental
    por hash (mesmo padrao de TRA-74).
    """

    __tablename__ = "shared_knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_base: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Versao do conteudo curado — permite auditar qual revisao respondeu.
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        # Identidade do chunk curado: um source_id por base de conhecimento.
        # Upsert incremental depende disso pra casar o que ja existe.
        Index(
            "ix_shared_knowledge_chunks_kb_source_id",
            "knowledge_base",
            "source_id",
            unique=True,
        ),
    )


class RagQueryAuditLog(Base):
    """
    Toda interacao do RAG — pergunta, chunks recuperados, resposta final —
    fica registrada aqui. Principio 4.3 do CLAUDE.md (acoes criticas devem
    ser auditavel). `guard_result` diferente de 'ok' significa que a
    resposta persistida e o fallback determinístico, nao o texto do
    modelo — a auditoria mostra que o guardrail disparou, nao esconde.
    """

    __tablename__ = "rag_query_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, default=list
    )
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    guard_result: Mapped[str] = mapped_column(String(32), nullable=False)
    # Frescor do pior chunk usado na resposta, em dias (TRA-77). Nullable:
    # None quando nenhum chunk tinha as_of conhecido. Guardado pra permitir
    # investigar depois "por que a resposta daquele dia estava errada".
    data_max_age_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_rag_query_audit_log_user_id", "user_id"),
    )
