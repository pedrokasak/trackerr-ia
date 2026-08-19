"""
Schema do vector store (TRA-35).

Um unico modelo por enquanto: chunks de documento do usuario, com
embedding. `user_id` e NOT NULL e obrigatorio em toda query de retrieval —
ver repository.py, onde o filtro nunca e opcional (risco P0 do doc: vazamento
de dado entre usuarios).

Dimensao do embedding fixa em 768 (Gemini text-embedding-004, o provider
default do LLM_PROVIDER — ver benchmark/providers/factory.py). Trocar de
modelo de embedding no futuro exige migracao (coluna vector tem dimensao
fixa no Postgres) — documentado aqui de proposito pra nao ser surpresa.
"""

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768


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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        # Toda query real filtra por user_id primeiro — indice cobre o caso
        # comum (usuario + tipo de fonte) sem exigir segundo index scan.
        Index("ix_document_chunks_user_id_source_type", "user_id", "source_type"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_rag_query_audit_log_user_id", "user_id"),
    )
