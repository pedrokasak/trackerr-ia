"""create ri_document_chunks (TRA-264 acervo de documentos de RI)

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-29

Tabela propria para o acervo de RI (ver RiDocumentChunk em rag/models.py):
busca sempre por emissor, com documento e pagina em cada chunk.

Sem indice ivfflat, de proposito. Toda busca filtra por emissor, e o ivfflat
filtra DEPOIS de escolher os vizinhos: com o filtro seletivo, a busca pode
voltar com menos chunks que o pedido, ou nenhum. Por emissor sao poucos
milhares de chunks; a distancia exata sobre esse recorte, pelo indice
(issuer, published_at), e barata e sempre completa.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ri_document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_key", sa.String(length=128), nullable=False),
        sa.Column("issuer", sa.String(length=16), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("company", sa.String(length=200), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("category", sa.String(length=120), nullable=True),
        sa.Column("document_type", sa.String(length=60), nullable=True),
        sa.Column("period", sa.String(length=40), nullable=True),
        sa.Column("published_at", sa.Date(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("document_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_ri_document_chunks_document_chunk",
        "ri_document_chunks",
        ["document_key", "chunk_index"],
        unique=True,
    )
    op.create_index(
        "ix_ri_document_chunks_issuer_published",
        "ri_document_chunks",
        ["issuer", "published_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ri_document_chunks_issuer_published", table_name="ri_document_chunks"
    )
    op.drop_index(
        "ix_ri_document_chunks_document_chunk", table_name="ri_document_chunks"
    )
    op.drop_table("ri_document_chunks")
