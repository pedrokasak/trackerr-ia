"""create shared_knowledge_chunks (TRA-87 shared curated knowledge)

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-21

Tabela separada de document_chunks de proposito (ver SharedKnowledgeChunk em
rag/models.py): conteudo compartilhado, sem user_id, sem tocar na query de
isolamento por usuario.

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shared_knowledge_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("knowledge_base", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_shared_knowledge_chunks_kb_source_id",
        "shared_knowledge_chunks",
        ["knowledge_base", "source_id"],
        unique=True,
    )
    op.execute(
        "CREATE INDEX ix_shared_knowledge_chunks_embedding_cosine "
        "ON shared_knowledge_chunks USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_shared_knowledge_chunks_embedding_cosine")
    op.drop_index(
        "ix_shared_knowledge_chunks_kb_source_id",
        table_name="shared_knowledge_chunks",
    )
    op.drop_table("shared_knowledge_chunks")
