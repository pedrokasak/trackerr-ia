"""add metadata, as_of and content_hash to document_chunks (TRA-74)

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-20

Aditiva de proposito: as tres colunas sao nullable e sem backfill. Chunk
gravado antes desta migracao continua valido pra retrieval (o embedding e
o content nao mudaram) e ganha os campos novos no proximo ciclo de
ingestao, que reescreve o chunk de qualquer forma.

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("document_chunks", sa.Column("as_of", sa.Date(), nullable=True))
    op.add_column(
        "document_chunks",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    # A ingestao incremental (TRA-74) busca os chunks existentes do usuario
    # por source_id pra comparar hash. Sem este indice, cada ciclo de
    # ingestao vira seq scan na tabela inteira.
    op.create_index(
        "ix_document_chunks_user_id_source_id",
        "document_chunks",
        ["user_id", "source_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_user_id_source_id", table_name="document_chunks"
    )
    op.drop_column("document_chunks", "content_hash")
    op.drop_column("document_chunks", "as_of")
    op.drop_column("document_chunks", "metadata")
