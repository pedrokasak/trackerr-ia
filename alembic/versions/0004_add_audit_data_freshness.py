"""add data_max_age_days to rag_query_audit_log (TRA-77 freshness)

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-21

Aditiva: coluna nullable, sem backfill. Linhas de auditoria antigas ficam
com NULL (frescor não registrado à época), o que é honesto.

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "rag_query_audit_log",
        sa.Column("data_max_age_days", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_query_audit_log", "data_max_age_days")
