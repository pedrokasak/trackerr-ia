"""create rag_query_audit_log (TRA-37 audit)

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-19

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rag_query_audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column(
            "retrieved_chunk_ids",
            sa.ARRAY(sa.Integer()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("guard_result", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_rag_query_audit_log_user_id",
        "rag_query_audit_log",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_rag_query_audit_log_user_id", table_name="rag_query_audit_log")
    op.drop_table("rag_query_audit_log")
