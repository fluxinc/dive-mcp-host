"""Store user ID and resource usage.

Revision ID: 4fcde574911e
Revises: ccfaea7e687b
Create Date: 2025-03-12 17:58:51.567316

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4fcde574911e"
down_revision: str | None = "ccfaea7e687b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_table(
        "resource_usage",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("total_input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("total_output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("total_run_time", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            "resource_usage_message_id_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )

    with op.batch_alter_table("chats") as batch_op:
        batch_op.add_column(
            sa.Column("user_id", sa.Text(), nullable=False),
        )
        batch_op.create_foreign_key(
            "fk_chats_user_id",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )

    with op.batch_alter_table("messages") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=False,
            autoincrement=True,
        )
        batch_op.drop_column("files")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("files", sa.TEXT(), nullable=True))
    with op.batch_alter_table("chats") as batch_op:
        batch_op.drop_constraint("fk_chats_user_id", type_="foreignkey")
        batch_op.drop_column("user_id")
    op.drop_table("resource_usage", if_exists=True)
    op.drop_table("users", if_exists=True)
