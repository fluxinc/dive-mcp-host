"""Update resource usage foreign key (message_id).

Revision ID: 6a3fdec4ab74
Revises: 4fcde574911e
Create Date: 2025-03-13 16:08:32.913090

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6a3fdec4ab74"
down_revision: str | None = "4fcde574911e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("messages") as batch_op:
        batch_op.create_unique_constraint(
            "messages_message_id_unique",
            ["message_id"],
        )

    with op.batch_alter_table("resource_usage") as batch_op:
        batch_op.drop_constraint(
            "resource_usage_message_id_fk",
            type_="foreignkey",
        )
        batch_op.alter_column(
            "message_id",
            existing_type=sa.BIGINT(),
            type_=sa.Text(),
            existing_nullable=False,
            postgresql_using="message_id::text",
        )
        batch_op.create_foreign_key(
            "resource_usage_message_id_fk",
            "messages",
            ["message_id"],
            ["message_id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("resource_usage") as batch_op:
        batch_op.drop_constraint(
            "resource_usage_message_id_fk",
            type_="foreignkey",
        )
        batch_op.alter_column(
            "message_id",
            existing_type=sa.Text(),
            type_=sa.BIGINT(),
            existing_nullable=False,
            postgresql_using="message_id::bigint",
        )
        batch_op.create_foreign_key(
            "resource_usage_message_id_fk",
            "messages",
            ["message_id"],
            ["id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_constraint(
            "messages_message_id_unique",
            type_="unique",
        )
