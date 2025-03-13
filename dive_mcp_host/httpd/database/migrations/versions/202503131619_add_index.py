"""Add indexes.

Revision ID: d74bc7569f6c
Revises: 6a3fdec4ab74
Create Date: 2025-03-13 16:19:14.259905

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d74bc7569f6c"
down_revision: str | None = "6a3fdec4ab74"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "idx_chats_user_id",
        "chats",
        ["user_id"],
        unique=False,
        postgresql_using="hash",
    )
    op.create_index(
        "idx_messages_chat_id",
        "messages",
        ["chat_id"],
        unique=False,
        postgresql_using="hash",
    )
    op.create_index(
        "messages_message_id_index",
        "messages",
        ["message_id"],
        unique=False,
        postgresql_using="hash",
    )
    op.create_index(
        "idx_resource_usage_message_id",
        "resource_usage",
        ["message_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_resource_usage_message_id", table_name="resource_usage")
    op.drop_index(
        "messages_message_id_index", table_name="messages", postgresql_using="hash"
    )
    op.drop_index(
        "idx_messages_chat_id", table_name="messages", postgresql_using="hash"
    )
    op.drop_index("idx_chats_user_id", table_name="chats", postgresql_using="hash")
