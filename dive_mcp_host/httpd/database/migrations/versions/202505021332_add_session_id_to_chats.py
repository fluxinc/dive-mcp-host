"""add_session_id_to_chats

Revision ID: d5d689d507f3
Revises: 77c3a2c91298
Create Date: 2025-05-02 13:32:30.155004

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5d689d507f3'
down_revision: Union[str, None] = '77c3a2c91298'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("chats") as batch_op:
        batch_op.add_column(
            sa.Column("session_id", sa.Text(), nullable=True)
        )
        batch_op.create_index(
            "ix_chats_session_id", ["session_id"], unique=False
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("chats") as batch_op:
        batch_op.drop_index("ix_chats_session_id")
        batch_op.drop_column("session_id")
