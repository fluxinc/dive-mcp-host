from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert

from dive_mcp.httpd.database.models import Chat
from dive_mcp.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp.httpd.database.orm_models import Chat as ORMChat


class PostgreSQLMessageStore(BaseMessageStore):
    """Message store for PostgreSQL."""

    async def create_chat(self, chat_id: str, title: str, user_id: str) -> Chat | None:
        """Create a new chat.

        Args:
            chat_id: Unique identifier for the chat.
            title: Title of the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            Created Chat object or None if creation failed.
        """
        query = (
            insert(ORMChat)
            .values(
                {
                    "id": chat_id,
                    "title": title,
                    "created_at": datetime.now(UTC),
                    "user_id": user_id,
                },
            )
            .on_conflict_do_nothing()
            .returning(ORMChat)
        )
        chat = await self._session.scalar(query)
        if chat is None:
            return None
        return Chat(
            id=chat.id,
            title=chat.title,
            createdAt=chat.created_at,
            user_id=chat.user_id,
        )
