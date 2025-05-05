from datetime import UTC, datetime

from sqlalchemy.dialects.sqlite import insert

from dive_mcp_host.httpd.database.models import Chat
from dive_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Users as ORMUsers


class SQLiteMessageStore(BaseMessageStore):
    """Message store for SQLite."""

    async def create_chat(
        self,
        chat_id: str,
        title: str,
        session_id: str,
        user_id: str | None = None,
        user_type: str | None = None,
    ) -> Chat | None:
        """Create a new chat.

        Args:
            chat_id: Unique identifier for the chat.
            title: Title of the chat.
            session_id: Session ID for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
            user_type: Optional user type

        Returns:
            Created Chat object or None if creation failed.
        """
        if user_id is not None:
            query = (
                insert(ORMUsers)
                .values(
                    {
                        "id": user_id,
                        "user_type": user_type,
                    }
                )
                .on_conflict_do_nothing()
            )
            await self._session.execute(query)

        query = (
            insert(ORMChat)
            .values(
                {
                    "id": chat_id,
                    "title": title,
                    "created_at": datetime.now(UTC),
                    "user_id": user_id,
                    "session_id": session_id,
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
            session_id=chat.session_id,
        )
