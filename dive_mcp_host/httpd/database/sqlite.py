# from sqlite3 import connect

from . import Database
from .models import (
    Chat,
    ChatMessage,
    Message,
    NewMessage,
    Options,
    QueryInput,
)


class SqliteDatabase(Database):
    """Sqlite database implementation."""

    def __init__(self, db_path: str) -> None:
        """Init sqlite database."""

    async def get_all_chats(self, opts: Options | None = None) -> list[Chat]:
        raise NotImplementedError

    async def get_chat_with_messages(
        self,
        chat_id: str,
        opts: Options | None = None,
    ) -> ChatMessage | None:
        raise NotImplementedError

    async def create_chat(
        self,
        chat_id: str,
        title: str,
        opts: Options | None = None,
    ) -> Chat | None:
        raise NotImplementedError

    async def create_message(
        self,
        message: NewMessage,
        opts: Options | None = None,
    ) -> Message:
        raise NotImplementedError

    async def check_chat_exists(
        self,
        chat_id: str,
        opts: Options | None = None,
    ) -> bool:
        raise NotImplementedError

    async def delete_chat(self, chat_id: str, opts: Options | None = None) -> None:
        raise NotImplementedError

    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
        opts: Options | None = None,
    ) -> None:
        raise NotImplementedError

    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        opts: Options | None = None,
    ) -> Message:
        raise NotImplementedError

    async def get_next_ai_message(self, chat_id: str, message_id: str) -> Message:
        raise NotImplementedError
