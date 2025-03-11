# new abstraction for database

from abc import ABC, abstractmethod

from .models import Chat, ChatMessage, Message, NewMessage, Options, QueryInput


class Database(ABC):
    """Abstract base class for database operations."""

    @abstractmethod
    async def get_all_chats(
        self,
        opts: Options | None = None,
    ) -> list[Chat]:
        """Retrieve all chats from the database.

        Args:
            opts: Optional database operation options.

        Returns:
            List of Chat objects.
        """

    @abstractmethod
    async def get_chat_with_messages(
        self,
        chat_id: str,
        opts: Options | None = None,
    ) -> ChatMessage | None:
        """Retrieve a chat with all its messages.

        Args:
            chat_id: Unique identifier for the chat.
            opts: Optional database operation options.

        Returns:
            ChatMessage object or None if not found.
        """

    @abstractmethod
    async def create_chat(
        self,
        chat_id: str,
        title: str,
        opts: Options | None = None,
    ) -> Chat | None:
        """Create a new chat.

        Args:
            chat_id: Unique identifier for the chat.
            title: Title of the chat.
            opts: Optional database operation options.

        Returns:
            Created Chat object or None if creation failed.
        """

    @abstractmethod
    async def create_message(
        self,
        message: NewMessage,
        opts: Options | None = None,
    ) -> Message:
        """Create a new message.

        Args:
            message: NewMessage object containing message data.
            opts: Optional database operation options.

        Returns:
            Created Message object.
        """

    @abstractmethod
    async def check_chat_exists(
        self,
        chat_id: str,
        opts: Options | None = None,
    ) -> bool:
        """Check if a chat exists in the database.

        Args:
            chat_id: Unique identifier for the chat.
            opts: Optional database operation options.

        Returns:
            True if chat exists, False otherwise.
        """

    @abstractmethod
    async def delete_chat(self, chat_id: str, opts: Options | None = None) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            opts: Optional database operation options.
        """

    @abstractmethod
    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
        opts: Options | None = None,
    ) -> None:
        """Delete all messages after a specific message in a chat.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID after which all messages will be deleted.
            opts: Optional database operation options.
        """

    @abstractmethod
    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        opts: Options | None = None,
    ) -> Message:
        """Update the content of a message.

        Args:
            message_id: Unique identifier for the message.
            data: New content for the message.
            opts: Optional database operation options.

        Returns:
            Updated Message object.
        """

    @abstractmethod
    async def get_next_ai_message(self, chat_id: str, message_id: str) -> Message:
        """Get the next AI message after a specific message.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID to find the next AI message after.

        Returns:
            Next AI Message object.
        """
