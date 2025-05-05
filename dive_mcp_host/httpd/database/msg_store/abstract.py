# new abstraction for database

from abc import ABC, abstractmethod

from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    Message,
    NewMessage,
    QueryInput,
)


class AbstractMessageStore(ABC):
    """Abstract base class for database operations."""

    @abstractmethod
    async def get_all_chats(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Chat]:
        """Retrieve all chats from the database.

        Args:
            user_id: User ID or fingerprint, depending on the prefix.
            session_id: Session ID to filter chats.

        Returns:
            List of Chat objects.
        """

    @abstractmethod
    async def get_chat_with_messages(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> ChatMessage | None:
        """Retrieve a chat with all its messages.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            ChatMessage object or None if not found.
        """

    @abstractmethod
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

    @abstractmethod
    async def create_message(
        self,
        message: NewMessage,
    ) -> Message:
        """Create a new message.

        Args:
            message: NewMessage object containing message data.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            Created Message object.
        """

    @abstractmethod
    async def check_chat_exists(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Check if a chat exists in the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            True if chat exists, False otherwise.
        """

    @abstractmethod
    async def delete_chat(
        self,
        chat_id: str,
        session_id: str,
        user_id: str | None = None,
    ) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            session_id: Session ID for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
        """

    @abstractmethod
    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete all messages after a specific message in a chat.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID after which all messages will be deleted.
            user_id: User ID or fingerprint, depending on the prefix.
        """

    # NOTE: Might change, currently not used
    @abstractmethod
    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Message:
        """Update the content of a message.

        Args:
            message_id: Unique identifier for the message.
            data: New content for the message.
            user_id: User ID or fingerprint, depending on the prefix.
            session_id: Session ID for the chat.

        Returns:
            Updated Message object.
        """

    @abstractmethod
    async def get_next_ai_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> Message:
        """Get the next AI message after a specific message.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID to find the next AI message after.

        Returns:
            Next AI Message object.
        """
