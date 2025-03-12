from sqlalchemy.ext.asyncio import AsyncSession

from .message_store_abc import AbstractMessageStore
from .models import (
    Chat,
    ChatMessage,
    Message,
    QueryInput,
    ResourceUsage,
)
from .orm_models import (
    Chat as ORMChat,
)
from .orm_models import (
    Message as ORMMessage,
)


class MessageStore(AbstractMessageStore):
    """Message store."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the message store.

        Args:
            session: SQLAlchemy async session.
        """
        self.session = session

    async def get_all_chats(
        self,
        user_id: str,
    ) -> list[Chat]:
        """Retrieve all chats from the database.

        Args:
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            List of Chat objects.
        """
        raise NotImplementedError

    async def get_chat_with_messages(
        self,
        chat_id: str,
        user_id: str,
    ) -> ChatMessage | None:
        """Retrieve a chat with all its messages.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            ChatMessage object or None if not found.
        """
        raise NotImplementedError

    async def create_chat(
        self,
        chat_id: str,
        title: str,
        user_id: str,
    ) -> Chat | None:
        """Create a new chat.

        Args:
            chat_id: Unique identifier for the chat.
            title: Title of the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            Created Chat object or None if creation failed.
        """
        raise NotImplementedError

    async def create_message(
        self,
        message: Message,
        user_id: str,
        resource_usage: ResourceUsage,
    ) -> Message:
        """Create a new message.

        Args:
            message: NewMessage object containing message data.
            user_id: User ID or fingerprint, depending on the prefix.
            resource_usage: Resource usage information.

        Returns:
            Created Message object.
        """
        raise NotImplementedError

    async def check_chat_exists(
        self,
        chat_id: str,
        user_id: str,
    ) -> bool:
        """Check if a chat exists in the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            True if chat exists, False otherwise.
        """
        raise NotImplementedError

    async def delete_chat(self, chat_id: str, user_id: str) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
        """
        raise NotImplementedError

    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
        user_id: str,
    ) -> None:
        """Delete all messages after a specific message in a chat.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID after which all messages will be deleted.
            user_id: User ID or fingerprint, depending on the prefix.
        """

    # NOTE: Currently not used
    #       Arguments might need to change, uncertain about the usecase
    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        user_id: str,
        resource_usage: ResourceUsage,
    ) -> Message:
        """Update the content of a message.

        Args:
            message_id: Unique identifier for the message.
            data: New content for the message.
            user_id: User ID or fingerprint, depending on the prefix.
            resource_usage: Resource usage information.

        Returns:
            Updated Message object.
        """
        raise NotImplementedError

    # NOTE: Currently not used
    #       Arguments might need to change, uncertain about the usecase
    async def get_next_ai_message(self, chat_id: str, message_id: str) -> Message:
        """Get the next AI message after a specific message.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID to find the next AI message after.

        Returns:
            Next AI Message object.
        """
        raise NotImplementedError
