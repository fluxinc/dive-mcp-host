from datetime import UTC, datetime

from sqlalchemy import delete, exists, insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Message as ORMMessage
from dive_mcp_host.httpd.database.orm_models import ResourceUsage as ORMResourceUsage

from .abstract import AbstractMessageStore


class BaseMessageStore(AbstractMessageStore):
    """Base Message store.

    Contains queries that can function in both SQLite and PostgreSQL
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the message store.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

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
        query = (
            select(ORMChat)
            .where(ORMChat.user_id == user_id)
            .order_by(ORMChat.created_at.desc())
        )
        chats = await self._session.scalars(query)
        return [
            Chat(
                id=chat.id,
                title=chat.title,
                createdAt=chat.created_at,
                user_id=chat.user_id,
            )
            for chat in chats
        ]

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
        query = (
            select(ORMChat)
            .options(
                selectinload(ORMChat.messages).selectinload(ORMMessage.resource_usage),
            )
            .where(ORMChat.user_id == user_id)
            .where(ORMChat.id == chat_id)
            .order_by(ORMChat.created_at.desc())
        )
        data = await self._session.scalar(query)
        if data is None:
            return None

        chat = Chat(
            id=data.id,
            title=data.title,
            createdAt=data.created_at,
            user_id=data.user_id,
        )
        messages: list[Message] = []
        for msg in data.messages:
            resource_usage = (
                ResourceUsage.model_validate(
                    msg.resource_usage,
                    from_attributes=True,
                )
                if msg.resource_usage is not None
                else None
            )
            messages.append(
                Message(
                    createdAt=msg.created_at,
                    content=msg.content,
                    role=Role(msg.role),
                    chatId=msg.chat_id,
                    messageId=msg.id,
                    resource_usage=resource_usage,
                ),
            )
        return ChatMessage(chat=chat, messages=messages)

    async def create_chat(
        self,
        chat_id: str,
        title: str,
        user_id: str,
        user_type: str | None = None,
    ) -> Chat | None:
        """Create a new chat.

        Args:
            chat_id: Unique identifier for the chat.
            title: Title of the chat.
            user_id: User ID or fingerprint, depending on the prefix.
            user_type: Optional user type.

        Returns:
            Created Chat object or None if creation failed.
        """
        raise NotImplementedError(
            "The implementation of the method varies on different database.",
        )

    async def create_message(self, message: NewMessage) -> Message:
        """Create a new message.

        Args:
            message: NewMessage object containing message data.

        Returns:
            Created Message object.
        """
        query = (
            insert(ORMMessage)
            .values(
                {
                    "created_at": datetime.now(UTC),
                    "content": message.content,
                    "role": message.role,
                    "chat_id": message.chat_id,
                    "id": message.id,
                },
            )
            .returning(ORMMessage)
        )
        new_msg = await self._session.scalar(query)
        if new_msg is None:
            raise Exception(f"Create message failed: {message}")

        # NOTE: Only LLM messages will have resource usage
        new_resource_usage = None
        if message.role == Role.ASSISTANT and message.resource_usage is not None:
            query = (
                insert(ORMResourceUsage)
                .values(
                    {
                        "id": message.id,
                        "model": message.resource_usage.model,
                        "total_input_tokens": message.resource_usage.total_input_tokens,
                        "total_output_tokens": message.resource_usage.total_output_tokens,  # noqa: E501
                        "total_run_time": message.resource_usage.total_run_time,
                    },
                )
                .returning(ORMResourceUsage)
            )
            new_resource_usage = await self._session.scalar(query)
            if new_resource_usage is None:
                raise Exception(f"Create resource usage failed: {message}")

        resource_usage = (
            ResourceUsage.model_validate(
                new_resource_usage,
                from_attributes=True,
            )
            if new_resource_usage is not None
            else None
        )
        return Message(
            createdAt=new_msg.created_at,
            content=new_msg.content,
            role=Role(new_msg.role),
            chatId=new_msg.chat_id,
            messageId=new_msg.id,
            resource_usage=resource_usage,
        )

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
        query = (
            exists(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
            .select()
        )
        exist = await self._session.scalar(query)
        return bool(exist)

    async def delete_chat(self, chat_id: str, user_id: str) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
        """
        query = (
            delete(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
        )
        await self._session.execute(query)

    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete all messages after a specific message in a chat."""
        query = (
            select(ORMMessage)
            .where(ORMMessage.id == message_id)
            .where(ORMMessage.chat_id == chat_id)
        )
        ancher_msg = await self._session.scalar(query)

        if ancher_msg is not None:
            query = (
                delete(ORMMessage)
                .where(ORMMessage.chat_id == chat_id)
                .where(ORMMessage.created_at > ancher_msg.created_at)
            )
            await self._session.execute(query)

    # NOTE: Might change, currently not used
    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        user_id: str,
    ) -> Message:
        """Update the content of a message.

        Args:
            message_id: Unique identifier for the message.
            data: New content for the message.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            Updated Message object.
        """
        raise NotImplementedError

    async def get_next_ai_message(self, chat_id: str, message_id: str) -> Message:
        """Get the next AI message after a specific message.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID to find the next AI message after.

        Returns:
            Next AI Message object.
        """
        query = (
            select(ORMMessage)
            .where(ORMMessage.id == message_id)
            .where(ORMMessage.role == Role.USER)
        )
        user_message = await self._session.scalar(query)
        if user_message is None:
            raise ValueError("Can only get next AI message for user messages")

        query = (
            select(ORMMessage)
            .options(
                selectinload(ORMMessage.resource_usage),
            )
            .where(ORMMessage.chat_id == chat_id)
            .where(ORMMessage.created_at > user_message.created_at)
            .where(ORMMessage.role == Role.ASSISTANT)
            .limit(1)
        )
        message = await self._session.scalar(query)
        if not message:
            raise ValueError(
                f"No AI message found after user message ${message_id}."
                "This indicates a data integrity issue.",
            )

        resource_usage = (
            ResourceUsage.model_validate(
                message.resource_usage,
                from_attributes=True,
            )
            if message.resource_usage is not None
            else None
        )
        return Message(
            createdAt=message.created_at,
            content=message.content,
            role=Role(message.role),
            chatId=message.chat_id,
            messageId=message.id,
            resource_usage=resource_usage,
        )
