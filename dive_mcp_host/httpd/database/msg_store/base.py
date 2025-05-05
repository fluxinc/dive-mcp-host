import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, desc, exists, func, insert, select, update
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
from dive_mcp_host.httpd.database.orm_models import (
    ResourceUsage as ORMResourceUsage,
)
from dive_mcp_host.httpd.routers.models import SortBy

from .abstract import AbstractMessageStore

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        user_id: str | None = None,
        session_id: str | None = None,
        sort_by: SortBy = SortBy.CHAT,
    ) -> list[Chat]:
        """Retrieve all chats from the database.

        Args:
            user_id: User ID or fingerprint, depending on the prefix.
            session_id: Session ID to filter chats.
            sort_by: Sort by.
                - 'chat': Sort by chat creation time.
                - 'msg': Sort by message creation time.
                default: 'chat'

        Returns:
            List of Chat objects.
        """
        filters = []
        if user_id is not None:
            filters.append(ORMChat.user_id == user_id)
        if session_id is not None:
            filters.append(ORMChat.session_id == session_id)

        if sort_by == SortBy.MESSAGE:
            query = (
                select(
                    ORMChat,
                    func.coalesce(
                        func.max(ORMMessage.created_at), ORMChat.created_at
                    ).label("last_message_at"),
                )
                .outerjoin(ORMMessage, ORMChat.id == ORMMessage.chat_id)
                .group_by(
                    ORMChat.id,
                    ORMChat.title,
                    ORMChat.created_at,
                    ORMChat.user_id,
                    ORMChat.session_id,
                )
                .where(*filters)
                .order_by(desc("last_message_at"))
            )
            result = await self._session.execute(query)
            chats: Sequence[ORMChat] = result.scalars().all()

        elif sort_by == SortBy.CHAT:
            query = (
                select(ORMChat)
                .where(*filters)
                .order_by(desc(ORMChat.created_at))
            )
            result = await self._session.scalars(query)
            chats: Sequence[ORMChat] = result.all()

        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}")

        return [
            Chat(
                id=chat.id,
                title=chat.title,
                createdAt=chat.created_at,
                user_id=chat.user_id,
                session_id=chat.session_id,
            )
            for chat in chats
        ]

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
            session_id=data.session_id,
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
                    id=msg.id,
                    createdAt=msg.created_at,
                    content=msg.content,
                    role=Role(msg.role),
                    chatId=msg.chat_id,
                    messageId=msg.message_id,
                    files=json.loads(msg.files) if msg.files else [],
                    tool_calls=msg.tool_calls or [],
                    resource_usage=resource_usage,
                ),
            )
        return ChatMessage(chat=chat, messages=messages)

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
                    "message_id": message.message_id,
                    "files": json.dumps(message.files),
                    "tool_calls": message.tool_calls,
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
                        "message_id": message.message_id,
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
            id=new_msg.id,
            createdAt=new_msg.created_at,
            content=new_msg.content,
            role=Role(new_msg.role),
            chatId=new_msg.chat_id,
            messageId=new_msg.message_id,
            files=json.loads(new_msg.files),
            tool_calls=new_msg.tool_calls or [],
            resource_usage=resource_usage,
        )

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
        query = (
            exists(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
            .select()
        )
        exist = await self._session.scalar(query)
        return bool(exist)

    async def delete_chat(
        self, 
        chat_id: str,
        session_id: str,
        user_id: str | None = None,
    ) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
            session_id: Session ID for the chat.
        """
        query = delete(ORMChat).where(ORMChat.id == chat_id)
        
        # Add user_id filter if provided
        if user_id is not None:
            query = query.where(ORMChat.user_id == user_id)
            
        # Add session_id filter if provided
        if session_id is not None:
            query = query.where(ORMChat.session_id == session_id)
            
        await self._session.execute(query)

    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete all messages after a specific message in a chat."""
        query = (
            delete(ORMMessage)
            .where(ORMMessage.chat_id == chat_id)
            .where(
                ORMMessage.created_at
                > (
                    select(ORMMessage.created_at)
                    .where(ORMMessage.chat_id == chat_id)
                    .where(ORMMessage.message_id == message_id)
                    .scalar_subquery()
                )
            )
        )
        await self._session.execute(query)

    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        session_id: str,
        user_id: str | None = None
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
        # Prepare files list
        files = []
        if data.images:
            files.extend(data.images)
        if data.documents:
            files.extend(data.documents)

        # Build the base query
        query = (
            update(ORMMessage)
            .where(ORMMessage.message_id == message_id)
            .where(ORMMessage.chat_id == ORMChat.id)
        )
        
        # Add user_id filter if provided
        if user_id is not None:
            query = query.where(ORMChat.user_id == user_id)
            
        # Add session_id filter if provided
        if session_id is not None:
            query = query.where(ORMChat.session_id == session_id)
            
        # Update the message content and files
        query = (
            query.values(
                content=data.text or "",
                files=json.dumps(files) if files else "",
                tool_calls=data.tool_calls,
            )
            .returning(ORMMessage)
            .options(selectinload(ORMMessage.resource_usage))
        )
        
        updated_message = await self._session.scalar(query)
        if updated_message is None:
            raise ValueError(f"Message {message_id} not found or access denied")

        resource_usage = (
            ResourceUsage.model_validate(
                updated_message.resource_usage,
                from_attributes=True,
            )
            if updated_message.resource_usage is not None
            else None
        )
        return Message(
            id=updated_message.id,
            createdAt=updated_message.created_at,
            content=updated_message.content,
            role=Role(updated_message.role),
            chatId=updated_message.chat_id,
            messageId=updated_message.message_id,
            files=json.loads(updated_message.files) if updated_message.files else [],
            tool_calls=updated_message.tool_calls or [],
            resource_usage=resource_usage,
        )

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
        query = (
            select(ORMMessage)
            .where(ORMMessage.message_id == message_id)
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
            .where(ORMMessage.id > user_message.id)
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
            id=message.id,
            createdAt=message.created_at,
            content=message.content,
            role=Role(message.role),
            chatId=message.chat_id,
            messageId=message.message_id,
            files=json.loads(message.files) if message.files else [],
            tool_calls=message.tool_calls or [],
            resource_usage=resource_usage,
        )
