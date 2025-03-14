from datetime import datetime

from sqlalchemy import (
    CHAR,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Users(Base):
    """Users model.

    Attributes:
        id: User ID or fingerprint, depending on the prefix.
    """

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    user_type: Mapped[str | None] = mapped_column(CHAR(10))

    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user",
        passive_deletes=True,
        uselist=True,
    )


class Chat(Base):
    """Chat model.

    Attributes:
        id: Chat ID.
        title: Chat title.
        created_at: Chat creation timestamp.
        user_id: User ID or fingerprint, depending on the prefix.
    """

    __tablename__ = "chats"
    __table_args__ = (Index("idx_chats_user_id", "user_id", postgresql_using="hash"),)
    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    title: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        passive_deletes=True,
        uselist=True,
    )
    user: Mapped["Users"] = relationship(foreign_keys=user_id, back_populates="chats")


class Message(Base):
    """Message model.

    Attributes:
        id: Message ID.
        created_at: Message creation timestamp.
        content: Message content.
        role: Message role.
        chat_id: Chat ID.
        message_id: Message ID.
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("messages_message_id_index", "message_id", postgresql_using="hash"),
        Index("idx_messages_chat_id", "chat_id", postgresql_using="hash"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text())
    role: Mapped[str] = mapped_column(Text())
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    message_id: Mapped[str] = mapped_column(Text(), unique=True)

    chat: Mapped["Chat"] = relationship(foreign_keys=chat_id, back_populates="messages")
    resource_usage: Mapped["ResourceUsage"] = relationship(
        back_populates="message",
    )


class ResourceUsage(Base):
    """Resource usage model.

    Attributes:
        id: Resource usage ID.
        message_id: Message ID.
        model: Model name.
        total_input_tokens: Total input tokens.
        total_output_tokens: Total output tokens.
        total_run_time: Total run time.
    """

    __tablename__ = "resource_usage"
    __table_args__ = (Index("idx_resource_usage_message_id", "message_id"),)
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.message_id", ondelete="CASCADE"),
    )
    model: Mapped[str] = mapped_column(Text())
    total_input_tokens: Mapped[int] = mapped_column(BigInteger())
    total_output_tokens: Mapped[int] = mapped_column(BigInteger())
    total_run_time: Mapped[float] = mapped_column(Float())

    message: Mapped["Message"] = relationship(
        foreign_keys=message_id,
        back_populates="resource_usage",
    )
