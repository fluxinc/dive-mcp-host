from logging import getLogger
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.httpd.database.migrate import db_migration
from dive_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp_host.httpd.database.msg_store.sqlite import SQLiteMessageStore
from dive_mcp_host.httpd.store.local import LocalStore

logger = getLogger(__name__)


class DiveHostAPI(FastAPI):
    """DiveHostAPI is a FastAPI application that is used to host the DiveHost API."""

    dive_host: dict[str, DiveMcpHost]  # shoud init "default" when preapre stage

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DiveHostAPI."""
        super().__init__(*args, **kwargs)

    async def prepare(self) -> None:
        """Setup the DiveHostAPI."""
        logger.info("Server Prepare")

        # NOTE: migration should be optional
        db_migration(uri="sqlite:///db.sqlite")

        self._engine = create_async_engine(
            "sqlite+aiosqlite:///db.sqlite",
            echo=False,
            pool_pre_ping=True,  # check connection before using
            pool_size=5,  # max connections
            pool_recycle=60,  # close connection after 60 seconds
            max_overflow=10,  # burst connections
        )
        self._db_sessionmaker = async_sessionmaker(self._engine, class_=AsyncSession)
        self._msg_store = SQLiteMessageStore

        self._store = LocalStore()

        logger.info("Server Prepare Complete")

    async def ready(self) -> bool:
        """Ready the DiveHostAPI."""
        try:
            # check db connection
            async with self._db_sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.exception("Server not ready")
            return False

    async def cleanup(self) -> None:
        """Cleanup the DiveHostAPI."""
        logger.info("Server Cleanup")
        await self._engine.dispose()
        logger.info("Server Cleanup Complete")

    @property
    def db_sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """Get the database sessionmaker."""
        return self._db_sessionmaker

    @property
    def msg_store(self) -> type[BaseMessageStore]:
        """Get message store type."""
        return self._msg_store

    @property
    def store(self) -> LocalStore:
        """Get the store."""
        return self._store
