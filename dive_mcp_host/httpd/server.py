from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from logging import getLogger
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from dive_mcp_host.host.conf import HostConfig, ServerConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.httpd.abort_controller import AbortController
from dive_mcp_host.httpd.conf.mcpserver.manager import MCPServerManager
from dive_mcp_host.httpd.conf.model.manager import ModelManager
from dive_mcp_host.httpd.conf.prompts.manager import PromptManager
from dive_mcp_host.httpd.conf.service.manager import ServiceManager
from dive_mcp_host.httpd.database.migrate import db_migration
from dive_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp_host.httpd.database.msg_store.sqlite import SQLiteMessageStore
from dive_mcp_host.httpd.store.cache import LocalFileCache
from dive_mcp_host.httpd.store.local import LocalStore

logger = getLogger(__name__)


class DiveHostAPI(FastAPI):
    """DiveHostAPI is a FastAPI application that is used to host the DiveHost API."""

    dive_host: dict[str, DiveMcpHost]  # shoud init "default" when preapre stage

    def __init__(
        self,
        config_path: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the DiveHostAPI."""
        super().__init__(*args, **kwargs)
        self._config_path = config_path

    def _load_configs(self) -> None:
        """Load all configs."""
        self._service_config_manager = ServiceManager(self._config_path)
        self._service_config_manager.initialize()
        if self._service_config_manager.current_setting is None:
            raise ValueError("Service manager is not initialized")

        self._mcp_server_config_manager = MCPServerManager(
            self._service_config_manager.current_setting.config_location.mcp_server_config_path
        )
        self._mcp_server_config_manager.initialize()

        self._model_config_manager = ModelManager(
            self._service_config_manager.current_setting.config_location.model_config_path
        )
        self._model_config_manager.initialize()

        self._prompt_config_manager = PromptManager(
            self._service_config_manager.current_setting.config_location.prompt_config_path
        )

        self._abort_controller = AbortController()

    @asynccontextmanager
    async def prepare(self) -> AsyncGenerator[None, None]:
        """Setup the DiveHostAPI."""
        logger.info("Server Prepare")

        self._load_configs()

        # ================================================
        # Database
        # ================================================
        if self._service_config_manager.current_setting is None:
            raise ValueError("Service manager is not initialized")

        if self._service_config_manager.current_setting.db.migrate:
            db_migration(uri=self._service_config_manager.current_setting.db.uri)

        self._engine = create_async_engine(
            self._service_config_manager.current_setting.db.async_uri,
            echo=self._service_config_manager.current_setting.db.echo,
            # check connection before using
            pool_pre_ping=self._service_config_manager.current_setting.db.pool_pre_ping,
            # max connections
            pool_size=self._service_config_manager.current_setting.db.pool_size,
            # close connection after 60 seconds
            pool_recycle=self._service_config_manager.current_setting.db.pool_recycle,
            # burst connections
            max_overflow=self._service_config_manager.current_setting.db.max_overflow,
        )
        self._db_sessionmaker = async_sessionmaker(self._engine, class_=AsyncSession)
        self._msg_store = SQLiteMessageStore

        # ================================================
        # Store
        # ================================================
        self._store = LocalStore(
            self._service_config_manager.current_setting.upload_dir
        )
        self._local_file_cache = LocalFileCache(
            self._service_config_manager.current_setting.local_file_cache_prefix
        )

        # ================================================
        # Dive Host
        # ================================================
        if self._model_config_manager.current_setting is None:
            raise ValueError("Model config manager is not initialized")

        if self._mcp_server_config_manager.current_config is None:
            raise ValueError("MCPServer config manager is not initialized")

        config = HostConfig(
            llm=self._model_config_manager.current_setting,
            checkpointer=self._service_config_manager.current_setting.checkpointer,
            mcp_servers={
                server_name: ServerConfig(
                    name=server_name,
                    command=server_config.command or "",
                    args=server_config.args or [],
                    env=server_config.env or {},
                    enabled=server_config.enabled,
                    url=server_config.url or None,
                )
                for server_name, server_config in self._mcp_server_config_manager.get_enabled_servers().items()  # noqa: E501
            },
        )

        async with AsyncExitStack() as stack:
            default_host = DiveMcpHost(config)
            await stack.enter_async_context(default_host)
            self.dive_host = {"default": default_host}

            logger.info("Server Prepare Complete")
            yield

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

    @property
    def local_file_cache(self) -> LocalFileCache:
        """Get the local file cache."""
        return self._local_file_cache

    @property
    def service_manager(self) -> ServiceManager:
        """Get the service manager."""
        return self._service_config_manager

    @property
    def mcp_server_config_manager(self) -> MCPServerManager:
        """Get the MCP server config manager."""
        return self._mcp_server_config_manager

    @property
    def model_config_manager(self) -> ModelManager:
        """Get the model config manager."""
        return self._model_config_manager

    @property
    def prompt_config_manager(self) -> PromptManager:
        """Get the prompt config manager."""
        return self._prompt_config_manager

    @property
    def abort_controller(self) -> AbortController:
        """Get the abort controller."""
        return self._abort_controller
