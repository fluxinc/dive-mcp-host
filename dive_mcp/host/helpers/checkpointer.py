from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def get_checkpointer(uri: str) -> AsyncIterator[BaseCheckpointSaver]:
    """Get an appropriate async checkpointer based on the database connection string.

    Args:
        uri (str): Database connection string, starting with either 'sqlite' or
        'postgres'

    Raises:
        ValueError: If the database type in the connection string is not supported

    Returns:
        AsyncIterator[BaseCheckpointSaver[V]]: An async checkpointer instance for the
        specified database
    """
    if uri.startswith("sqlite"):
        path = uri.lstrip("sqlite:///")  # noqa: B005
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            yield saver
    elif uri.startswith("postgres"):
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            yield saver
    else:
        raise ValueError(f"Unsupported database: {uri}")
