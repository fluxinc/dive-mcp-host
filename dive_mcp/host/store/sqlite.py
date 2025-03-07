"""SQLite version of langgraph-checkpoint-postgres.

Use sqlite-vector-store for vector search.
"""
from langgraph.store.base.batch import AsyncBatchedBaseStore


class AsyncSQLiteStore(AsyncBatchedBaseStore):
    """Async SQLite store.

    from langgraph.store.sqlite import SQLiteStore

    conn_string = "sqlite:///./db.sqlite"

    async with AsyncSQLiteStore.from_conn_string(conn_string) as store:
        await store.setup()

        # Store and retrieve data
        await store.aput(("users", "123"), "prefs", {"theme": "dark"})
        item = await store.aget(("users", "123"), "prefs")

    """
