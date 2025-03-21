from os import getenv

POSTGRES_URI = getenv("POSTGRES_URI", "postgresql://mcp:mcp@localhost:5432/mcp")
SQLITE_URI = getenv("SQLITE_URI", "sqlite:///dummy.db")

POSTGRES_URI_ASYNC = getenv(
    "POSTGRES_URI_ASYNC",
    "postgresql+asyncpg://mcp:mcp@localhost:5432/mcp",
)
SQLITE_URI_ASYNC = getenv("SQLITE_URI_ASYNC", "sqlite+aiosqlite:///dummy.db")
