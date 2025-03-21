from typing import Any

from fastapi import Request

from .app import app, kwargs_func


async def get_db_opts(request: Request) -> dict[str, Any]:
    """Get the database options."""
    return {"user_id": "auserid"}


kwargs_func["db_opts"] = get_db_opts


@app.get("/")
async def index(request: Request):
    """Get the database options."""
    return await request.state.get_kwargs("db_opts")
