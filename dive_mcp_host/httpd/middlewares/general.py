from collections.abc import Callable
from typing import TypedDict

from starlette.requests import Request
from starlette.responses import Response

from dive_mcp_host.httpd.routers.models import ResultResponse, UserInputError


async def error_handler(request: Request, call_next: Callable) -> Response:
    """Error handling middleware.

    Args:
        request (Request): The request object.
        call_next (Callable): The next middleware to call.

    Returns:
        ResultResponse: The response object.
    """
    try:
        return await call_next(request)
    except UserInputError as e:
        return Response(
            status_code=400,
            content=ResultResponse(success=False, message=e.message).model_dump_json(
                by_alias=True
            ),
        )


class DiveUser(TypedDict):
    """User-related state storage.

    This state can be accessed by all middlewares and handlers.
    """

    user_id: str | None
    user_name: str | None
    user_type: str | None
    token_spent: int
    """The amount of tokens spent by the user in this period."""
    token_limit: int
    """The amount of tokens the user can use in this period."""
    token_increased: int
    """The amount of tokens increased in this request."""


async def default_state(request: Request, call_next: Callable) -> Response:
    """Prefill default state."""
    request.state.dive_user = DiveUser(
        user_id=None,
        user_name=None,
        user_type=None,
        token_spent=0,
        token_limit=0,
        token_increased=0,
    )
    return await call_next(request)
