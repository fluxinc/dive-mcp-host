from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import Response

from ..routers import ResultResponse, UserInputError  # noqa: TID252


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
            content=ResultResponse(success=False, message=e.message),
        )


async def default_state(request: Request, call_next: Callable) -> Response:
    """Prefill default state."""
    request.state.dive_user = {}
    return await call_next(request)
