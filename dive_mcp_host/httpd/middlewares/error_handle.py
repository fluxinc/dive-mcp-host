from collections.abc import Callable
from logging import getLogger

from starlette.requests import Request
from starlette.responses import Response

from ..routers import ResultResponse, UserInputError  # noqa: TID252

logger = getLogger(__name__)


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
        logger.exception("API exception")
        return Response(
            status_code=400,
            content=ResultResponse(success=False, message=e.message),
        )
