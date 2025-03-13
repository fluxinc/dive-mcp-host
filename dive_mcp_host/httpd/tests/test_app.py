from collections.abc import AsyncGenerator
from http import HTTPStatus

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from ..app import app  # noqa: TID252


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[TestClient, None]:
    """Setup the app for testing."""
    yield TestClient(app)


@pytest.mark.asyncio
async def test_openai_root(client: TestClient) -> None:
    """Test the OpenAI root endpoint returns the expected welcome message."""
    response = client.get("/v1/openai")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "success": True,
        "message": "Welcome to Dive Compatible API! 🚀",
    }
