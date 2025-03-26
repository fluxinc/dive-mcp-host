import tempfile
from collections.abc import Generator

import pytest


@pytest.fixture
def sqlite_uri() -> Generator[str, None, None]:
    """Create a temporary SQLite URI."""
    with tempfile.NamedTemporaryFile(
        prefix="testServiceConfig_", suffix=".json"
    ) as service_config_file:
        yield f"sqlite:///{service_config_file.name}"
