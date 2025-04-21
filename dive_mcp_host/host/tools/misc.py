from typing import Any

from langchain_core.tools import BaseTool


class TestTool(BaseTool):
    """Test tool."""

    name: str = "weather_tool"
    description: str = "get weather information for a specific city"  # noqa: E501
    called: bool = False

    def _run(self, *_args: Any, **_kwargs: Any) -> Any:
        self.called = True
        return {"result": "success"}
