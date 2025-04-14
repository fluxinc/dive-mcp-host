from typing import Any

from langchain_core.tools import BaseTool


class TestTool(BaseTool):
    """Test tool."""

    name: str = "test_tool"
    description: str = "a simple test tool check tool functionality call it any name with any arguments, returns nothing"  # noqa: E501
    called: bool = False

    def _run(self, *_args: Any, **_kwargs: Any) -> Any:
        self.called = True
        return {"result": "success"}
