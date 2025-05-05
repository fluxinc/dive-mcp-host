import json
import re
import uuid
from logging import getLogger

from langchain_core.messages import AIMessage, ToolCall

logger = getLogger(__name__)


def extract_tool_calls(response: AIMessage) -> AIMessage:
    """Extract the tool calls from the response content."""
    if isinstance(response.content, str):
        # Extract the tool call content
        tool_call_content = r"<tool_call>(.*?)</tool_call>"
        regex = re.compile(tool_call_content, re.DOTALL)
        matches = regex.findall(response.content)

        # Transform the tool call content into a ToolCall
        for match in matches:
            tool_call = json.loads(match.strip())
            tool_call_id = str(uuid.uuid4())
            tool_call = ToolCall(
                name=tool_call["name"],
                args=tool_call["arguments"],
                id=tool_call_id,
            )
            logger.debug("found tool call: %s", tool_call)
            response.content = response.content.replace(
                f"<tool_call>\n{match}\n</tool_call>", ""
            )
            response.tool_calls.append(tool_call)

    else:
        logger.warning(
            "Response content is not a string, cannot extract tool calls: %s",
            response.content,
        )

    return response
