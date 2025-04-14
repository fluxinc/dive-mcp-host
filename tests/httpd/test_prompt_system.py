from langchain_core.prompts import ChatPromptTemplate

from dive_mcp_host.httpd.conf.system_prompt import system_prompt


def test_system_prompt():
    """Test the system prompt."""
    prompt_txt = system_prompt("")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_txt),
            ("placeholder", "{messages}"),
        ],
    )
    assert prompt.input_variables == []
    assert prompt.optional_variables == ["messages"]
