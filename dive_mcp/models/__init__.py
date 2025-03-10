"""Additional models for the MCP."""

from typing import Any

from langchain_core.language_models import BaseChatModel


def load_model(
    provider: str,
    model_name: str,
    *args: Any,
    **kwargs: Any,
) -> BaseChatModel:
    """Load a model from the models directory.

    Args:
        provider: provider name. Two special providers are supported:
            - "dive": use the model in dive_mcp.models
            - "__load__": load the model from the configuration
        model_name: The name of the model to load.
        args: Additional arguments to pass to the model.
        kwargs: Additional keyword arguments to pass to the model.

    Returns:
        The loaded model.

    If the provider is "dive", it should be like this:
        import dive_mcp.models.model_name_in_lower_case as model_module
        model = model_module.load_model(*args, **kwargs)
    If the provider is "__load__", the model_name is the class name of the model.
    For example, with model_name="package.module:ModelClass", it will be like this:
        import package.module as model_module
        model = model_module.ModelClass(*args, **kwargs)
    If the provider is neither "dive" nor "__load__", it will load model from langchain.
    """
    # XXX Pass configurations/parameters to the model
    raise NotImplementedError
