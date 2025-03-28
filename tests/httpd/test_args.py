from pathlib import Path

from dive_mcp_host.httpd.conf.arguments import Arguments


def test_argparser():
    """Test the argument parser."""
    args = Arguments.parse_args(["--config", "config.json"])
    should_be = Arguments(
        config=Path("config.json"),
        model_config=Path.cwd().joinpath("model_config.json"),
        mcp_config=Path.cwd().joinpath("mcp_config.json"),
    )

    # Test with default values
    assert args.httpd_config == should_be.httpd_config
    assert args.llm_config == should_be.llm_config
    assert args.mcp_config == should_be.mcp_config
    assert args.custom_rules == should_be.custom_rules
    assert args.command_alias_config == should_be.command_alias_config
    assert args.auto_reload is False

    # Test with custom values
    custom_args = [
        "--config",
        "custom_config.json",
        "--model_config",
        "custom_model_config.json",
        "--mcp_config",
        "custom_mcp_config.json",
        "--custom_rules",
        "custom_rules_dir",
        "--command_alias_config",
        "custom_command_alias.json",
    ]
    args = Arguments.parse_args(custom_args)
    assert args.httpd_config == Path("custom_config.json")
    assert args.llm_config == Path("custom_model_config.json")
    assert args.mcp_config == Path("custom_mcp_config.json")
    assert args.custom_rules == Path("custom_rules_dir")
    assert args.command_alias_config == Path("custom_command_alias.json")
