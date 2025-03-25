from os import getenv

POSTGRES_URI = getenv("POSTGRES_URI", "postgresql://mcp:mcp@localhost:5432/mcp")
SQLITE_URI = getenv("SQLITE_URI", "sqlite:///dummy.db")

POSTGRES_URI_ASYNC = getenv(
    "POSTGRES_URI_ASYNC",
    "postgresql+asyncpg://mcp:mcp@localhost:5432/mcp",
)
SQLITE_URI_ASYNC = getenv("SQLITE_URI_ASYNC", "sqlite+aiosqlite:///dummy.db")


def dict_subset(superset: dict, subset: dict) -> None:
    """Check if subset is a subset of superset.

    Args:
        superset: The dictionary that should contain all items from subset
        subset: The dictionary that should be contained within superset

    Returns:
        bool: True if subset is a subset of superset, False otherwise
    """
    assert isinstance(superset, dict)
    assert isinstance(subset, dict)
    for key, value in subset.items():
        assert key in superset, f"{key} is not in {superset}"
        if isinstance(value, dict) and isinstance(superset[key], dict):
            assert dict_subset(superset[key], value) is None, (
                f"{key} is not a subset of {superset[key]}"
            )
        elif isinstance(value, list) and isinstance(superset[key], list):
            assert len(value) == len(superset[key]), (
                f"List lengths for {key} are different: {len(value)} != {len(superset[key])}"  # noqa: E501
            )
            for i, (subset_item, superset_item) in enumerate(
                zip(value, superset[key], strict=False)
            ):
                if isinstance(subset_item, dict) and isinstance(superset_item, dict):
                    assert dict_subset(superset_item, subset_item) is None, (
                        f"List item {i} in {key} is not a subset"
                    )
                else:
                    assert subset_item == superset_item, (
                        f"List item {i} in {key} are not equal: {subset_item} != {superset_item}"  # noqa: E501
                    )
        else:
            assert superset[key] == value, (
                f"{key} is not equal {superset[key]} and {value}"
            )
