from .helper import dict_subset


def test_dict_subset():
    """Test the dict_subset function."""
    assert dict_subset({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})
    assert dict_subset({"a": 1, "b": {"a": 1, "b": 2}}, {"a": 1, "b": {"a": 1}})
