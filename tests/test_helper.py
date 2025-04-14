import pytest

from .helper import dict_subset


def test_dict_subset():
    """Test the dict_subset function."""
    dict_subset({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})
    dict_subset({"a": 1, "b": {"a": 1, "b": 2}}, {"a": 1, "b": {"a": 1}})

    # Test cases for lists
    dict_subset({"a": [1, 2], "b": [3, 4]}, {"a": [1, 2]})
    dict_subset({"a": [{"x": 1, "y": 3}]}, {"a": [{"x": 1}]})
    with pytest.raises(AssertionError):
        dict_subset({"a": [1, 2, 3]}, {"a": [1, 2]})
    with pytest.raises(AssertionError):
        dict_subset({"a": [{"x": 1}, {"y": 2}]}, {"a": [{"x": 1}]})
