import pytest

from dive_mcp_host.httpd.store.cache import CacheKeys, LocalFileCache


@pytest.fixture
def cleanup_cache():
    """Cleanup the cache."""
    cache = LocalFileCache()
    cache.delete(CacheKeys.LIST_TOOLS)
    try:
        yield
    finally:
        cache.delete(CacheKeys.LIST_TOOLS)


def test_local_file_cache(cleanup_cache):
    """Test the local file cache."""
    cache = LocalFileCache()

    assert not cache.get_cache_file_path(CacheKeys.LIST_TOOLS).exists()
    assert cache.get(CacheKeys.LIST_TOOLS) is None

    cache.set(CacheKeys.LIST_TOOLS, "test")
    assert cache.get(CacheKeys.LIST_TOOLS) == "test"
    assert cache.get_cache_file_path(CacheKeys.LIST_TOOLS).exists()
