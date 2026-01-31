"""Redis 客户端测试

测试 Redis 连接池、缓存和分布式锁功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.redis import (
    RedisCache,
    RedisLock,
    close_redis,
    get_cache,
    get_redis,
    ping,
)


class TestRedisConnection:
    """Redis 连接测试"""

    @patch("app.infra.redis._get_pool")
    async def test_get_redis(self, mock_get_pool) -> None:
        """测试获取 Redis 客户端"""
        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool

        client = await get_redis()

        assert client.connection_pool is mock_pool

    @patch("app.infra.redis._pool")
    async def test_close_redis(self, mock_pool) -> None:
        """测试关闭 Redis 连接"""
        mock_pool.aclose = AsyncMock()

        await close_redis()

        mock_pool.aclose.assert_called_once()

    @patch("app.infra.redis.get_redis")
    async def test_ping_success(self, mock_get_redis) -> None:
        """测试 Ping 成功"""
        mock_client = MagicMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_client

        result = await ping()

        assert result is True

    @patch("app.infra.redis.get_redis")
    async def test_ping_failure(self, mock_get_redis) -> None:
        """测试 Ping 失败"""
        from redis.exceptions import RedisError

        mock_client = MagicMock()
        mock_client.ping = AsyncMock(side_effect=RedisError("Connection lost"))
        mock_get_redis.return_value = mock_client

        result = await ping()

        assert result is False


class TestRedisCache:
    """RedisCache 测试"""

    @pytest.fixture
    def cache(self) -> RedisCache:
        """创建缓存实例"""
        return RedisCache(key_prefix="test:")

    def test_make_key(self, cache) -> None:
        """测试键生成"""
        assert cache._make_key("user:123") == "test:user:123"

    @patch("app.infra.redis.get_redis")
    async def test_get(self, mock_get_redis, cache) -> None:
        """测试获取缓存"""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value='"cached_value"')
        mock_get_redis.return_value = mock_client

        result = await cache.get("key")

        assert result == '"cached_value"'
        mock_client.get.assert_called_once_with("test:key")

    @patch("app.infra.redis.get_redis")
    async def test_get_not_found(self, mock_get_redis, cache) -> None:
        """测试获取不存在的键"""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=None)
        mock_get_redis.return_value = mock_client

        result = await cache.get("nonexistent")

        assert result is None

    @patch("app.infra.redis.get_redis")
    async def test_set_without_ttl(self, mock_get_redis, cache) -> None:
        """测试设置缓存（无过期时间）"""
        mock_client = MagicMock()
        mock_client.set = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_client

        result = await cache.set("key", "value")

        assert result is True
        mock_client.set.assert_called_once_with("test:key", "value")

    @patch("app.infra.redis.get_redis")
    async def test_set_with_ttl(self, mock_get_redis, cache) -> None:
        """测试设置缓存（有过期时间）"""
        mock_client = MagicMock()
        mock_client.setex = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_client

        result = await cache.set("key", "value", ttl=60)

        assert result is True
        mock_client.setex.assert_called_once_with("test:key", 60, "value")

    @patch("app.infra.redis.get_redis")
    async def test_delete(self, mock_get_redis, cache) -> None:
        """测试删除缓存"""
        mock_client = MagicMock()
        mock_client.delete = AsyncMock(return_value=2)
        mock_get_redis.return_value = mock_client

        result = await cache.delete("key1", "key2")

        assert result == 2

    @patch("app.infra.redis.get_redis")
    async def test_delete_empty(self, mock_get_redis, cache) -> None:
        """测试删除空列表"""
        result = await cache.delete()
        assert result == 0

    @patch("app.infra.redis.get_redis")
    async def test_exists(self, mock_get_redis, cache) -> None:
        """测试检查键是否存在"""
        mock_client = MagicMock()
        mock_client.exists = AsyncMock(return_value=2)
        mock_get_redis.return_value = mock_client

        result = await cache.exists("key1", "key2", "key3")

        assert result == 2

    @patch("app.infra.redis.get_redis")
    async def test_expire(self, mock_get_redis, cache) -> None:
        """测试设置过期时间"""
        mock_client = MagicMock()
        mock_client.expire = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_client

        result = await cache.expire("key", 300)

        assert result is True

    @patch("app.infra.redis.get_redis")
    async def test_incr(self, mock_get_redis, cache) -> None:
        """测试递增"""
        mock_client = MagicMock()
        mock_client.incr = AsyncMock(return_value=5)
        mock_get_redis.return_value = mock_client

        result = await cache.incr("counter")

        assert result == 5

    @patch("app.infra.redis.get_redis")
    async def test_incr_by(self, mock_get_redis, cache) -> None:
        """测试按值递增"""
        mock_client = MagicMock()
        mock_client.incrby = AsyncMock(return_value=15)
        mock_get_redis.return_value = mock_client

        result = await cache.incr("counter", delta=10)

        assert result == 15

    @patch("app.infra.redis.get_redis")
    async def test_ttl(self, mock_get_redis, cache) -> None:
        """测试获取过期时间"""
        mock_client = MagicMock()
        mock_client.ttl = AsyncMock(return_value=300)
        mock_get_redis.return_value = mock_client

        result = await cache.ttl("key")

        assert result == 300


class TestRedisLock:
    """RedisLock 测试"""

    @pytest.fixture
    def lock(self) -> RedisLock:
        """创建锁实例"""
        return RedisLock("my_resource")

    def test_init(self, lock) -> None:
        """测试初始化"""
        assert lock.resource == "my_resource"
        assert lock.key == "kiki:lock:my_resource"
        assert lock._token is None

    @patch("app.infra.redis.get_redis")
    async def test_acquire_success(self, mock_get_redis, lock) -> None:
        """测试获取锁成功"""
        mock_client = MagicMock()
        mock_client.set = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_client

        result = await lock.acquire(ttl=30)

        assert result is True
        assert lock._token is not None
        mock_client.set.assert_called_once()

    @patch("app.infra.redis.get_redis")
    async def test_acquire_failure(self, mock_get_redis, lock) -> None:
        """测试获取锁失败"""
        mock_client = MagicMock()
        mock_client.set = AsyncMock(return_value=False)
        mock_get_redis.return_value = mock_client

        result = await lock.acquire(ttl=30)

        assert result is False

    @patch("app.infra.redis.get_redis")
    async def test_release_success(self, mock_get_redis, lock) -> None:
        """测试释放锁成功"""
        lock._token = "test_token"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value="test_token")
        mock_client.eval = AsyncMock(return_value=1)
        mock_get_redis.return_value = mock_client

        result = await lock.release()

        assert result is True

    @patch("app.infra.redis.get_redis")
    async def test_release_no_token(self, mock_get_redis, lock) -> None:
        """测试无 token 时释放"""
        result = await lock.release()
        assert result is False

    @patch("app.infra.redis.get_redis")
    async def test_release_token_mismatch(self, mock_get_redis, lock) -> None:
        """测试 token 不匹配时释放"""
        lock._token = "my_token"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value="other_token")
        mock_client.eval = AsyncMock(return_value=0)
        mock_get_redis.return_value = mock_client

        result = await lock.release()

        assert result is False


class TestGetCache:
    """get_cache 工厂函数测试"""

    def test_get_cache_default_prefix(self) -> None:
        """测试默认前缀"""
        cache1 = get_cache()
        cache2 = get_cache()

        # 应该返回同一个实例（lru_cache）
        assert cache1 is cache2
        assert cache1.key_prefix == "kiki:cache:"

    def test_get_cache_custom_prefix(self) -> None:
        """测试自定义前缀"""
        cache = get_cache(key_prefix="custom:")

        assert cache.key_prefix == "custom:"


@pytest.mark.parametrize("delta,expected", [
    (1, 5),
    (5, 9),
    (-1, 3),
])
async def test_redis_cache_incr_variations(delta: int, expected: int) -> None:
    """参数化测试递增变化"""
    cache = RedisCache()

    with patch("app.infra.redis.get_redis") as mock_get_redis:
        mock_client = MagicMock()
        if delta == 1:
            mock_client.incr = AsyncMock(return_value=expected)
        else:
            mock_client.incrby = AsyncMock(return_value=expected)
        mock_get_redis.return_value = mock_client

        result = await cache.incr("counter", delta=delta)
        assert result == expected
