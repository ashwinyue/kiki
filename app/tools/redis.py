"""Redis 客户端模块

提供连接池管理和常用操作的封装。
"""

from functools import lru_cache

from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

# 全局连接池
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """获取 Redis 连接池（单例）

    Returns:
        ConnectionPool 实例
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_pool_size,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            decode_responses=settings.redis_decode_responses,
        )
        logger.info(
            "redis_pool_created",
            url=settings.redis_url,
            pool_size=settings.redis_pool_size,
        )
    return _pool


async def get_redis() -> Redis:
    """获取 Redis 客户端实例

    每次调用返回新的客户端实例，但共享底层连接池。

    Returns:
        Redis 客户端
    """
    pool = _get_pool()
    return Redis(connection_pool=pool)


async def close_redis() -> None:
    """关闭 Redis 连接池"""
    global _pool
    if _pool:
        await _pool.aclose()
        _pool = None
        logger.info("redis_pool_closed")


async def ping() -> bool:
    """检查 Redis 连接状态

    Returns:
        True 表示连接正常
    """
    try:
        client = await get_redis()
        return await client.ping()
    except RedisError as e:
        logger.error("redis_ping_failed", error=str(e))
        return False


class RedisCache:
    """Redis 缓存封装

    提供常用的缓存操作方法。

    Examples:
        ```python
        cache = RedisCache()

        # 设置缓存
        await cache.set("key", "value", ttl=60)

        # 获取缓存
        value = await cache.get("key")

        # 删除缓存
        await cache.delete("key")
        ```
    """

    def __init__(self, key_prefix: str = "kiki:cache:") -> None:
        """初始化缓存

        Args:
            key_prefix: 键前缀，用于命名空间隔离
        """
        self.key_prefix = key_prefix

    def _make_key(self, key: str) -> str:
        """生成带前缀的键"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> str | None:
        """获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在时返回 None
        """
        try:
            client = await get_redis()
            value = await client.get(self._make_key(key))
            return value
        except RedisError as e:
            logger.error("redis_get_failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | None = None,
    ) -> bool:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None 表示不过期

        Returns:
            是否设置成功
        """
        try:
            client = await get_redis()
            full_key = self._make_key(key)
            if ttl:
                await client.setex(full_key, ttl, value)
            else:
                await client.set(full_key, value)
            return True
        except RedisError as e:
            logger.error("redis_set_failed", key=key, error=str(e))
            return False

    async def delete(self, *keys: str) -> int:
        """删除缓存

        Args:
            *keys: 要删除的键

        Returns:
            删除的数量
        """
        if not keys:
            return 0
        try:
            client = await get_redis()
            full_keys = [self._make_key(key) for key in keys]
            return await client.delete(*full_keys)
        except RedisError as e:
            logger.error("redis_delete_failed", keys=keys, error=str(e))
            return 0

    async def exists(self, *keys: str) -> int:
        """检查键是否存在

        Args:
            *keys: 要检查的键

        Returns:
            存在的键数量
        """
        if not keys:
            return 0
        try:
            client = await get_redis()
            full_keys = [self._make_key(key) for key in keys]
            return await client.exists(*full_keys)
        except RedisError as e:
            logger.error("redis_exists_failed", keys=keys, error=str(e))
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间

        Args:
            key: 缓存键
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        try:
            client = await get_redis()
            return await client.expire(self._make_key(key), ttl)
        except RedisError as e:
            logger.error("redis_expire_failed", key=key, error=str(e))
            return False

    async def incr(self, key: str, delta: int = 1) -> int | None:
        """增加计数器

        Args:
            key: 键
            delta: 增量

        Returns:
            增加后的值，失败时返回 None
        """
        try:
            client = await get_redis()
            if delta == 1:
                return await client.incr(self._make_key(key))
            else:
                return await client.incrby(self._make_key(key), delta)
        except RedisError as e:
            logger.error("redis_incr_failed", key=key, error=str(e))
            return None

    async def ttl(self, key: str) -> int:
        """获取剩余过期时间

        Args:
            key: 缓存键

        Returns:
            剩余秒数，-1 表示永不过期，-2 表示键不存在
        """
        try:
            client = await get_redis()
            return await client.ttl(self._make_key(key))
        except RedisError as e:
            logger.error("redis_ttl_failed", key=key, error=str(e))
            return -2


class RedisLock:
    """Redis 分布式锁

    基于 SET NX EX 实现。

    Examples:
        ```python
        lock = RedisLock("my_resource")

        # 获取锁
        if await lock.acquire(ttl=30):
            try:
                # 执行临界区代码
                ...
            finally:
                await lock.release()
        ```
    """

    def __init__(self, resource: str, key_prefix: str = "kiki:lock:") -> None:
        """初始化锁

        Args:
            resource: 锁定的资源名称
            key_prefix: 键前缀
        """
        self.resource = resource
        self.key = f"{key_prefix}{resource}"
        self._token: str | None = None

    async def acquire(self, ttl: int = 30) -> bool:
        """获取锁

        Args:
            ttl: 锁的过期时间（秒）

        Returns:
            是否获取成功
        """
        import uuid

        self._token = str(uuid.uuid4())
        try:
            client = await get_redis()
            acquired = await client.set(
                self.key,
                self._token,
                nx=True,
                ex=ttl,
            )
            if acquired:
                logger.debug("redis_lock_acquired", resource=self.resource, ttl=ttl)
            return bool(acquired)
        except RedisError as e:
            logger.error("redis_lock_acquire_failed", resource=self.resource, error=str(e))
            return False

    async def release(self) -> bool:
        """释放锁

        使用 Lua 脚本确保只释放自己持有的锁。

        Returns:
            是否释放成功
        """
        if not self._token:
            return False

        try:
            client = await get_redis()
            # Lua 脚本：只删除值匹配的键
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            result = await client.eval(script, 1, self.key, self._token)
            released = bool(result)
            if released:
                logger.debug("redis_lock_released", resource=self.resource)
            return released
        except RedisError as e:
            logger.error("redis_lock_release_failed", resource=self.resource, error=str(e))
            return False


@lru_cache
def get_cache(key_prefix: str = "kiki:cache:") -> RedisCache:
    """获取缓存实例（带缓存的工厂函数）

    Args:
        key_prefix: 键前缀

    Returns:
        RedisCache 实例
    """
    return RedisCache(key_prefix=key_prefix)
