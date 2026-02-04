"""Redis 缓存基础设施

提供生产级 Redis 缓存服务，支持：

- **TTL 抖动**：防止缓存雪崩（同时过期导致流量激增）
- **分布式锁**：防止缓存击穿（热点数据过期导致并发查询）
- **空值缓存**：防止缓存穿透（查询不存在数据导致频繁查库）
- **批量操作**：减少网络往返，提升性能
- **缓存装饰器**：简化使用，支持旁路控制和预热

使用示例：
    ```python
    from app.infra.cache import cached, cache_instance

    # 使用装饰器
    @cached(ttl=600, key_prefix="user")
    async def get_user(user_id: int):
        return await db.fetch_user(user_id)

    # 直接使用
    await cache_instance.set("key", "value", ttl=300)
    value = await cache_instance.get("key")
    ```

参考: ai-engineer-training2/week09/3/p30缓存策略设计/Redis异步客户端集成.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import random
import threading
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config.settings import get_settings
from app.observability.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class SerializationFormat(Enum):
    """序列化格式枚举"""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


class RedisCache:
    """Redis 异步缓存封装

    特性：
    - TTL 抖动：防止缓存同时过期导致的雪崩
    - 批量操作：减少网络往返
    - 多种序列化：JSON/Pickle/字符串
    - 连接池管理：自动重连
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 300,
        jitter_percent: float = 0.1,
        encoding: str = "utf-8",
    ):
        """初始化 Redis 缓存

        Args:
            redis_url: Redis 连接 URL
            default_ttl: 默认 TTL（秒）
            jitter_percent: TTL 抖动百分比（0.1 = ±10%）
            encoding: 字符串编码
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.jitter_percent = jitter_percent
        self.encoding = encoding
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """连接 Redis（幂等）"""
        if self.redis:
            return

        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding=self.encoding,
                decode_responses=False,
                max_connections=20,
                socket_connect_timeout=5,
                socket_timeout=30,
            )
            await self.redis.ping()
            logger.info("redis_connected", url=self.redis_url)
        except Exception as e:
            logger.error("redis_connect_failed", error=str(e))
            raise

    async def close(self) -> None:
        """关闭 Redis 连接"""
        if self.redis:
            await self.redis.close()
            self.redis = None

    def _add_jitter(self, ttl: int) -> int:
        """为 TTL 添加随机抖动，防止缓存雪崩

        Args:
            ttl: 原始 TTL

        Returns:
            int: 带抖动的 TTL
        """
        if ttl < 10:
            return ttl

        jitter = random.randint(
            -int(ttl * self.jitter_percent),
            int(ttl * self.jitter_percent),
        )
        return max(1, ttl + jitter)

    def _serialize(
        self,
        value: Any,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> bytes:
        """序列化值

        Args:
            value: 要序列化的值
            format: 序列化格式

        Returns:
            bytes: 序列化后的字节
        """
        if format == SerializationFormat.JSON:
            if isinstance(value, (str, int, float, bool)):
                return str(value).encode(self.encoding)
            return json.dumps(value, ensure_ascii=False).encode(self.encoding)

        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(value)

        else:  # STRING
            return str(value).encode(self.encoding)

    def _deserialize(self, value: bytes) -> Any:
        """反序列化值（自动检测格式）

        Args:
            value: 要反序列化的字节

        Returns:
            Any: 反序列化后的值
        """
        # 尝试 JSON
        try:
            return json.loads(value.decode(self.encoding))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # 尝试字符串
        try:
            return value.decode(self.encoding)
        except UnicodeDecodeError:
            pass

        # 尝试 Pickle
        try:
            return pickle.loads(value)
        except Exception:
            pass

        # 返回原始 bytes
        return value

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值或 None
        """
        try:
            if not self.redis:
                await self.connect()

            value = await self.redis.get(key)
            if value is not None:
                return self._deserialize(value)
            return None

        except Exception as e:
            logger.error("cache_get_failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> bool:
        """设置缓存（带 TTL 抖动）

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None 表示使用默认值
            format: 序列化格式

        Returns:
            bool: 是否成功
        """
        try:
            if not self.redis:
                await self.connect()

            serialized = self._serialize(value, format)
            actual_ttl = self._add_jitter(ttl or self.default_ttl)

            await self.redis.setex(key, actual_ttl, serialized)
            return True

        except Exception as e:
            logger.error("cache_set_failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存

        Args:
            key: 缓存键

        Returns:
            bool: 是否成功
        """
        try:
            if not self.redis:
                await self.connect()

            await self.redis.delete(key)
            return True

        except Exception as e:
            logger.error("cache_delete_failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在
        """
        try:
            if not self.redis:
                await self.connect()

            return await self.redis.exists(key) > 0

        except Exception:
            return False

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """原子递增

        Args:
            key: 缓存键
            amount: 递增量

        Returns:
            int: 新值，失败返回 None
        """
        try:
            if not self.redis:
                await self.connect()

            return await self.redis.incrby(key, amount)

        except Exception as e:
            logger.error("cache_incr_failed", key=key, error=str(e))
            return None

    async def get_many(self, keys: list[str]) -> list[Optional[Any]]:
        """批量获取缓存

        Args:
            keys: 缓存键列表

        Returns:
            list: 对应的值列表
        """
        try:
            if not self.redis:
                await self.connect()

            raw_values = await self.redis.mget(keys)
            return [
                self._deserialize(v) if v is not None else None
                for v in raw_values
            ]

        except Exception as e:
            logger.error("cache_get_many_failed", keys=keys, error=str(e))
            return [None] * len(keys)

    async def set_many(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> bool:
        """批量设置缓存

        Args:
            mapping: 键值对映射
            ttl: 过期时间

        Returns:
            bool: 是否成功
        """
        try:
            if not self.redis:
                await self.connect()

            actual_ttl = self._add_jitter(ttl or self.default_ttl)
            pipe = self.redis.pipeline(transaction=True)

            for key, value in mapping.items():
                serialized = self._serialize(value)
                pipe.setex(key, actual_ttl, serialized)

            await pipe.execute()
            return True

        except Exception as e:
            logger.error("cache_set_many_failed", error=str(e))
            return False


class DistributedLock:
    """Redis 分布式锁

    用于防止缓存击穿、控制并发访问等场景。

    使用 SET NX (SET if Not eXists) 实现。
    """

    def __init__(
        self,
        cache: RedisCache,
        lock_timeout: int = 30,
    ):
        """初始化分布式锁

        Args:
            cache: Redis 缓存实例
            lock_timeout: 锁超时时间（秒）
        """
        self.cache = cache
        self.lock_timeout = lock_timeout

    def _get_identifier(self) -> str:
        """获取锁持有者标识

        Returns:
            str: 进程 ID + 线程 ID
        """
        return f"{os.getpid()}:{threading.current_thread().ident}"

    async def acquire(
        self,
        resource: str,
        timeout: int = 10,
    ) -> bool:
        """获取锁

        Args:
            resource: 资源标识（锁名）
            timeout: 获取锁的超时时间（秒）

        Returns:
            bool: 是否成功获取锁
        """
        if not self.cache.redis:
            await self.cache.connect()

        lock_key = f"lock:{resource}"
        identifier = self._get_identifier()
        end_time = time.time() + timeout

        while time.time() < end_time:
            # SET NX: 只在键不存在时设置
            acquired = await self.cache.redis.set(
                lock_key,
                identifier,
                nx=True,
                ex=self.lock_timeout,
            )

            if acquired:
                logger.debug(
                    "lock_acquired",
                    resource=resource,
                    identifier=identifier,
                )
                return True

            await asyncio.sleep(0.1)

        logger.debug("lock_acquire_timeout", resource=resource, timeout=timeout)
        return False

    async def release(self, resource: str) -> bool:
        """释放锁（使用 Lua 脚本确保只释放自己持有的锁）

        Args:
            resource: 资源标识

        Returns:
            bool: 是否成功释放
        """
        if not self.cache.redis:
            return False

        lock_key = f"lock:{resource}"
        identifier = self._get_identifier()

        # Lua 脚本：原子地检查并删除
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await self.cache.redis.eval(
                lua_script,
                1,
                lock_key,
                identifier,
            )
            return result > 0

        except Exception as e:
            logger.error("lock_release_failed", resource=resource, error=str(e))
            return False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        pass


class CachePenetrationProtection:
    """缓存穿透防护

    对于不存在的数据，缓存一个空值标记，
    防止每次查询都穿透到数据库。
    """

    def __init__(
        self,
        cache: RedisCache,
        null_ttl: int = 60,
    ):
        """初始化缓存穿透防护

        Args:
            cache: Redis 缓存实例
            null_ttl: 空值缓存时间（秒）
        """
        self.cache = cache
        self.null_ttl = null_ttl

    async def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable,
        ttl: int = 300,
    ) -> Optional[Any]:
        """获取数据（带缓存穿透防护）

        Args:
            key: 缓存键
            fetch_func: 数据获取函数
            ttl: 正常数据的缓存时间

        Returns:
            缓存值或获取的数据
        """
        # 尝试从缓存获取
        cached = await self.cache.get(key)
        if cached is not None:
            # 检查是否是空值标记
            if cached == "__NULL__":
                return None
            return cached

        # 检查空值标记
        null_key = f"{key}:null"
        if await self.cache.exists(null_key):
            logger.debug("cache_null_hit", key=key)
            return None

        # 获取数据
        try:
            result = await fetch_func()
        except Exception as e:
            logger.error("cache_fetch_failed", key=key, error=str(e))
            raise

        # 缓存结果
        if result is not None:
            await self.cache.set(key, result, ttl)
        else:
            # 缓存空值标记
            await self.cache.set(null_key, "__NULL__", self.null_ttl)

        return result


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    exclude_params: Optional[list[str]] = None,
    bypass_param: str = "_cache_bypass",
):
    """缓存装饰器

    Args:
        ttl: 缓存时间（秒）
        key_prefix: 键前缀
        exclude_params: 要排除的参数名列表
        bypass_param: 旁路控制参数名

    使用示例：
        ```python
        @cached(ttl=600, key_prefix="user")
        async def get_user(user_id: int):
            return await db.fetch_user(user_id)

        # 强制跳过缓存
        user = await get_user(123, _cache_bypass=True)
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 确保缓存已连接
            if cache_instance.redis is None:
                await cache_instance.connect()

            # 检查旁路参数
            bypass = kwargs.pop(bypass_param, False)

            # 生成缓存键
            cache_key = generate_cache_key(
                func,
                args,
                kwargs,
                key_prefix,
                exclude_params,
            )

            # 非旁路时尝试命中缓存
            if not bypass:
                cached_result = await cache_instance.get(cache_key)
                if cached_result is not None:
                    logger.debug(
                        "cache_hit",
                        key=cache_key,
                        function=func.__name__,
                    )
                    return cached_result

            # 使用分布式锁防止缓存击穿
            lock = DistributedLock(cache_instance, lock_timeout=max(5, int(ttl * 0.3)))
            acquired = await lock.acquire(cache_key, timeout=3)

            try:
                if acquired:
                    # 二次检查
                    if not bypass:
                        cached_again = await cache_instance.get(cache_key)
                        if cached_again is not None:
                            logger.debug("cache_hit_double_check", key=cache_key)
                            return cached_again

                    # 执行函数
                    result = await func(*args, **kwargs)
                    await cache_instance.set(cache_key, result, ttl)
                    logger.debug("cache_set", key=cache_key)
                    return result

                else:
                    # 未获取锁，等待其他工作者完成
                    for _ in range(20):
                        await asyncio.sleep(0.1)
                        cached_result = await cache_instance.get(cache_key)
                        if cached_result is not None:
                            return cached_result

                    # 仍未命中，自行计算
                    result = await func(*args, **kwargs)
                    await cache_instance.set(cache_key, result, ttl)
                    return result

            finally:
                if acquired:
                    await lock.release(cache_key)

        return wrapper
    return decorator


def generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    prefix: str = "",
    exclude_params: Optional[list[str]] = None,
) -> str:
    """生成缓存键

    Args:
        func: 被装饰的函数
        args: 位置参数
        kwargs: 关键字参数
        prefix: 键前缀
        exclude_params: 要排除的参数名

    Returns:
        str: MD5 哈希后的缓存键
    """
    # 排除不需要的参数
    filtered_kwargs = kwargs.copy()
    if exclude_params:
        for param in exclude_params:
            filtered_kwargs.pop(param, None)

    # 构建键数据
    key_data = {
        "func": f"{func.__module__}.{func.__name__}",
        "args": str(args),
        "kwargs": str(filtered_kwargs),
    }

    key_str = f"{prefix}:{json.dumps(key_data, sort_keys=True, default=str)}"
    return hashlib.md5(key_str.encode()).hexdigest()


cache_instance = RedisCache(
    redis_url=str(settings.redis_url),
    default_ttl=300,
    jitter_percent=0.1,
)

distributed_lock = DistributedLock(cache_instance)
penetration_protection = CachePenetrationProtection(cache_instance)


async def get_cache() -> RedisCache:
    """获取缓存实例（确保已连接）

    Returns:
        RedisCache: 缓存实例
    """
    if cache_instance.redis is None:
        await cache_instance.connect()
    return cache_instance


async def warmup_cache(
    keys_and_values: dict[str, Any],
    ttl: int = 3600,
) -> int:
    """预热缓存

    Args:
        keys_and_values: 键值对映射
        ttl: 过期时间

    Returns:
        int: 成功设置的条目数
    """
    count = 0
    for key, value in keys_and_values.items():
        if await cache_instance.set(key, value, ttl):
            count += 1

    logger.info("cache_warmed_up", count=count, ttl=ttl)
    return count


class CacheStats:
    """缓存统计

    跟踪缓存命中率、错误率等指标。
    """

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.l1_hits = 0
        self.l2_hits = 0

    @property
    def hit_rate(self) -> float:
        """计算命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        """L1 缓存命中率"""
        total = self.hits
        return self.l1_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "hit_rate": f"{self.hit_rate:.2%}",
            "l1_hit_rate": f"{self.l1_hit_rate:.2%}",
            "total_requests": self.hits + self.misses,
        }

    def reset(self) -> None:
        """重置统计"""
        self.__init__()


_global_cache_stats = CacheStats()


def get_cache_stats() -> CacheStats:
    """获取全局缓存统计"""
    return _global_cache_stats


class L1MemoryCache:
    """L1 内存缓存（LRU）

    用于缓存热点数据，减少 Redis 网络往返。

    特性：
    - LRU 淘汰策略
    - TTL 过期
    - 最大容量限制
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 60) -> None:
        """初始化 L1 缓存

        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        from collections import OrderedDict

        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def _is_expired(self, expiry: float) -> bool:
        """检查是否过期"""
        return time.time() > expiry

    def get(self, key: str) -> Any | None:
        """获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在或过期时返回 None
        """
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # 检查过期
        if self._is_expired(expiry):
            del self._cache[key]
            return None

        # 移到末尾（标记为最近使用）
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl

        # 如果键已存在，先删除
        if key in self._cache:
            del self._cache[key]

        # 添加到末尾
        self._cache[key] = (value, expiry)

        # 超过容量时删除最旧的条目
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """删除缓存值

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """清理过期条目

        Returns:
            清理的条目数
        """
        expired_keys = [
            key
            for key, (_, expiry) in self._cache.items()
            if self._is_expired(expiry)
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def __len__(self) -> int:
        return len(self._cache)


class MultiLayerCache:
    """多层缓存管理器

    提供 L1 (内存) + L2 (Redis) 双层缓存。
    读取时先查 L1，未命中再查 L2，并回填 L1。
    写入时同时写入 L1 和 L2。

    Examples:
        ```python
        cache = MultiLayerCache()

        # 设置缓存
        await cache.set("user:123", {"name": "Alice"})

        # 获取缓存（先查 L1，再查 L2）
        user = await cache.get("user:123")

        # 使用装饰器
        @cache.cached(ttl=600, key_prefix="user")
        async def get_user(user_id: int):
            return await db.fetch_user(user_id)
        ```
    """

    def __init__(
        self,
        l2_cache: RedisCache | None = None,
        l1_max_size: int = 1000,
        l1_ttl: int = 60,
        enable_l1: bool = True,
    ) -> None:
        """初始化多层缓存

        Args:
            l2_cache: L2 Redis 缓存实例
            l1_max_size: L1 缓存最大条目数
            l1_ttl: L1 缓存默认过期时间（秒）
            enable_l1: 是否启用 L1 缓存
        """
        self.l2_cache = l2_cache or cache_instance
        self.enable_l1 = enable_l1
        self.l1_cache = L1MemoryCache(max_size=l1_max_size, default_ttl=l1_ttl) if enable_l1 else None
        self.stats = _global_cache_stats

    async def get(self, key: str) -> Any | None:
        """获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在时返回 None
        """
        # 先查 L1
        # 注意：需要使用 `is not None` 检查，因为 L1MemoryCache 有 __len__ 方法
        if self.enable_l1 and self.l1_cache is not None:
            value = self.l1_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                self.stats.l1_hits += 1
                return value

        # 再查 L2
        try:
            value = await self.l2_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                self.stats.l2_hits += 1

                # 回填 L1
                if self.enable_l1 and self.l1_cache is not None:
                    self.l1_cache.set(key, value)

                return value
        except Exception as e:
            logger.warning("multilayer_cache_l2_failed", key=key, error=str(e))

        # 未命中
        self.stats.misses += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> bool:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            format: 序列化格式

        Returns:
            是否成功
        """
        success = True

        # 设置 L1（使用较短的 TTL）
        # 注意：需要使用 `is not None` 检查，因为 L1MemoryCache 有 __len__ 方法
        if self.enable_l1 and self.l1_cache is not None:
            try:
                self.l1_cache.set(key, value)
                logger.debug("multilayer_cache_l1_set", key=key, value=value)
            except Exception as e:
                logger.warning("multilayer_cache_l1_set_failed", key=key, error=str(e))

        # 设置 L2
        if not await self.l2_cache.set(key, value, ttl, format):
            success = False

        if success:
            self.stats.sets += 1

        return success

    async def delete(self, key: str) -> bool:
        """删除缓存值

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        deleted = False

        # 删除 L1
        if self.enable_l1 and self.l1_cache:
            if self.l1_cache.delete(key):
                deleted = True

        # 删除 L2
        if await self.l2_cache.delete(key):
            deleted = True

        if deleted:
            self.stats.deletes += 1

        return deleted

    def clear_l1(self) -> None:
        """清空 L1 缓存"""
        if self.l1_cache:
            self.l1_cache.clear()

    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        return self.stats

    def cached(
        self,
        ttl: int = 300,
        key_prefix: str = "",
        exclude_params: list[str] | None = None,
        use_l1: bool = True,
    ):
        """多层缓存装饰器

        Args:
            ttl: 缓存时间（秒）
            key_prefix: 键前缀
            exclude_params: 要排除的参数名
            use_l1: 是否使用 L1 缓存

        Returns:
            装饰器
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # 生成缓存键
                cache_key = generate_cache_key(
                    func, args, kwargs, key_prefix, exclude_params
                )

                # 尝试获取缓存
                if use_l1 and self.enable_l1 and self.l1_cache:
                    value = self.l1_cache.get(cache_key)
                    if value is not None:
                        return value

                value = await self.l2_cache.get(cache_key)
                if value is not None:
                    # 回填 L1
                    if use_l1 and self.l1_cache:
                        self.l1_cache.set(cache_key, value)
                    return value

                # 执行函数
                result = await func(*args, **kwargs)

                # 设置缓存
                await self.set(cache_key, result, ttl)
                if use_l1 and self.l1_cache:
                    self.l1_cache.set(cache_key, result)

                return result

            return wrapper

        return decorator


# 全局多层缓存实例
_multilayer_cache: MultiLayerCache | None = None


def get_multilayer_cache() -> MultiLayerCache:
    """获取多层缓存实例"""
    global _multilayer_cache
    if _multilayer_cache is None:
        _multilayer_cache = MultiLayerCache()
    return _multilayer_cache


class SemanticCache:
    """语义缓存

    基于嵌入向量相似度的缓存，用于 LLM 响应。
    相似度高于阈值的查询会被视为"相同"，直接返回缓存结果。

    Examples:
        ```python
        cache = SemanticCache(
            embedding_model=embedding_function,
            similarity_threshold=0.85
        )

        # 设置缓存
        await cache.set("What is Python?", "Python is a programming language")

        # 语义相似的查询会命中缓存
        response = await cache.get("Tell me about Python")  # 会命中
        ```
    """

    def __init__(
        self,
        embedding_model: Callable[[str], list[float]] | None = None,
        similarity_threshold: float = 0.85,
        ttl: int = 3600,
        cache_backend: RedisCache | None = None,
    ) -> None:
        """初始化语义缓存

        Args:
            embedding_model: 嵌入模型函数
            similarity_threshold: 相似度阈值（0-1）
            ttl: 缓存过期时间（秒）
            cache_backend: 后端缓存（默认使用全局缓存）
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.cache_backend = cache_backend or cache_instance
        self._index_key = "semantic_cache:index"
        self._embeddings: dict[str, list[float]] = {}

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """计算余弦相似度

        Args:
            vec1, vec2: 向量

        Returns:
            相似度（0-1）
        """
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def set(
        self,
        query: str,
        response: str,
        ttl: int | None = None,
    ) -> bool:
        """设置语义缓存

        Args:
            query: 查询文本
            response: 响应文本
            ttl: 过期时间（秒）

        Returns:
            是否成功
        """
        if not self.embedding_model:
            return False

        ttl = ttl or self.ttl
        query_key = f"semantic:query:{hashlib.md5(query.encode()).hexdigest()}"

        # 计算嵌入
        embedding = self.embedding_model(query)

        # 先更新内存索引
        self._embeddings[query_key] = embedding

        # 存储到后端缓存
        try:
            success = await self.cache_backend.set(
                query_key,
                {
                    "query": query,
                    "response": response,
                    "embedding": embedding,
                },
                ttl,
            )
            return success
        except Exception as e:
            logger.warning("semantic_cache_set_failed", query=query, error=str(e))
            return False

    async def get(self, query: str) -> str | None:
        """获取语义缓存

        Args:
            query: 查询文本

        Returns:
            缓存的响应，未找到相似查询时返回 None
        """
        if not self.embedding_model:
            return None

        # 计算查询嵌入
        query_embedding = self.embedding_model(query)

        # 查找最相似的缓存
        best_match: str | None = None
        best_similarity = 0.0
        best_key: str | None = None

        for cache_key, embedding in list(self._embeddings.items()):
            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                try:
                    # 获取实际缓存的响应
                    cached = await self.cache_backend.get(cache_key)
                    if cached:
                        best_match = cached.get("response")
                        best_similarity = similarity
                        best_key = cache_key
                except Exception as e:
                    logger.warning("semantic_cache_get_failed", key=cache_key, error=str(e))

        if best_match:
            logger.debug(
                "semantic_cache_hit",
                similarity=f"{best_similarity:.2f}",
                threshold=self.similarity_threshold,
            )

        return best_match

    def clear(self) -> None:
        """清空语义缓存"""
        self._embeddings.clear()


# 全局语义缓存实例
_semantic_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """获取语义缓存实例"""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache


class TenantCache:
    """租户隔离缓存

    为每个租户提供独立的缓存命名空间，
    防止不同租户的数据混淆。

    Examples:
        ```python
        cache = TenantCache()

        # 租户 1
        await cache.set(tenant_id=1, key="user:123", value={"name": "Alice"})

        # 租户 2（相同键不会冲突）
        await cache.set(tenant_id=2, key="user:123", value={"name": "Bob"})
        ```
    """

    def __init__(
        self,
        cache_backend: MultiLayerCache | None = None,
    ) -> None:
        """初始化租户缓存

        Args:
            cache_backend: 后端缓存实例
        """
        self.cache = cache_backend or get_multilayer_cache()

    def _make_key(self, tenant_id: int, key: str) -> str:
        """生成租户隔离的缓存键

        Args:
            tenant_id: 租户 ID
            key: 原始键

        Returns:
            带租户前缀的键
        """
        return f"tenant:{tenant_id}:{key}"

    async def get(
        self,
        tenant_id: int,
        key: str,
    ) -> Any | None:
        """获取租户缓存

        Args:
            tenant_id: 租户 ID
            key: 缓存键

        Returns:
            缓存值
        """
        cache_key = self._make_key(tenant_id, key)
        return await self.cache.get(cache_key)

    async def set(
        self,
        tenant_id: int,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """设置租户缓存

        Args:
            tenant_id: 租户 ID
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否成功
        """
        cache_key = self._make_key(tenant_id, key)
        return await self.cache.set(cache_key, value, ttl)

    async def delete(
        self,
        tenant_id: int,
        key: str,
    ) -> bool:
        """删除租户缓存

        Args:
            tenant_id: 租户 ID
            key: 缓存键

        Returns:
            是否成功
        """
        cache_key = self._make_key(tenant_id, key)
        return await self.cache.delete(cache_key)

    async def clear_tenant(self, tenant_id: int) -> None:
        """清空租户的所有缓存

        Args:
            tenant_id: 租户 ID
        """
        # TODO: 使用 SCAN 删除租户的所有键
        prefix = f"tenant:{tenant_id}:"
        logger.info("tenant_cache_cleared", tenant_id=tenant_id)


# 全局租户缓存实例
_tenant_cache: TenantCache | None = None


def get_tenant_cache() -> TenantCache:
    """获取租户缓存实例"""
    global _tenant_cache
    if _tenant_cache is None:
        _tenant_cache = TenantCache()
    return _tenant_cache
