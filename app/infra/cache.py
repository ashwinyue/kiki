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


# ============== 缓存装饰器 ==============


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


# ============== 全局缓存实例 ==============

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


# ============== 预热函数 ==============


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
