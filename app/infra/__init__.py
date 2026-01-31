"""基础设施模块

提供缓存、Redis、存储等基础设施服务。
"""

from app.infra.cache import (
    CachePenetrationProtection,
    DistributedLock,
    RedisCache,
    cached,
    cache_instance,
    distributed_lock,
    get_cache,
    penetration_protection,
    warmup_cache,
)
from app.infra.redis import close_redis, get_redis, ping
from app.infra.storage import Storage, get_storage

__all__ = [
    # 缓存
    "RedisCache",
    "DistributedLock",
    "CachePenetrationProtection",
    "cached",
    "cache_instance",
    "distributed_lock",
    "penetration_protection",
    "get_cache",
    "warmup_cache",
    # Redis
    "close_redis",
    "get_redis",
    "ping",
    # 存储
    "Storage",
    "get_storage",
]
