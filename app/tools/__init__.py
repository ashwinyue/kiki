"""通用工具集

提供缓存、存储、搜索等基础设施工具。
"""

from app.tools.cache import (
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
from app.tools.redis import close_redis, get_redis, ping
from app.tools.search import SearchEngine, get_search_engine, with_web_search
from app.tools.storage import Storage, get_storage

__all__ = [
    # Cache
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
    # Storage
    "Storage",
    "get_storage",
    # Search
    "SearchEngine",
    "get_search_engine",
    "with_web_search",
]
