"""通用工具集

提供缓存、数据库、搜索等基础设施工具。
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
from app.infra.database import (
    AsyncSession,
    close_db,
    get_async_engine,
    get_session,
    health_check,
    init_db,
)
from app.infra.redis import close_redis, get_redis, ping
from app.infra.search import SearchEngine, get_search_engine, with_web_search

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
    # Database
    "AsyncSession",
    "get_async_engine",
    "get_session",
    "init_db",
    "close_db",
    "health_check",
    # Redis
    "close_redis",
    "get_redis",
    "ping",
    # Search
    "SearchEngine",
    "get_search_engine",
    "with_web_search",
]
