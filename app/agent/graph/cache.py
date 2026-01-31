"""图缓存模块

提供 CompiledGraph 实例的缓存机制，避免每次请求都重新编译图。
使用 LRU 策略管理缓存，支持基于 system_prompt 的缓存键。
"""

from collections.abc import Callable
from threading import RLock
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 线程锁，保护图创建过程
_lock = RLock()

# 默认缓存大小
_DEFAULT_CACHE_SIZE = 8


def _create_cache_key(system_prompt: str | None) -> str:
    """创建缓存键

    Args:
        system_prompt: 系统提示词

    Returns:
        缓存键字符串
    """
    if system_prompt is None:
        return "__default__"
    # 使用哈希值避免过长的键，添加版本号避免冲突
    return f"v1_{hash(system_prompt)}"


class GraphCache:
    """图缓存管理器

    使用 LRU 策略缓存已编译的图实例。
    线程安全，支持并发访问。
    """

    def __init__(
        self,
        max_size: int = _DEFAULT_CACHE_SIZE,
        graph_factory: Callable[[str | None], CompiledStateGraph] | None = None,
    ) -> None:
        """初始化图缓存

        Args:
            max_size: 最大缓存大小
            graph_factory: 图工厂函数，用于测试或自定义
        """
        self._max_size = max_size
        self._cache: dict[str, CompiledStateGraph] = {}
        self._cache_keys: list[str] = []  # 用于追踪 LRU 顺序
        self._lock = RLock()
        self._graph_factory = graph_factory or self._default_graph_factory

        logger.info("graph_cache_initialized", max_size=max_size)

    def _default_graph_factory(self, system_prompt: str | None) -> CompiledStateGraph:
        """默认图工厂函数

        Args:
            system_prompt: 系统提示词

        Returns:
            编译后的 StateGraph 实例
        """
        from app.agent.graph.builder import compile_chat_graph

        return compile_chat_graph(system_prompt=system_prompt)

    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存项"""
        if self._cache_keys:
            oldest_key = self._cache_keys.pop(0)
            deleted = self._cache.pop(oldest_key, None)
            logger.debug("cache_evicted", key=oldest_key, was_present=deleted is not None)

    def get(self, system_prompt: str | None = None) -> CompiledStateGraph:
        """获取或创建图实例

        Args:
            system_prompt: 系统提示词，用作缓存键

        Returns:
            编译后的 StateGraph 实例
        """
        cache_key = _create_cache_key(system_prompt)

        with self._lock:
            # 检查缓存
            if cache_key in self._cache:
                # 更新 LRU 顺序
                self._cache_keys.remove(cache_key)
                self._cache_keys.append(cache_key)
                logger.debug("graph_cache_hit", key=cache_key)
                return self._cache[cache_key]

            # 缓存未命中，创建新图
            logger.debug("graph_cache_miss", key=cache_key, creating_new=True)

            # 检查缓存大小，必要时淘汰
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            # 创建并缓存图
            graph = self._graph_factory(system_prompt)

            self._cache[cache_key] = graph
            self._cache_keys.append(cache_key)

            logger.info(
                "graph_cached",
                key=cache_key,
                cache_size=len(self._cache),
                max_size=self._max_size,
            )

            return graph

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_keys.clear()
            logger.info("graph_cache_cleared")

    def invalidate(self, system_prompt: str | None = None) -> bool:
        """使指定缓存失效

        Args:
            system_prompt: 系统提示词（None 表示默认缓存）

        Returns:
            是否成功删除缓存项
        """
        cache_key = _create_cache_key(system_prompt)
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                if cache_key in self._cache_keys:
                    self._cache_keys.remove(cache_key)
                logger.info("cache_invalidated", key=cache_key)
                return True
            return False

    def size(self) -> int:
        """获取当前缓存大小"""
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, int]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "keys_count": len(self._cache_keys),
            }


# 全局图缓存实例
_global_cache: GraphCache | None = None


def get_graph_cache(max_size: int = _DEFAULT_CACHE_SIZE) -> GraphCache:
    """获取全局图缓存实例（单例）

    Args:
        max_size: 最大缓存大小（首次初始化时生效）

    Returns:
        GraphCache 实例
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = GraphCache(max_size=max_size)

    return _global_cache


def get_cached_graph(system_prompt: str | None = None) -> CompiledStateGraph:
    """获取缓存的图实例（便捷函数）

    Args:
        system_prompt: 系统提示词

    Returns:
        编译后的 StateGraph 实例

    Examples:
        ```python
        from app.agent.graph import get_cached_graph

        # 获取默认图
        graph = get_cached_graph()

        # 获取带自定义提示词的图
        graph = get_cached_graph(system_prompt="你是一个专业助手")
        ```
    """
    cache = get_graph_cache()
    return cache.get(system_prompt)


def clear_graph_cache() -> None:
    """清空全局图缓存"""
    global _global_cache

    if _global_cache is not None:
        _global_cache.clear()


def get_graph_cache_stats() -> dict[str, int]:
    """获取图缓存统计信息"""
    cache = get_graph_cache()
    return cache.stats()


def invalidate_graph(system_prompt: str | None = None) -> bool:
    """使指定图缓存失效（便捷函数）

    Args:
        system_prompt: 系统提示词（None 表示默认缓存）

    Returns:
        是否成功删除缓存项
    """
    cache = get_graph_cache()
    return cache.invalidate(system_prompt)


__all__ = [
    "GraphCache",
    "get_graph_cache",
    "get_cached_graph",
    "clear_graph_cache",
    "get_graph_cache_stats",
    "invalidate_graph",
]
