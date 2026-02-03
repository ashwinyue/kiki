"""Web 搜索工具

使用 LangChain 的 DuckDuckGoSearchAPIWrapper 实现。
支持异步执行，带超时控制。
"""

import asyncio
from functools import wraps

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    _duckduckgo_available = True
except ImportError:
    _duckduckgo_available = False
    logger.warning("langchain_community_not_installed")

# 默认超时时间（秒）
_DEFAULT_TIMEOUT = 10.0


class LangChainSearchWrapper:
    """LangChain 搜索包装器

    使用 LangChain 的 DuckDuckGoSearchAPIWrapper 实现。
    """


    def __init__(self) -> None:
        """初始化搜索包装器"""
        if _duckduckgo_available:
            self._wrapper = DuckDuckGoSearchAPIWrapper()
        else:
            self._wrapper = None

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """执行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        if self._wrapper is None:
            return []

        # 使用 LangChain 的工具方法
        raw_results = self._wrapper.results(query, max_results=max_results)

        results = []
        for result in raw_results:
            results.append(
                {
                    "title": result.get("title", ""),
                    "href": result.get("link", ""),
                    "body": result.get("snippet", ""),
                }
            )

        return results

    def format_results(self, results: list[dict[str, str]], max_length: int = 200) -> str:
        """格式化搜索结果

        Args:
            results: 搜索结果列表
            max_length: 摘要最大长度

        Returns:
            格式化后的结果字符串
        """
        if not results:
            return "未找到相关结果"

        output_parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            href = result.get("href", "")
            body = result.get("body", "")

            output_parts.append(f"[{i}] {title}")
            if href:
                output_parts.append(f"   链接: {href}")
            if body:
                body_preview = body[:max_length] + "..." if len(body) > max_length else body
                output_parts.append(f"   摘要: {body_preview}")
            output_parts.append("")

        return "\n".join(output_parts)


# 全局搜索包装器实例
_search_wrapper: LangChainSearchWrapper | None = None


def _get_search_wrapper() -> LangChainSearchWrapper:
    """获取搜索包装器实例"""
    global _search_wrapper
    if _search_wrapper is None:
        _search_wrapper = LangChainSearchWrapper()
    return _search_wrapper


def _run_in_executor(timeout: float | None = None):
    """装饰器：将同步函数在线程池中执行

    Args:
        timeout: 超时时间（秒），None 表示不超时

    Returns:
        装饰后的异步函数
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, func, *args, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro

        return wrapper

    return decorator


@_run_in_executor(timeout=_DEFAULT_TIMEOUT)
def _sync_search(query: str, max_results: int) -> list[dict[str, str]]:
    """同步执行 DuckDuckGo 搜索

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    wrapper = _get_search_wrapper()
    return wrapper.search(query, max_results)


@tool
async def search_web(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网络

    在线程池中执行搜索，避免阻塞事件循环。
    带有超时保护。

    Args:
        query: 搜索查询关键词
        max_results: 最大结果数（1-10）

    Returns:
        搜索结果摘要，包含编号、标题、链接和摘要
    """
    if not _duckduckgo_available:
        return "搜索功能不可用，请安装 langchain-community: uv add langchain-community"

    # 限制 max_results 范围
    max_results = max(1, min(10, max_results))

    logger.info("web_search_started", query=query, max_results=max_results)

    try:
        results = await _sync_search(query, max_results)

        if not results:
            logger.info("web_search_no_results", query=query)
            return "未找到相关结果"

        # 使用包装器格式化结果
        wrapper = _get_search_wrapper()
        result_text = wrapper.format_results(results)

        logger.info("web_search_completed", query=query, result_count=len(results))
        return result_text

    except TimeoutError:
        logger.warning("web_search_timeout", query=query, timeout=_DEFAULT_TIMEOUT)
        return f"搜索超时（{_DEFAULT_TIMEOUT}秒），请稍后重试或尝试其他搜索词"

    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        return f"搜索失败: {str(e)}"


# 导出为工具
__all__ = ["search_web"]
