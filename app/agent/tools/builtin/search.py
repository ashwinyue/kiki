"""Web 搜索工具

支持 DuckDuckGo 和 Tavily 搜索引擎。
异步实现，带超时控制。
"""

import asyncio
from functools import wraps

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
try:
    from duckduckgo_search import DDGS

    _duckduckgo_available = True
except ImportError:
    _duckduckgo_available = False
    logger.warning("duckduckgo_search_not_installed")

# 默认超时时间（秒）
_DEFAULT_TIMEOUT = 10.0


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
    results = []
    with DDGS() as ddgs:
        ddgs_gen = ddgs.text(query, max_results=max_results)
        for result in ddgs_gen:
            results.append(
                {
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", ""),
                }
            )
    return results


@tool
async def search_web(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网络（异步版本）

    在线程池中执行搜索，避免阻塞事件循环。
    带有 10 秒超时保护。

    Args:
        query: 搜索查询关键词
        max_results: 最大结果数（1-10）

    Returns:
        搜索结果摘要，包含标题、链接和摘要
    """
    if not _duckduckgo_available:
        return "搜索功能不可用，请安装 duckduckgo-search"

    # 限制 max_results 范围
    max_results = max(1, min(10, max_results))

    logger.info("web_search_started", query=query, max_results=max_results)

    try:
        results = await _sync_search(query, max_results)

        if not results:
            logger.info("web_search_no_results", query=query)
            return "未找到相关结果"

        # 格式化结果
        output_parts = []
        for result in results:
            title = result["title"]
            href = result["href"]
            body = result["body"]

            output_parts.append(f"标题: {title}")
            output_parts.append(f"链接: {href}")
            if body:
                output_parts.append(f"摘要: {body[:200]}...")
            output_parts.append("")  # 空行分隔

        result_text = "\n".join(output_parts)
        logger.info("web_search_completed", query=query, result_count=len(results))
        return result_text

    except TimeoutError:
        logger.warning("web_search_timeout", query=query, timeout=_DEFAULT_TIMEOUT)
        return f"搜索超时（{_DEFAULT_TIMEOUT}秒），请稍后重试或尝试其他搜索词"

    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        return f"搜索失败: {str(e)}"
