"""Wikipedia 和 ArXiv 搜索工具

使用 LangChain Community 的内置工具。
参考 DeerFlow 的搜索工具设计。

安装依赖:
    uv add langchain-community wikipedia arxiv

使用示例:
```python
from app.agent.tools.builtin.academic import search_wikipedia, search_arxiv

# Wikipedia
wiki_result = await search_wikipedia("Python编程语言")

# ArXiv
arxiv_result = await search_arxiv("langchain agents")
```
"""

import asyncio
import functools
import logging
from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
_wikipedia_available = False
_arxiv_available = False

try:
    import wikipedia as wikipedia_lib

    _wikipedia_available = True
except ImportError:
    wikipedia_lib = None  # type: ignore
    logger.warning("wikipedia_not_installed")

try:
    import arxiv as arxiv_lib

    _arxiv_available = True
except ImportError:
    arxiv_lib = None  # type: ignore
    logger.warning("arxiv_not_installed")

# 默认超时
_DEFAULT_TIMEOUT = 10.0


def _run_in_executor(timeout: float | None = None):
    """装饰器：将同步函数在线程池中执行"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, func, *args, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        return wrapper
    return decorator


# ==================== Wikipedia 工具 ====================

@_run_in_executor(timeout=_DEFAULT_TIMEOUT)
def _sync_wikipedia_search(
    query: str,
    lang: str = "zh",
    sentences: int = 5,
) -> str:
    """同步执行 Wikipedia 搜索"""
    if not _wikipedia_available:
        return "Wikipedia 功能不可用，请安装: uv add wikipedia"

    wikipedia_lib.set_lang(lang)

    try:
        # 搜索页面
        page_titles = wikipedia_lib.search(query, results=1)
        if not page_titles:
            return f"未找到关于 '{query}' 的 Wikipedia 页面"

        # 获取页面摘要
        page = wikipedia_lib.page(page_titles[0], auto_suggest=False)
        summary = page.summary(sentences=sentences)

        return f"# {page.title}\n\n{summary}\n\n**阅读更多**: {page.url}"

    except wikipedia_lib.exceptions.PageError:
        return f"Wikipedia 页面不存在: {query}"
    except wikipedia_lib.exceptions.DisambiguationError as e:
        return f"Wikipedia 存在多个页面匹配: {e.options}"
    except Exception as e:
        return f"Wikipedia 搜索失败: {str(e)}"


@tool
async def search_wikipedia(
    query: str,
    lang: str = "zh",
    sentences: int = 5,
) -> str:
    """使用 Wikipedia 搜索百科知识

    Args:
        query: 搜索查询（支持中英文）
        lang: 语言代码（zh=中文, en=英文）
        sentences: 摘要句子数量

    Returns:
        Wikipedia 页面摘要

    Examples:
        >>> result = await search_wikipedia("Python编程语言")
        >>> result = await search_wikipedia("Machine Learning", lang="en")
    """
    logger.info(
        "wikipedia_search_start",
        query=query,
        lang=lang,
        sentences=sentences,
    )

    try:
        result = await _sync_wikipedia_search(query, lang, sentences)
        logger.info("wikipedia_search_complete", query=query)
        return result
    except TimeoutError:
        logger.warning("wikipedia_search_timeout", query=query)
        return f"Wikipedia 搜索超时（{_DEFAULT_TIMEOUT}秒）"
    except Exception as e:
        logger.error("wikipedia_search_failed", query=query, error=str(e))
        return f"Wikipedia 搜索失败: {str(e)}"


# ==================== ArXiv 工具 ====================

@_run_in_executor(timeout=_DEFAULT_TIMEOUT)
def _sync_arxiv_search(
    query: str,
    max_results: int = 5,
) -> str:
    """同步执行 ArXiv 搜索"""
    if not _arxiv_available:
        return "ArXiv 功能不可用，请安装: uv add arxiv"

    try:
        # 构建 ArXiv 查询
        search = arxiv_lib.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv_lib.SortCriterion.Relevance,
        )

        results = []
        for result in search.results():
            # 格式化结果
            title = result.title
            authors = ", ".join([a.name for a in result.authors[:3]])
            if len(result.authors) > 3:
                authors += " et al."

            published = result.published.strftime("%Y-%m-%d") if result.published else "n/a"
            url = result.entry_id
            summary = result.summary.replace("\n", " ")[:200] + "..." if result.summary else ""

            results.append(f"## {title}")
            results.append(f"**作者**: {authors}")
            results.append(f"**发布**: {published}")
            results.append(f"**摘要**: {summary}")
            results.append(f"**链接**: {url}")
            results.append("")

        if not results:
            return f"未找到关于 '{query}' 的 ArXiv 论文"

        return "\n".join(results)

    except Exception as e:
        return f"ArXiv 搜索失败: {str(e)}"


@tool
async def search_arxiv(
    query: str,
    max_results: int = 5,
) -> str:
    """使用 ArXiv 搜索学术论文

    ArXiv 是一个免费的学术论文预印本服务器，
    涵盖物理学、数学、计算机科学、定量生物学等领域。

    Args:
        query: 搜索查询（支持 ArXiv 查询语法）
        max_results: 最大结果数（1-10）

    Returns:
        论文列表，包含标题、作者、摘要和链接

    Examples:
        >>> result = await search_arxiv("langchain agents")
        >>> result = await search_arxiv("machine learning NLP", max_results=10)

    ArXiv 查询语法示例:
        - "ti:agent AND au:smith" - 标题包含 agent，作者包含 smith
        - "cat:cs.AI" - 计算机科学人工智能类别
        - "all:machine learning" - 所有字段搜索
    """
    logger.info(
        "arxiv_search_start",
        query=query,
        max_results=max_results,
    )

    try:
        result = await _sync_arxiv_search(query, max_results)
        logger.info("arxiv_search_complete", query=query)
        return result
    except TimeoutError:
        logger.warning("arxiv_search_timeout", query=query)
        return f"ArXiv 搜索超时（{_DEFAULT_TIMEOUT}秒）"
    except Exception as e:
        logger.error("arxiv_search_failed", query=query, error=str(e))
        return f"ArXiv 搜索失败: {str(e)}"


# ==================== 多源学术搜索 ====================

@tool
async def search_academic(
    query: str,
    sources: list[str] = ["wikipedia", "arxiv"],
    max_results: int = 3,
) -> str:
    """多源学术搜索

    同时搜索 Wikipedia 和 ArXiv，返回合并结果。

    Args:
        query: 搜索查询
        sources: 数据源列表（wikipedia/arxiv）
        max_results: 每个源的最大结果数

    Returns:
        合并的搜索结果

    Examples:
        >>> result = await search_academic("neural networks")
        >>> result = await search_academic("transformer", sources=["wikipedia"])
    """
    logger.info(
        "academic_search_start",
        query=query,
        sources=sources,
        max_results=max_results,
    )

    results = []
    tasks = []

    # 创建搜索任务
    if "wikipedia" in sources:
        tasks.append(search_wikipedia(query, sentences=max_results))

    if "arxiv" in sources:
        tasks.append(search_arxiv(query, max_results=max_results))

    if not tasks:
        return "错误：未指定有效的数据源"

    # 并行执行搜索
    try:
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for i, output in enumerate(outputs):
            source = sources[i]
            if isinstance(output, Exception):
                logger.error(
                    "academic_source_failed",
                    source=source,
                    error=str(output),
                )
                results.append(f"## {source.title()}\n\n搜索失败\n")
            else:
                results.append(f"## {source.title()}\n\n{output}\n")

        return "\n".join(results)

    except Exception as e:
        logger.error("academic_search_failed", query=query, error=str(e))
        return f"学术搜索失败: {str(e)}"


# ==================== 工具函数 ====================

def is_wikipedia_available() -> bool:
    """检查 Wikipedia 是否可用"""
    return _wikipedia_available


def is_arxiv_available() -> bool:
    """检查 ArXiv 是否可用"""
    return _arxiv_available


def get_available_sources() -> list[str]:
    """获取可用的学术数据源"""
    sources = []
    if _wikipedia_available:
        sources.append("wikipedia")
    if _arxiv_available:
        sources.append("arxiv")
    return sources
