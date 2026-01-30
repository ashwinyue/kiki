"""Web 搜索工具

支持多种搜索引擎的集成。
"""

import os
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.metrics import track_tool_call


logger = get_logger(__name__)


# ============== DuckDuckGo 搜索 ==============

try:
    from duckduckgo_search import DDGS

    _duckduckgo_available = True
except ImportError:
    _duckduckgo_available = False
    logger.warning("duckduckgo_search_not_installed")


@tool
async def search_web(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网络

    Args:
        query: 搜索查询
        max_results: 最大结果数（1-10）

    Returns:
        搜索结果摘要
    """
    if not _duckduckgo_available:
        return "搜索功能不可用，请安装 duckduckgo-search: pip install duckduckgo-search"

    async with track_tool_call("search_web"):
        try:
            results = []
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(
                    query,
                    max_results=max_results,
                )
                for result in ddgs_gen:
                    results.append(f"- {result.get('title', '')}: {result.get('href', '')}")
                    if result.get('body'):
                        results.append(f"  {result['body'][:200]}...")

            logger.info("web_search_completed", query=query, result_count=len(results))
            return "\n".join(results) if results else "未找到相关结果"

        except Exception as e:
            logger.error("web_search_failed", query=query, error=str(e))
            return f"搜索失败: {str(e)}"


# ============== Tavily 搜索（备选方案）============

class TavilySearchInput(BaseModel):
    """Tavily 搜索输入"""
    query: str = Field(description="搜索查询")
    max_results: int = Field(default=5, ge=1, le=10, description="最大结果数")
    search_depth: str = Field(default="basic", description="搜索深度: basic/advanced")


@tool
async def search_web_tavily(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> str:
    """使用 Tavily API 进行网络搜索（需要 API Key）

    Args:
        query: 搜索查询
        max_results: 最大结果数
        search_depth: 搜索深度

    Returns:
        搜索结果摘要
    """
    async with track_tool_call("search_web_tavily"):
        try:
            from tavily import TavilyClient
        except ImportError:
            return "Tavily 搜索不可用，请安装: pip install tavily-python"

        import os
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Tavily API Key 未配置，请设置 TAVILY_API_KEY 环境变量"

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=[],
                exclude_domains=[],
            )

            results = []
            for answer in response.get("answer", []):
                results.append(f"答案: {answer}")

            for result in response.get("results", []):
                results.append(f"- {result['title']}")
                results.append(f"  URL: {result['url']}")
                if result.get("content"):
                    results.append(f"  内容: {result['content'][:200]}...")

            logger.info("tavily_search_completed", query=query, result_count=len(results))
            return "\n".join(results) if results else "未找到相关结果"

        except Exception as e:
            logger.error("tavily_search_failed", query=query, error=str(e))
            return f"Tavily 搜索失败: {str(e)}"


# ============== 统一搜索接口 ==============

class SearchEngine:
    """搜索引擎管理类

    支持多个搜索引擎，自动回退。
    """

    def __init__(self) -> None:
        self._engines = ["duckduckgo", "tavily"]
        self._current_index = 0

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        """执行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            搜索结果
        """
        # 尝试所有搜索引擎
        for _ in range(len(self._engines)):
            engine = self._engines[self._current_index]

            try:
                if engine == "duckduckgo" and _duckduckgo_available:
                    return await search_web.func(query=query, max_results=max_results)
                elif engine == "tavily":
                    return await search_web_tavily.func(
                        query=query,
                        max_results=max_results,
                    )

            except Exception as e:
                logger.warning("search_engine_failed", engine=engine, error=str(e))
                # 切换到下一个引擎
                self._current_index = (self._current_index + 1) % len(self._engines)

        return "所有搜索引擎都失败了"

    def set_engine(self, engine: str) -> None:
        """设置首选搜索引擎

        Args:
            engine: 引擎名称 (duckduckgo/tavily)
        """
        if engine in self._engines:
            self._current_index = self._engines.index(engine)
            logger.info("search_engine_set", engine=engine)


# 全局搜索引擎实例
_search_engine: SearchEngine | None = None


def get_search_engine() -> SearchEngine:
    """获取搜索引擎实例（单例）

    Returns:
        SearchEngine 实例
    """
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine


# ============== 装饰器：为 Agent 添加搜索能力 ==============

def with_web_search(agent_class: type) -> type:
    """为 Agent 类添加 Web 搜索能力

    Args:
        agent_class: Agent 类

    Returns:
        增强后的 Agent 类

    Examples:
        ```python
        @with_web_search
        class MyAgent:
            ...
        ```
    """
    original_init = agent_class.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._search_engine = get_search_engine()

    # 添加搜索方法
    async def search_web(self, query: str, max_results: int = 5) -> str:
        """搜索网络"""
        return await self._search_engine.search(query, max_results)

    agent_class.__init__ = __init__
    agent_class.search_web = search_web

    return agent_class
