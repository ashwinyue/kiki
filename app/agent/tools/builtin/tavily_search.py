"""Tavily 搜索集成

使用 Tavily API 进行网络搜索，支持图像和高级搜索。
参考 DeerFlow 的 TavilySearchWithImages 设计。

环境变量:
    TAVILY_API_KEY: Tavily API 密钥

安装依赖:
    uv add tavily-python

使用示例:
```python
from app.agent.tools.builtin.tavily_search import search_web_tavily

results = await search_web_tavily("Python LangGraph tutorial")
```
"""

from typing import Any, Literal

from langchain_core.tools import tool

from app.agent.tools.search_postprocessor import SearchResultPostProcessor
from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# 检查依赖
try:
    from tavily import TavilyClient as TavilySyncClient
    _tavily_available = True
except ImportError:
    _tavily_available = False
    TavilySyncClient = None  # type: ignore
    logger.warning("tavily_not_installed")


class TavilySearchConfig:
    """Tavily 搜索配置"""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = True,
        include_image_descriptions: bool = True,
        search_depth: Literal["basic", "advanced"] = "advanced",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        days: int = 3,
    ):
        self.api_key = api_key
        self.max_results = max_results
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.include_image_descriptions = include_image_descriptions
        self.search_depth = search_depth
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self.days = days


async def _search_with_config(
    query: str,
    config: TavilySearchConfig,
) -> dict[str, Any]:
    """使用配置执行搜索

    Args:
        query: 搜索查询
        config: 搜索配置

    Returns:
        搜索结果字典
    """
    if not _tavily_available:
        return {
            "error": "Tavily not available. Install with: uv add tavily-python",
        }

    if not config.api_key:
        return {"error": "TAVILY_API_KEY not configured"}

    try:
        client = TavilySyncClient(api_key=config.api_key)

        # 执行搜索
        response = client.search(
            query=query,
            max_results=config.max_results,
            include_answer=config.include_answer,
            include_raw_content=config.include_raw_content,
            include_images=config.include_images,
            include_image_descriptions=config.include_image_descriptions,
            search_depth=config.search_depth,
            include_domains=config.include_domains if config.include_domains else None,
            exclude_domains=config.exclude_domains if config.exclude_domains else None,
            days=config.days,
        )

        return response

    except Exception as e:
        logger.error("tavily_search_error", query=query, error=str(e))
        return {"error": str(e)}


def _format_tavily_results(response: dict[str, Any], query: str) -> str:
    """格式化 Tavily 搜索结果

    Args:
        response: Tavily API 响应
        query: 原始查询

    Returns:
        格式化的结果字符串
    """
    if "error" in response:
        return f"搜索失败: {response['error']}"

    parts = []

    # 添加答案（如果有）
    if response.get("answer"):
        parts.append(f"## 答案\n\n{response['answer']}\n")

    # 添加搜索结果
    results = response.get("results", [])
    if results:
        parts.append("## 搜索结果\n")
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("score", 0)

            parts.append(f"### {i}. {title}")
            parts.append(f"**URL**: {url}")
            if score:
                parts.append(f"**相关性**: {score:.2f}")
            if content:
                # 截断过长内容
                content_preview = content[:500] + "..." if len(content) > 500 else content
                parts.append(f"**摘要**: {content_preview}")
            parts.append("")

    # 添加图像（如果有）
    images = response.get("images", [])
    if images:
        parts.append("## 相关图像\n")
        for i, image in enumerate(images[:5], 1):  # 最多显示 5 张
            url = image.get("url", "")
            description = image.get("description", "")
            parts.append(f"{i}. ![]({url})")
            if description:
                parts.append(f"   *{description}*")
            parts.append("")

    return "\n".join(parts)


@tool
async def search_web_tavily(
    query: str,
    max_results: int = 5,
    include_images: bool = True,
    search_depth: Literal["basic", "advanced"] = "advanced",
) -> str:
    """使用 Tavily 搜索网络（支持图像）

    Tavily 是一个强大的搜索 API，支持：
    - AI 驱动的搜索结果
    - 图像搜索
    - 高级搜索模式
    - 域名过滤

    Args:
        query: 搜索查询关键词
        max_results: 最大结果数（1-10）
        include_images: 是否包含图像结果
        search_depth: 搜索深度（basic/advanced）

    Returns:
        搜索结果，包含答案、搜索结果和图像

    Examples:
        >>> results = await search_web_tavily("Python LangGraph")
        >>> results = await search_web_tavily("最新 AI 新闻", max_results=10)
    """
    if not _tavily_available:
        return "Tavily 搜索不可用，请安装: uv add tavily-python"

    api_key = settings.llm_api_key  # 使用统一的 API key 配置
    # 或者从专用配置读取
    # api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Tavily API Key 未配置，请设置 TAVILY_API_KEY 环境变量"

    # 限制 max_results 范围
    max_results = max(1, min(10, max_results))

    config = TavilySearchConfig(
        api_key=api_key,
        max_results=max_results,
        include_images=include_images,
        search_depth=search_depth,
    )

    logger.info(
        "tavily_search_start",
        query=query,
        max_results=max_results,
        include_images=include_images,
    )

    try:
        response = await _search_with_config(query, config)
        formatted = _format_tavily_results(response, query)

        logger.info(
            "tavily_search_complete",
            query=query,
            result_count=len(response.get("results", [])),
            image_count=len(response.get("images", [])),
        )

        return formatted

    except Exception as e:
        logger.error("tavily_search_failed", query=query, error=str(e))
        return f"搜索失败: {str(e)}"


# 同步版本（用于非异步上下文）
@tool
def search_web_tavily_sync(
    query: str,
    max_results: int = 5,
    include_images: bool = True,
) -> str:
    """使用 Tavily 搜索网络（同步版本）

    Args:
        query: 搜索查询关键词
        max_results: 最大结果数
        include_images: 是否包含图像

    Returns:
        搜索结果
    """
    import asyncio

    return asyncio.run(search_web_tavily(
        query=query,
        max_results=max_results,
        include_images=include_images,
    ))


class TavilySearchTool:
    """Tavily 搜索工具类

    提供更灵活的搜索配置选项。

    Examples:
        >>> from app.agent.tools.builtin.tavily_search import TavilySearchTool
        >>>
        >>> tool = TavilySearchTool(
        ...     api_key="your-key",
        ...     max_results=10,
        ...     include_images=True,
        ... )
        >>> results = await tool.search("LangGraph tutorial")
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = True,
        search_depth: Literal["basic", "advanced"] = "advanced",
        days: int = 3,
        post_processor: SearchResultPostProcessor | None = None,
    ):
        """初始化 Tavily 搜索工具

        Args:
            api_key: Tavily API 密钥
            max_results: 最大结果数
            include_answer: 是否包含答案
            include_raw_content: 是否包含原始内容
            include_images: 是否包含图像
            search_depth: 搜索深度
            days: 搜索天数范围
            post_processor: 搜索结果后处理器
        """
        self.config = TavilySearchConfig(
            api_key=api_key,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
            search_depth=search_depth,
            days=days,
        )
        self.post_processor = post_processor or SearchResultPostProcessor()

    async def search(self, query: str) -> str:
        """执行搜索

        Args:
            query: 搜索查询

        Returns:
            格式化的搜索结果
        """
        response = await _search_with_config(query, self.config)

        # 使用后处理器清理结果
        if "results" in response:
            cleaned_results = self.post_processor.process_results(response["results"])
            response["results"] = cleaned_results

        return _format_tavily_results(response, query)

    def create_langchain_tool(self):
        """创建 LangChain 工具

        Returns:
            LangChain 兼容的工具函数
        """
        async def _tool(query: str) -> str:
            return await self.search(query)

        _tool.description = self.__doc__ or "Search the web with Tavily"

        return _tool
