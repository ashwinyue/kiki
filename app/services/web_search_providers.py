"""网络搜索提供商抽象

提供统一的搜索提供商接口，支持多种搜索引擎实现。
对齐 WeKnora 的 internal/application/service/web_search/ 目录
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

from app.observability.logging import get_logger
from app.schemas.web_search import WebSearchResult

logger = get_logger(__name__)


class BaseWebSearchProvider(ABC):
    """Web 搜索提供商抽象基类

    对应 WeKnora 的 interfaces.WebSearchProvider
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """初始化提供商

        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> BaseWebSearchProvider:
        """异步上下文管理器入口"""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """异步上下文管理器退出"""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_date: bool = False,
    ) -> list[WebSearchResult]:
        """执行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            include_date: 是否包含日期

        Returns:
            搜索结果列表
        """

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """获取提供商信息

        Returns:
            提供商信息字典
        """


class DuckDuckGoProvider(BaseWebSearchProvider):
    """DuckDuckGo 搜索提供商

    使用 duckduckgo-search 库实现免费搜索。
    对应 WeKnora 的 DuckDuckGoProvider
    """

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__(timeout)
        self._ddgs: Any | None = None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_date: bool = False,  # noqa: ARG002
    ) -> list[WebSearchResult]:
        """执行 DuckDuckGo 搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            include_date: 是否包含日期（暂不支持）

        Returns:
            搜索结果列表
        """
        try:
            from duckduckgo_search import DDGS

            if self._ddgs is None:
                self._ddgs = DDGS()

            # 在线程池中执行同步调用
            loop = asyncio.get_event_loop()
            results_data = await loop.run_in_executor(
                None,
                lambda: self._ddgs.text(  # type: ignore[attr-defined]
                    query,
                    max_results=max_results,
                ),
            )

            if not results_data:
                logger.warning(
                    "duckduckgo_no_results",
                    query=query,
                )
                return []

            results = [
                WebSearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("link", "")),
                    snippet=str(item.get("body", "")),
                    source="duckduckgo",
                )
                for item in results_data
            ]

            logger.info(
                "duckduckgo_search_success",
                query=query,
                result_count=len(results),
            )

            return results

        except ImportError as e:
            logger.error(
                "duckduckgo_import_error",
                error=str(e),
            )
            raise RuntimeError(
                "duckduckgo-search 包未安装，请使用: uv add duckduckgo-search"
            ) from e
        except Exception as e:
            logger.exception(
                "duckduckgo_search_failed",
                query=query,
                error=str(e),
            )
            return []

    def get_info(self) -> dict[str, Any]:
        """获取提供商信息"""
        return {
            "id": "duckduckgo",
            "name": "DuckDuckGo",
            "free": True,
            "requires_api_key": False,
            "description": "DuckDuckGo Search API - 免费搜索引擎",
        }


class TavilyProvider(BaseWebSearchProvider):
    """Tavily 搜索提供商

    使用 Tavily API 实现专业搜索。
    对应 WeKnora 的 TavilyProvider（预留）
    """

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        super().__init__(timeout)
        self.api_key = api_key
        self._client_tavily: Any | None = None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_date: bool = False,  # noqa: ARG002
    ) -> list[WebSearchResult]:
        """执行 Tavily 搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            include_date: 是否包含日期

        Returns:
            搜索结果列表
        """
        try:
            from tavily import TavilyClient

            if self._client_tavily is None:
                self._client_tavily = TavilyClient(api_key=self.api_key)

            # 在线程池中执行同步调用
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client_tavily.search(
                    query=query,
                    max_results=max_results,
                    search_depth="basic",
                    include_raw_content=False,
                ),
            )

            results = []
            for item in response.get("results", []):
                # 解析发布日期
                published_at = None
                if item.get("published_date"):
                    try:
                        published_at = datetime.fromisoformat(
                            item["published_date"].replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                results.append(
                    WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        content=item.get("content"),
                        source="tavily",
                        published_at=published_at,
                    )
                )

            logger.info(
                "tavily_search_success",
                query=query,
                result_count=len(results),
            )

            return results

        except ImportError as e:
            logger.error(
                "tavily_import_error",
                error=str(e),
            )
            raise RuntimeError(
                "tavily-python 包未安装，请使用: uv add tavily-python"
            ) from e
        except Exception as e:
            logger.exception(
                "tavily_search_failed",
                query=query,
                error=str(e),
            )
            return []

    def get_info(self) -> dict[str, Any]:
        """获取提供商信息"""
        return {
            "id": "tavily",
            "name": "Tavily",
            "free": False,
            "requires_api_key": True,
            "description": "Tavily Search API - 专业搜索引擎",
            "api_url": "https://api.tavily.com",
        }


class GoogleProvider(BaseWebSearchProvider):
    """Google 搜索提供商（预留）

    使用 Google Custom Search JSON API。
    对应 WeKnora 的 GoogleProvider
    """

    def __init__(self, api_key: str, engine_id: str, timeout: float = 30.0) -> None:
        super().__init__(timeout)
        self.api_key = api_key
        self.engine_id = engine_id

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_date: bool = False,  # noqa: ARG002
    ) -> list[WebSearchResult]:
        """执行 Google 搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            include_date: 是否包含日期

        Returns:
            搜索结果列表
        """
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.engine_id,
                "q": query,
                "num": min(max_results, 10),  # Google API 限制每次最多 10 个结果
            }

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append(
                    WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                    )
                )

            logger.info(
                "google_search_success",
                query=query,
                result_count=len(results),
            )

            return results

        except httpx.HTTPStatusError as e:
            logger.exception(
                "google_search_http_error",
                query=query,
                status_code=e.response.status_code,
            )
            return []
        except Exception as e:
            logger.exception(
                "google_search_failed",
                query=query,
                error=str(e),
            )
            return []

    def get_info(self) -> dict[str, Any]:
        """获取提供商信息"""
        return {
            "id": "google",
            "name": "Google",
            "free": False,
            "requires_api_key": True,
            "description": "Google Custom Search API",
            "api_url": "https://www.googleapis.com/customsearch/v1",
        }


class BingProvider(BaseWebSearchProvider):
    """Bing 搜索提供商（预留）

    使用 Bing Web Search API v7。
    """

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        super().__init__(timeout)
        self.api_key = api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_date: bool = False,  # noqa: ARG002
    ) -> list[WebSearchResult]:
        """执行 Bing 搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            include_date: 是否包含日期

        Returns:
            搜索结果列表
        """
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {
                "q": query,
                "count": min(max_results, 50),  # Bing API 限制每次最多 50 个结果
            }

            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append(
                    WebSearchResult(
                        title=item.get("name", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        source="bing",
                    )
                )

            logger.info(
                "bing_search_success",
                query=query,
                result_count=len(results),
            )

            return results

        except httpx.HTTPStatusError as e:
            logger.exception(
                "bing_search_http_error",
                query=query,
                status_code=e.response.status_code,
            )
            return []
        except Exception as e:
            logger.exception(
                "bing_search_failed",
                query=query,
                error=str(e),
            )
            return []

    def get_info(self) -> dict[str, Any]:
        """获取提供商信息"""
        return {
            "id": "bing",
            "name": "Bing",
            "free": False,
            "requires_api_key": True,
            "description": "Bing Web Search API v7",
            "api_url": "https://api.bing.microsoft.com/v7.0/search",
        }


def matches_blacklist_rule(url: str, rule: str) -> bool:
    """检查 URL 是否匹配黑名单规则

    支持通配符匹配（如 *://*.example.com/*）和正则表达式（如 /example\\.(net|org)/）

    Args:
        url: 要检查的 URL
        rule: 黑名单规则

    Returns:
        是否匹配
    """
    # 检查是否为正则表达式（以 / 开头和结尾）
    if rule.startswith("/") and rule.endswith("/"):
        pattern = rule[1:-1]
        try:
            return bool(re.search(pattern, url))
        except re.error:
            logger.warning(
                "invalid_regex_pattern",
                rule=rule,
            )
            return False

    # 通配符匹配（如 *://*.example.com/*）
    pattern = re.escape(rule).replace(r"\*", ".*")
    pattern = f"^{pattern}$"
    try:
        return bool(re.search(pattern, url))
    except re.error:
        logger.warning(
            "invalid_pattern",
            rule=rule,
        )
        return False


def filter_blacklist(
    results: list[WebSearchResult],
    blacklist: list[str],
) -> list[WebSearchResult]:
    """根据黑名单过滤搜索结果

    Args:
        results: 搜索结果列表
        blacklist: 黑名单规则列表

    Returns:
        过滤后的结果列表
    """
    if not blacklist:
        return results

    filtered = [
        result
        for result in results
        if not any(matches_blacklist_rule(result.url, rule) for rule in blacklist)
    ]

    if len(filtered) < len(results):
        logger.info(
            "blacklist_filtered",
            original_count=len(results),
            filtered_count=len(filtered),
            removed_count=len(results) - len(filtered),
        )

    return filtered
