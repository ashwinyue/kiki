"""网络搜索服务

提供统一的网络搜索服务，支持多提供商切换和黑名单过滤。
对齐 WeKnora 的 internal/application/service/web_search.go
"""

from __future__ import annotations

import os
from typing import Any

from app.observability.logging import get_logger
from app.schemas.web_search import (
    WebSearchConfig,
    WebSearchProviderInfo,
    WebSearchResult,
)
from app.services.web_search_providers import (
    BaseWebSearchProvider,
    BingProvider,
    DuckDuckGoProvider,
    GoogleProvider,
    TavilyProvider,
    filter_blacklist,
)

logger = get_logger(__name__)


class WebSearchProviderRegistry:
    """Web 搜索提供商注册表

    对应 WeKnora 的 web_search.Registry
    """

    def __init__(self) -> None:
        """初始化注册表"""
        self._providers: dict[str, type[BaseWebSearchProvider]] = {}
        self._instances: dict[str, BaseWebSearchProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """注册默认提供商"""
        self.register("duckduckgo", DuckDuckGoProvider)
        self.register("tavily", TavilyProvider)
        self.register("google", GoogleProvider)
        self.register("bing", BingProvider)

    def register(
        self,
        provider_id: str,
        provider_class: type[BaseWebSearchProvider],
    ) -> None:
        """注册搜索提供商

        Args:
            provider_id: 提供商ID
            provider_class: 提供商类
        """
        self._providers[provider_id] = provider_class
        logger.info(
            "web_search_provider_registered",
            provider_id=provider_id,
        )

    def get_provider(self, provider_id: str) -> BaseWebSearchProvider | None:
        """获取提供商实例

        Args:
            provider_id: 提供商ID

        Returns:
            提供商实例或 None
        """
        # 如果已存在实例，直接返回
        if provider_id in self._instances:
            return self._instances[provider_id]

        # 获取提供商类
        provider_class = self._providers.get(provider_id)
        if provider_class is None:
            logger.warning(
                "web_search_provider_not_found",
                provider_id=provider_id,
            )
            return None

        # 创建实例
        try:
            instance = self._create_provider_instance(provider_id, provider_class)
            if instance:
                self._instances[provider_id] = instance
            return instance
        except Exception as e:
            logger.exception(
                "web_search_provider_creation_failed",
                provider_id=provider_id,
                error=str(e),
            )
            return None

    def _create_provider_instance(
        self,
        provider_id: str,
        provider_class: type[BaseWebSearchProvider],
    ) -> BaseWebSearchProvider | None:
        """创建提供商实例

        Args:
            provider_id: 提供商ID
            provider_class: 提供商类

        Returns:
            提供商实例或 None
        """
        if provider_id == "duckduckgo":
            return provider_class()

        if provider_id == "tavily":
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning("tavily_api_key_not_set")
                return None
            return provider_class(api_key=api_key)

        if provider_id == "google":
            api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
            engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
            if not api_key or not engine_id:
                logger.warning("google_search_credentials_not_set")
                return None
            return provider_class(api_key=api_key, engine_id=engine_id)

        if provider_id == "bing":
            api_key = os.getenv("BING_SEARCH_API_KEY")
            if not api_key:
                logger.warning("bing_search_api_key_not_set")
                return None
            return provider_class(api_key=api_key)

        return None

    def get_all_providers_info(self) -> list[WebSearchProviderInfo]:
        """获取所有提供商信息

        Returns:
            提供商信息列表
        """
        providers_info = []

        for provider_id, provider_class in self._providers.items():
            # 创建临时实例获取信息
            try:
                if provider_id == "duckduckgo":
                    instance = provider_class()
                elif provider_id == "tavily":
                    api_key = os.getenv("TAVILY_API_KEY")
                    instance = provider_class(api_key=api_key or "")
                elif provider_id == "google":
                    api_key = os.getenv("GOOGLE_SEARCH_API_KEY") or ""
                    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID") or ""
                    instance = provider_class(api_key=api_key, engine_id=engine_id)
                elif provider_id == "bing":
                    api_key = os.getenv("BING_SEARCH_API_KEY") or ""
                    instance = provider_class(api_key=api_key)
                else:
                    continue

                info_dict = instance.get_info()
                info_dict["available"] = self._is_provider_available(provider_id)
                providers_info.append(WebSearchProviderInfo(**info_dict))
            except Exception:
                # 如果创建失败，至少返回基本信息
                providers_info.append(
                    WebSearchProviderInfo(
                        id=provider_id,
                        name=provider_id.capitalize(),
                        free=provider_id == "duckduckgo",
                        requires_api_key=provider_id != "duckduckgo",
                        description=f"{provider_id} search provider",
                        available=False,
                    )
                )

        return providers_info

    def _is_provider_available(self, provider_id: str) -> bool:
        """检查提供商是否可用

        Args:
            provider_id: 提供商ID

        Returns:
            是否可用
        """
        if provider_id == "duckduckgo":
            try:
                import duckduckgo_search  # noqa: F401

                return True
            except ImportError:
                return False

        if provider_id == "tavily":
            return bool(os.getenv("TAVILY_API_KEY"))

        if provider_id == "google":
            return bool(
                os.getenv("GOOGLE_SEARCH_API_KEY")
                and os.getenv("GOOGLE_SEARCH_ENGINE_ID")
            )

        if provider_id == "bing":
            return bool(os.getenv("BING_SEARCH_API_KEY"))

        return False

    def get_default_provider(self) -> str:
        """获取默认提供商

        Returns:
            默认提供商ID
        """
        # 优先选择可用的免费提供商
        if self._is_provider_available("duckduckgo"):
            return "duckduckgo"
        if self._is_provider_available("tavily"):
            return "tavily"

        # 返回第一个注册的提供商
        for provider_id in self._providers:
            if self._is_provider_available(provider_id):
                return provider_id

        return "duckduckgo"


class WebSearchService:
    """Web 搜索服务

    对应 WeKnora 的 WebSearchService
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """初始化搜索服务

        Args:
            timeout: 搜索超时时间（秒）
        """
        self.timeout = timeout
        self.registry = WebSearchProviderRegistry()

    async def search(
        self,
        query: str,
        config: WebSearchConfig | None = None,
    ) -> list[WebSearchResult]:
        """执行 Web 搜索

        Args:
            query: 搜索查询
            config: 搜索配置

        Returns:
            搜索结果列表
        """
        if not query or not query.strip():
            logger.warning("empty_search_query")
            return []

        # 使用默认配置
        if config is None:
            config = WebSearchConfig()

        # 确定使用的提供商
        provider_id = config.provider
        if not provider_id or provider_id == "auto":
            provider_id = self.registry.get_default_provider()

        # 获取提供商实例
        provider = self.registry.get_provider(provider_id)
        if provider is None:
            logger.error(
                "web_search_provider_not_available",
                provider_id=provider_id,
            )
            return []

        # 执行搜索
        logger.info(
            "web_search_start",
            query=query,
            provider=provider_id,
            max_results=config.max_results,
        )

        try:
            results = await provider.search(
                query=query,
                max_results=config.max_results,
                include_date=config.include_date,
            )

            # 应用黑名单过滤
            if config.blacklist:
                results = filter_blacklist(results, config.blacklist)

            logger.info(
                "web_search_complete",
                query=query,
                provider=provider_id,
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.exception(
                "web_search_failed",
                query=query,
                provider=provider_id,
                error=str(e),
            )
            return []

    def get_providers(self) -> list[WebSearchProviderInfo]:
        """获取可用提供商列表

        Returns:
            提供商信息列表
        """
        return self.registry.get_all_providers_info()

    def get_default_provider(self) -> str:
        """获取默认提供商

        Returns:
            默认提供商ID
        """
        return self.registry.get_default_provider()


# 全局服务实例
_web_search_service: WebSearchService | None = None


def get_web_search_service() -> WebSearchService:
    """获取 Web 搜索服务实例（单例）

    Returns:
        WebSearchService 实例
    """
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service
