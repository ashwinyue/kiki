"""网络搜索 API 路由

提供网络搜索接口，支持多种搜索引擎。
对齐 WeKnora 的 internal/handler/web_search.go
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.response import ApiResponse
from app.schemas.web_search import (
    WebSearchCompressRequest,
    WebSearchCompressResponse,
    WebSearchProviderInfo,
    WebSearchProvidersResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from app.services.web.web_search import get_web_search_service

logger = get_logger(__name__)

router = APIRouter(prefix="/web-search", tags=["web-search"])


@router.post("/search", response_model=ApiResponse[WebSearchResponse])
@limiter.limit(RateLimit.API)
async def web_search(request: WebSearchRequest) -> ApiResponse[WebSearchResponse]:
    """执行 Web 搜索

    类似 WeKnora 的 /web-search/search 接口。

    Args:
        request: 搜索请求

    Returns:
        WebSearchResponse: 搜索结果

    Raises:
        HTTPException: 搜索失败时
    """
    try:
        service = get_web_search_service()

        # 获取提供商配置
        provider = request.provider
        if not provider or provider == "auto":
            provider = service.get_default_provider()

        # 构建搜索配置
        from app.schemas.web_search import WebSearchConfig

        config = WebSearchConfig(
            provider=provider,
            max_results=request.max_results or 5,
            include_date=request.include_date,
            blacklist=request.blacklist,
        )

        # 执行搜索
        results = await service.search(
            query=request.query,
            config=config,
        )

        response = WebSearchResponse(
            results=results,
            provider=provider,
            query=request.query,
            total=len(results),
        )

        logger.info(
            "web_search_api_success",
            query=request.query,
            provider=provider,
            result_count=len(results),
        )

        return ApiResponse.ok(data=response, message=f"找到 {len(results)} 条结果")

    except Exception as e:
        logger.exception(
            "web_search_api_error",
            query=request.query,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索失败: {str(e)}",
        ) from e


@router.get("/providers", response_model=ApiResponse[WebSearchProvidersResponse])
@limiter.limit(RateLimit.API)
async def get_providers() -> ApiResponse[WebSearchProvidersResponse]:
    """获取可用的 Web 搜索提供商列表

    类似 WeKnora 的 /web-search/providers 接口。

    Returns:
        WebSearchProvidersResponse: 提供商列表
    """
    try:
        service = get_web_search_service()
        providers = service.get_providers()
        default_provider = service.get_default_provider()

        response = WebSearchProvidersResponse(
            providers=providers,
            default_provider=default_provider,
        )

        logger.info(
            "web_search_providers_listed",
            provider_count=len(providers),
            default_provider=default_provider,
        )

        return ApiResponse.ok(data=response)

    except Exception as e:
        logger.exception(
            "web_search_providers_error",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取提供商列表失败: {str(e)}",
        ) from e


@router.post("/compress", response_model=ApiResponse[WebSearchCompressResponse])
@limiter.limit(RateLimit.API)
async def compress_search_results(
    request: WebSearchCompressRequest,
) -> ApiResponse[WebSearchCompressResponse]:
    """RAG 压缩搜索结果（预留接口）

    类似 WeKnora 的 /web-search/compress 接口。
    此接口预留用于未来的 RAG 压缩功能。

    Args:
        request: 压缩请求

    Returns:
        WebSearchCompressResponse: 压缩结果

    Raises:
        HTTPException: 压缩失败时
    """
    try:
        # 当前版本仅返回原始结果
        # RAG 压缩功能将在后续版本实现
        response = WebSearchCompressResponse(
            results=request.results,
            temp_kb_id=request.temp_kb_id,
            knowledge_ids=[],
        )

        logger.info(
            "web_search_compress_not_implemented",
            session_id=request.session_id,
            result_count=len(request.results),
        )

        return ApiResponse.ok(
            data=response,
            message="RAG 压缩功能尚未实现，返回原始结果",
        )

    except Exception as e:
        logger.exception(
            "web_search_compress_error",
            session_id=request.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"压缩失败: {str(e)}",
        ) from e


# 为了保持与 chat.py 中现有的 SearchProviderInfo 兼容
# 导出一个兼容的包装函数
async def get_search_providers_for_chat() -> list[dict[str, Any]]:
    """获取搜索提供商信息（兼容 chat.py）

    Returns:
        提供商信息字典列表
    """
    service = get_web_search_service()
    providers = service.get_providers()

    return [
        {
            "name": p.id,
            "display_name": p.name,
            "available": p.available,
            "requires_api_key": p.requires_api_key,
            "supported_depths": ["basic", "advanced"] if p.id == "tavily" else ["basic"],
            "description": p.description,
        }
        for p in providers
    ]
