"""FAQ 管理 API

提供 FAQ 的创建、列表、查询、更新、删除、搜索、导出等功能。
使用 Service 层处理业务逻辑，API 层仅负责请求/响应处理。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response
from starlette.requests import Request as StarletteRequest

from app.api.v1.auth import get_current_user_id as get_current_user_id_int
from app.models.faq import (
    FAQBulkUpdate,
    FAQCategory,
    FAQCreate,
    FAQFeedbackCreate,
    FAQReorderRequest,
    FAQStatus,
    FAQUpdate,
)
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.faq import (
    FAQBulkUpdateResponse,
    FAQFeedbackResponse,
    FAQListResponse,
    FAQSearchRequest,
    FAQSearchResponse,
    FAQStatsResponse,
)
from app.services.faq_export import ExportFormat, FAQExporter, get_faq_exporter
from app.services.faq_service import FAQService, get_faq_service

router = APIRouter(prefix="/faq", tags=["FAQ"])
logger = get_logger(__name__)


# ============== FAQ CRUD 接口 ==============


@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="创建 FAQ",
    description="创建一个新的常见问题",
    responses={
        status.HTTP_201_CREATED: {"description": "FAQ 创建成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_409_CONFLICT: {"description": "Slug 已存在"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "请求过于频繁"},
    },
)
@limiter.limit(RateLimit.API)
async def create_faq(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    data: FAQCreate,
) -> dict:
    """创建 FAQ"""
    result = await service.create_faq(data, user_id)
    return {"success": True, "data": result}


@router.get(
    "",
    response_model=FAQListResponse,
    summary="列出 FAQ",
    description="分页列出 FAQ，支持按分类、状态、语言、标签筛选",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 FAQ 列表"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def list_faqs(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    page: int = 1,
    size: int = 20,
    status: FAQStatus | None = None,
    category: FAQCategory | None = None,
    locale: str | None = None,
    tags: str | None = None,  # 逗号分隔的标签
    search: str | None = None,
) -> FAQListResponse:
    """列出 FAQ"""
    tag_list = tags.split(",") if tags else None
    return await service.list_faqs(
        page=page,
        size=size,
        status=status,
        category=category,
        locale=locale,
        tags=tag_list,
        search=search,
    )


@router.get(
    "/{faq_id}",
    response_model=dict,
    summary="获取 FAQ 详情",
    description="获取指定 FAQ 的详细信息",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 FAQ 详情"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_404_NOT_FOUND: {"description": "FAQ 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def get_faq(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    faq_id: int,
) -> dict:
    """获取 FAQ 详情"""
    result = await service.get_faq(faq_id)
    return {"success": True, "data": result}


@router.get(
    "/slug/{slug}",
    response_model=dict,
    summary="根据 slug 获取 FAQ",
    description="通过 URL 友好的标识符获取 FAQ",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 FAQ 详情"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_404_NOT_FOUND: {"description": "FAQ 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def get_faq_by_slug(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    slug: str,
) -> dict:
    """根据 slug 获取 FAQ"""
    result = await service.get_faq_by_slug(slug)
    return {"success": True, "data": result}


@router.patch(
    "/{faq_id}",
    response_model=dict,
    summary="更新 FAQ",
    description="更新 FAQ 的内容、分类、状态等",
    responses={
        status.HTTP_200_OK: {"description": "FAQ 更新成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_404_NOT_FOUND: {"description": "FAQ 不存在"},
        status.HTTP_409_CONFLICT: {"description": "Slug 已存在"},
    },
)
@limiter.limit(RateLimit.API)
async def update_faq(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    faq_id: int,
    data: FAQUpdate,
) -> dict:
    """更新 FAQ"""
    result = await service.update_faq(faq_id, data, user_id)
    return {"success": True, "data": result}


@router.delete(
    "/{faq_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除 FAQ",
    description="删除指定的 FAQ",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "FAQ 删除成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_404_NOT_FOUND: {"description": "FAQ 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def delete_faq(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    faq_id: int,
) -> None:
    """删除 FAQ"""
    await service.delete_faq(faq_id)


# ============== 搜索接口 ==============


@router.post(
    "/search",
    response_model=FAQSearchResponse,
    summary="搜索 FAQ",
    description="在问题和答案中搜索关键词",
    responses={
        status.HTTP_200_OK: {"description": "成功返回搜索结果"},
        status.HTTP_400_BAD_REQUEST: {"description": "请求参数验证失败"},
    },
)
@limiter.limit(RateLimit.API)
async def search_faqs(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    data: FAQSearchRequest,
) -> FAQSearchResponse:
    """搜索 FAQ"""
    return await service.search_faqs(
        query=data.query,
        locale=data.locale,
        category=data.category,
        limit=data.limit,
    )


# ============== 反馈接口 ==============


@router.post(
    "/{faq_id}/feedback",
    response_model=FAQFeedbackResponse,
    summary="FAQ 反馈",
    description="提交对 FAQ 的有用性反馈",
    responses={
        status.HTTP_200_OK: {"description": "反馈提交成功"},
        status.HTTP_404_NOT_FOUND: {"description": "FAQ 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def add_faq_feedback(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    faq_id: int,
    data: FAQFeedbackCreate,
) -> FAQFeedbackResponse:
    """提交 FAQ 反馈"""
    return await service.add_feedback(faq_id, data.helpful)


# ============== 统计接口 ==============


@router.get(
    "/stats/overview",
    response_model=FAQStatsResponse,
    summary="FAQ 统计",
    description="获取 FAQ 的统计信息，包括总数、分类分布、热门 FAQ 等",
    responses={
        status.HTTP_200_OK: {"description": "成功返回统计信息"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def get_faq_stats(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
) -> FAQStatsResponse:
    """获取 FAQ 统计"""
    return await service.get_stats()


# ============== 批量操作接口 ==============


@router.post(
    "/bulk/status",
    response_model=FAQBulkUpdateResponse,
    summary="批量更新 FAQ 状态",
    description="批量更新多个 FAQ 的状态",
    responses={
        status.HTTP_200_OK: {"description": "批量更新成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def bulk_update_faq_status(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    data: FAQBulkUpdate,
) -> FAQBulkUpdateResponse:
    """批量更新 FAQ 状态"""
    if data.status is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status is required for bulk update",
        )

    return await service.bulk_update_status(
        faq_ids=data.ids,
        status=data.status,
        user_id=user_id,
    )


@router.post(
    "/bulk/reorder",
    response_model=FAQBulkUpdateResponse,
    summary="重新排序 FAQ",
    description="批量更新 FAQ 的优先级顺序",
    responses={
        status.HTTP_200_OK: {"description": "重新排序成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def reorder_faqs(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    data: FAQReorderRequest,
) -> FAQBulkUpdateResponse:
    """重新排序 FAQ"""
    return await service.reorder_faqs(
        id_orders=data.id_orders,
        user_id=user_id,
    )


# ============== 公开接口（无需认证） ==============


@router.get(
    "/public/published",
    response_model=FAQListResponse,
    summary="获取已发布的 FAQ（公开）",
    description="获取已发布的 FAQ 列表，无需认证",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 FAQ 列表"},
    },
)
@limiter.limit(RateLimit.API)
async def list_published_faqs(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    page: int = 1,
    size: int = 20,
    category: FAQCategory | None = None,
    locale: str = "zh-CN",
) -> FAQListResponse:
    """获取已发布的 FAQ（公开接口）"""
    return await service.list_faqs(
        page=page,
        size=size,
        status=FAQStatus.PUBLISHED,
        category=category,
        locale=locale,
    )


@router.get(
    "/public/{faq_id}",
    response_model=dict,
    summary="获取 FAQ 详情（公开）",
    description="获取指定 FAQ 的详细信息，无需认证",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 FAQ 详情"},
        status.HTTP_404_NOT_FOUND: {"description": "FAQ 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def get_published_faq(
    request: StarletteRequest,
    service: Annotated[FAQService, Depends(get_faq_service)],
    faq_id: int,
) -> dict:
    """获取已发布的 FAQ 详情（公开接口）"""
    result = await service.get_faq(faq_id, increment_view=True)
    if result.status != FAQStatus.PUBLISHED:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAQ not found or not published",
        )
    return {"success": True, "data": result}


# ============== 导出接口 ==============


@router.get(
    "/export",
    summary="导出 FAQ",
    description="将 FAQ 导出为 CSV、JSON 或 Excel 格式",
    responses={
        status.HTTP_200_OK: {
            "description": "成功返回导出文件",
            "content": {
                "text/csv": {"schema": {"type": "string", "format": "binary"}},
                "application/json": {"schema": {"type": "string", "format": "binary"}},
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
                    "schema": {"type": "string", "format": "binary"}
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "请求参数错误"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def export_faqs(
    request: StarletteRequest,
    exporter: Annotated[FAQExporter, Depends(get_faq_exporter)],
    format: Annotated[
        ExportFormat,
        Query(description="导出格式: csv, json, excel"),
    ] = ExportFormat.CSV,
    status: Annotated[FAQStatus | None, Query(description="按状态筛选")] = None,
    category: Annotated[
        FAQCategory | None,
        Query(description="按分类筛选"),
    ] = None,
    locale: Annotated[str | None, Query(description="按语言筛选")] = None,
) -> Response:
    """导出 FAQ

    对齐 WeKnora 的 FAQ 导出 API，支持 CSV、JSON、Excel 格式。
    CSV 格式会添加 BOM 头以确保 Excel 兼容性。
    """
    if format == ExportFormat.CSV:
        filename, content = await exporter.export_to_csv(
            status=status,
            category=category,
            locale=locale,
        )
        # 添加 UTF-8 BOM 以确保 Excel 正确识别中文
        bom = b"\xEF\xBB\xBF"
        return Response(
            content=bom + content.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    elif format == ExportFormat.JSON:
        filename, content = await exporter.export_to_json(
            status=status,
            category=category,
            locale=locale,
        )
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    else:  # EXCEL
        filename, content = await exporter.export_to_excel(
            status=status,
            category=category,
            locale=locale,
        )
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )
