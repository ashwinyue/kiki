"""FAQ 相关模式

提供 API 请求和响应的数据结构定义。
"""

from typing import Any

from pydantic import BaseModel, Field

from app.models.faq import (
    FAQCategory,
    FAQDetail,
    FAQRead,
    FAQSearchResult,
    FAQStatus,
)
from app.repositories.base import PaginatedResult

# ============== 请求模式 ==============


class FAQListRequest(BaseModel):
    """FAQ 列表请求参数"""

    category: FAQCategory | None = Field(None, description="按分类筛选")
    status: FAQStatus | None = Field(None, description="按状态筛选")
    locale: str | None = Field(None, description="按语言筛选")
    tags: list[str] | None = Field(None, description="按标签筛选")
    search: str | None = Field(None, description="搜索关键词")
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页数量")
    sort_by: str = Field("priority", description="排序字段")
    sort_order: str = Field("desc", description="排序方向: asc/desc")


class FAQSearchRequest(BaseModel):
    """FAQ 搜索请求"""

    query: str = Field(..., min_length=1, description="搜索查询")
    category: FAQCategory | None = Field(None, description="按分类筛选")
    locale: str = Field("zh-CN", description="语言")
    limit: int = Field(10, ge=1, le=50, description="返回数量限制")
    include_answer: bool = Field(True, description="是否包含答案")


# ============== 导出请求模式 ==============


class FAQExportRequest(BaseModel):
    """FAQ 导出请求参数"""

    format: str = Field("csv", description="导出格式: csv, json, excel")
    status: FAQStatus | None = Field(None, description="按状态筛选")
    category: FAQCategory | None = Field(None, description="按分类筛选")
    locale: str | None = Field(None, description="按语言筛选")


# ============== 响应模式 ==============


class FAQResponse(BaseModel):
    """FAQ 响应"""

    success: bool
    data: FAQRead | None = None
    error: str | None = None
    message: str | None = None


class FAQDetailResponse(BaseModel):
    """FAQ 详情响应"""

    success: bool
    data: FAQDetail | None = None
    error: str | None = None
    message: str | None = None


class FAQListResponse(BaseModel):
    """FAQ 列表响应"""

    items: list[FAQRead]
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def from_paginated_result(cls, result: PaginatedResult) -> "FAQListResponse":
        """从分页结果创建响应

        Args:
            result: 分页结果

        Returns:
            FAQ 列表响应
        """
        return cls(
            items=result.items,
            total=result.total,
            page=result.page,
            size=result.size,
            pages=result.pages,
        )


class FAQSearchResponse(BaseModel):
    """FAQ 搜索响应"""

    query: str
    results: list[FAQSearchResult]
    total: int
    took_ms: int | None = Field(None, description="查询耗时（毫秒）")


class FAQExportResponse(BaseModel):
    """FAQ 导出响应（用于异步任务）"""

    success: bool
    task_id: str | None = Field(None, description="导出任务 ID")
    message: str | None = None


# ============== 反馈模式 ==============


class FAQFeedbackResponse(BaseModel):
    """FAQ 反馈响应"""

    success: bool
    helpful_count: int
    not_helpful_count: int
    message: str | None = None


# ============== 统计模式 ==============


class FAQStatsResponse(BaseModel):
    """FAQ 统计响应"""

    total: int
    by_status: dict[str, int]
    by_category: dict[str, int]
    by_locale: dict[str, int]
    most_viewed: list[dict[str, Any]]
    most_helpful: list[dict[str, Any]]


# ============== 批量操作响应 ==============


class FAQBulkUpdateResponse(BaseModel):
    """批量更新响应"""

    success: bool
    updated_count: int
    failed_ids: list[int]
    message: str | None = None
