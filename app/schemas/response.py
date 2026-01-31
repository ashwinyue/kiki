"""统一 API 响应模型

提供标准化的 API 响应格式，确保所有接口返回结构一致。
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from app.repositories.base import PaginatedResult

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应格式

    标准化所有 API 接口的返回结构，便于前端处理。

    Attributes:
        success: 请求是否成功
        data: 响应数据
        error: 错误信息（失败时）
        message: 提示信息
    """

    success: bool = Field(..., description="请求是否成功")
    data: T | None = Field(None, description="响应数据")
    error: str | None = Field(None, description="错误信息")
    message: str | None = Field(None, description="提示信息")

    @classmethod
    def ok(cls, data: T, message: str | None = None) -> "ApiResponse[T]":
        """创建成功响应

        Args:
            data: 响应数据
            message: 提示信息

        Returns:
            成功响应实例
        """
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(cls, error: str, message: str | None = None) -> "ApiResponse[T]":
        """创建失败响应

        Args:
            error: 错误信息
            message: 提示信息

        Returns:
            失败响应实例
        """
        return cls(success=False, data=None, error=error, message=message)


class PaginationMeta(BaseModel):
    """分页元数据"""

    total: int = Field(..., description="总记录数")
    page: int = Field(..., ge=1, description="当前页码")
    size: int = Field(..., ge=1, le=100, description="每页大小")
    pages: int = Field(..., ge=0, description="总页数")


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应

    用于返回分页列表数据。

    Attributes:
        data: 数据列表
        pagination: 分页元数据
    """

    data: list[T] = Field(default_factory=list, description="数据列表")
    pagination: PaginationMeta = Field(..., description="分页元数据")

    @classmethod
    def from_paginated_result(cls, result: PaginatedResult[T]) -> "PaginatedResponse[T]":
        """从 PaginatedResult 创建响应

        Args:
            result: 分页结果

        Returns:
            分页响应实例
        """
        return cls(
            data=result.items,
            pagination=PaginationMeta(
                total=result.total,
                page=result.page,
                size=result.size,
                pages=result.pages,
            ),
        )


# 便捷类型别名
DataResponse = ApiResponse[T]
ListResponse = ApiResponse[list[T]]
