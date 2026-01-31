"""租户相关模式"""

from typing import Any

from pydantic import BaseModel, Field


class TenantListResponse(BaseModel):
    """租户列表响应"""

    items: list[dict]
    total: int
    page: int = 1
    size: int = 20


class ApiKeyResponse(BaseModel):
    """API Key 响应"""

    api_key: str
    tenant_id: int


class TenantSearchRequest(BaseModel):
    """租户搜索请求"""

    keyword: str | None = Field(None, description="搜索关键词（名称、描述、业务类型）")
    status: str | None = Field(None, description="状态筛选")
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页数量")


class TenantItem(BaseModel):
    """租户列表项"""

    id: int
    name: str
    description: str | None = None
    status: str
    business: str
    storage_quota: int
    storage_used: int
    created_at: Any
    updated_at: Any

    model_config = {"from_attributes": True}


class TenantSearchResponse(BaseModel):
    """租户搜索响应"""

    items: list[TenantItem]
    total: int
    page: int
    page_size: int
