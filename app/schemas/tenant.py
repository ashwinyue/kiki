"""租户相关模式"""

from pydantic import BaseModel


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
