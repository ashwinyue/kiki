"""中间件模块

包含：
- auth: 认证中间件（租户认证、JWT、API Key）
- observability: 可观测性中间件（日志、指标、请求上下文）
- security: 安全中间件（安全响应头、请求大小限制）
"""

from app.middleware.auth import (
    RequiredTenantIdDep,
    TenantContextDep,
    TenantIdDep,
    TenantMiddleware,
    get_current_tenant_id,
    get_current_user_dep,
    get_tenant_context,
    get_tenant_context_dep,
    get_tenant_id,
    get_tenant_id_dep,
    require_current_user,
    require_tenant,
)
from app.middleware.observability import (
    ObservabilityMiddleware,
    RequestContextMiddleware,
)
from app.middleware.security import (
    MaxRequestSizeMiddleware,
    SecurityHeadersMiddleware,
)

__all__ = [
    "TenantMiddleware",
    "get_tenant_context",
    "get_tenant_id_dep",
    "require_tenant",
    "TenantIdDep",
    "RequiredTenantIdDep",
    "TenantContextDep",
    "get_tenant_context_dep",
    "get_tenant_id",
    "get_current_user_dep",
    "require_current_user",
    "get_current_tenant_id",
    "ObservabilityMiddleware",
    "RequestContextMiddleware",
    "SecurityHeadersMiddleware",
    "MaxRequestSizeMiddleware",
]
