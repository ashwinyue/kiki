"""认证授权模块

提供 JWT 认证、API Key 管理、租户上下文等功能。
"""

from app.auth.api_key import (
    ApiKeyService,
    CurrentApiKey,
    has_scope,
    is_mcp_key,
    is_service_key,
    require_api_key,
    require_scope,
    verify_api_key,
    verify_api_key_or_token,
)
from app.auth.jwt import (
    Token,
    create_access_token,
    decode_token,
    get_current_user,
    get_current_user_id,
    get_token_sub,
    verify_token,
)
from app.auth.middleware import (
    RequiredTenantIdDep,
    TenantIdDep,
    TenantMiddleware,
)
from app.auth.tenant import TenantContext, get_tenant_context, set_tenant_context
from app.auth.tenant_api_key import (
    DecodedAPIKey,
    extract_tenant_id_from_api_key,
    generate_api_key,
    get_api_key_secret,
    validate_api_key,
)

__all__ = [
    # JWT
    "Token",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "get_current_user_id",
    "get_token_sub",
    "verify_token",
    # API Key
    "ApiKeyService",
    "CurrentApiKey",
    "verify_api_key",
    "verify_api_key_or_token",
    "require_api_key",
    "require_scope",
    "is_mcp_key",
    "is_service_key",
    "has_scope",
    # Tenant
    "TenantContext",
    "get_tenant_context",
    "set_tenant_context",
    # Tenant API Key
    "DecodedAPIKey",
    "generate_api_key",
    "extract_tenant_id_from_api_key",
    "validate_api_key",
    "get_api_key_secret",
    # Middleware
    "TenantMiddleware",
    "TenantIdDep",
    "RequiredTenantIdDep",
]
