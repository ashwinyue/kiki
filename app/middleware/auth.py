"""认证中间件

支持 JWT Token 和 X-API-Key 两种认证方式。
"""

from typing import Annotated

from fastapi import Depends, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.auth.tenant_api_key import extract_tenant_id_from_api_key
from app.auth.jwt import verify_token
from app.auth.tenant import TenantContext, set_tenant_context, clear_tenant_context
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 无需认证的路径
NO_AUTH_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
}


class TenantMiddleware(BaseHTTPMiddleware):
    """租户认证中间件

    功能：
    1. JWT Token 认证：从 Authorization 头提取用户和租户信息
    2. X-API-Key 认证：从 API Key 解密获取租户 ID
    3. 跨租户访问：支持 X-Tenant-ID 头进行跨租户访问（需权限）
    4. 租户上下文：将租户信息绑定到 ContextVar
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enable_cross_tenant: bool = False,
    ) -> None:
        super().__init__(app)
        self.enable_cross_tenant = enable_cross_tenant

    async def dispatch(self, request: Request, call_next):
        # 清除之前的上下文
        clear_tenant_context()

        # 跳过无需认证的路径
        if request.url.path in NO_AUTH_PATHS:
            return await call_next(request)

        # 跳过 OPTIONS 请求
        if request.method == "OPTIONS":
            return await call_next(request)

        authenticated = False

        # 尝试 JWT Token 认证
        authorization = request.headers.get("authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            try:
                payload = verify_token(token)
            except ValueError:
                payload = None

            if payload:
                authenticated = await self._handle_jwt_auth(request, payload)
                if authenticated:
                    return await call_next(request)

        # 尝试 X-API-Key 认证
        api_key = request.headers.get("x-api-key")
        if api_key:
            authenticated = await self._handle_api_key_auth(request, api_key)
            if authenticated:
                return await call_next(request)

        # 未提供认证信息 - 继续处理，由具体端点决定是否需要认证
        return await call_next(request)

    async def _handle_jwt_auth(self, request: Request, payload: dict) -> bool:
        """处理 JWT Token 认证"""
        from app.infra.database import session_scope
        from app.models.database import User, Tenant

        user_id = payload.get("sub")
        if not user_id:
            return False

        try:
            async with session_scope() as session:
                # 获取用户信息
                from sqlalchemy import select

                user_stmt = select(User).where(User.id == int(user_id))
                user_result = await session.execute(user_stmt)
                user = user_result.scalar_one_or_none()

                if not user:
                    logger.warning("user_not_found", user_id=user_id)
                    return False

                # 确定目标租户 ID
                target_tenant_id = user.tenant_id

                # 检查跨租户访问
                if (
                    self.enable_cross_tenant
                    and user.can_access_all_tenants
                    and user.tenant_id
                ):
                    tenant_header = request.headers.get("x-tenant-id")
                    if tenant_header:
                        try:
                            target_tenant_id = int(tenant_header)
                            # 验证目标租户存在
                            tenant_stmt = select(Tenant).where(
                                Tenant.id == target_tenant_id
                            )
                            tenant_result = await session.execute(tenant_stmt)
                            target_tenant = tenant_result.scalar_one_or_none()

                            if not target_tenant:
                                logger.warning(
                                    "target_tenant_not_found", tenant_id=target_tenant_id
                                )
                                return False

                            logger.info(
                                "cross_tenant_access",
                                user_id=user_id,
                                from_tenant_id=user.tenant_id,
                                to_tenant_id=target_tenant_id,
                            )
                        except ValueError:
                            pass

                # 获取租户信息
                if target_tenant_id:
                    tenant_stmt = select(Tenant).where(Tenant.id == target_tenant_id)
                    tenant_result = await session.execute(tenant_stmt)
                    tenant = tenant_result.scalar_one_or_none()

                    if tenant:
                        # 设置租户上下文
                        set_tenant_context(
                            TenantContext(
                                tenant_id=tenant.id,
                                tenant_name=tenant.name,
                                api_key=tenant.api_key,
                                config=tenant.config or {},
                            )
                        )
                        # 设置到 request.state 供依赖注入使用
                        request.state.tenant_id = tenant.id
                        request.state.user_id = user.id
                        request.state.tenant = tenant
                        return True

                return False
        except Exception as e:
            logger.error("jwt_auth_error", error=str(e))
            return False

    async def _handle_api_key_auth(self, request: Request, api_key: str) -> bool:
        """处理 API Key 认证"""
        from app.infra.database import session_scope
        from app.models.database import Tenant

        # 解密获取租户 ID
        tenant_id = extract_tenant_id_from_api_key(api_key)
        if not tenant_id:
            logger.warning("invalid_api_key", api_key_prefix=api_key[:10])
            return False

        try:
            async with session_scope() as session:
                from sqlalchemy import select

                tenant_stmt = select(Tenant).where(
                    Tenant.id == tenant_id,
                    Tenant.api_key == api_key,
                    Tenant.status == "active",
                )
                tenant_result = await session.execute(tenant_stmt)
                tenant = tenant_result.scalar_one_or_none()

                if tenant:
                    set_tenant_context(
                        TenantContext(
                            tenant_id=tenant.id,
                            tenant_name=tenant.name,
                            api_key=tenant.api_key,
                            config=tenant.config or {},
                        )
                    )
                    request.state.tenant_id = tenant.id
                    request.state.tenant = tenant
                    logger.info("api_key_auth_success", tenant_id=tenant_id)
                    return True
                else:
                    logger.warning("api_key_not_found_or_inactive", tenant_id=tenant_id)
                    return False
        except Exception as e:
            logger.error("api_key_auth_error", error=str(e))
            return False


# ============== FastAPI 依赖 ==============


async def get_tenant_id_dep(
    request: Request,
) -> int | None:
    """获取当前租户 ID（依赖注入）

    Args:
        request: FastAPI Request

    Returns:
        租户 ID，未认证返回 None
    """
    return getattr(request.state, "tenant_id", None)


# 兼容旧代码的别名
get_tenant_id = get_tenant_id_dep


async def require_tenant(
    request: Request,
) -> int:
    """要求租户认证（依赖注入）

    Args:
        request: FastAPI Request

    Returns:
        租户 ID

    Raises:
        HTTPException: 未认证时抛出 401
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="租户认证失败",
        )
    return tenant_id


async def get_tenant_context_dep(
    request: Request,
) -> TenantContext | None:
    """获取租户上下文（依赖注入）

    Args:
        request: FastAPI Request

    Returns:
        TenantContext 对象，未认证返回 None
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        return None

    return get_tenant_context()


# 导出 get_tenant_context 供直接使用
def get_tenant_context() -> TenantContext | None:
    """获取当前租户上下文

    可以在任何地方调用，不依赖 FastAPI 依赖注入。

    Returns:
        TenantContext 对象，未认证返回 None
    """
    from app.auth.tenant import get_tenant_context as _get

    return _get()


# 类型别名
TenantIdDep = Annotated[int | None, Depends(get_tenant_id_dep)]
RequiredTenantIdDep = Annotated[int, Depends(require_tenant)]
TenantContextDep = Annotated[TenantContext | None, Depends(get_tenant_context_dep)]
