"""API Key 服务

提供 API Key 生成、验证和管理功能。
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.models.api_key import (
    ApiKey,
    ApiKeyType,
    ApiKeyStatus,
    ApiKeyVerifyResponse,
)
from app.observability.logging import get_logger
from app.repositories.api_key import ApiKeyRepository
from app.services.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

settings = get_settings()


# ============== API Key 生成与管理 ==============


class ApiKeyService:
    """API Key 服务"""

    # API Key 前缀格式
    PREFIX_FORMAT = "{type}_{random}"

    # API Key 长度
    KEY_LENGTH = 32

    # 类型前缀映射
    TYPE_PREFIXES = {
        ApiKeyType.PERSONAL: "kiki",
        ApiKeyType.SERVICE: "kiki_srv",
        ApiKeyType.MCP: "kiki_mcp",
        ApiKeyType.WEBHOOK: "kiki_wh",
    }

    @classmethod
    def generate_key(cls, key_type: ApiKeyType = ApiKeyType.PERSONAL) -> tuple[str, str]:
        """生成 API Key

        Args:
            key_type: API Key 类型

        Returns:
            (完整 Key, 前缀) 元组
        """
        import secrets

        # 生成随机字符串
        random_part = secrets.token_urlsafe(cls.KEY_LENGTH)

        # 构建前缀
        prefix = cls.TYPE_PREFIXES.get(key_type, "kiki")

        # 完整 Key
        full_key = f"{prefix}_{random_part}"

        logger.info("api_key_generated", prefix=prefix, key_type=key_type.value)

        return full_key, prefix

    @classmethod
    def hash_key(cls, key: str) -> str:
        """对 API Key 进行哈希

        Args:
            key: 原始 API Key

        Returns:
            哈希后的字符串
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(key.encode(), salt)
        return hashed.decode()

    @classmethod
    def verify_key_hash(cls, key: str, hashed_key: str) -> bool:
        """验证 API Key 哈希

        Args:
            key: 原始 API Key
            hashed_key: 哈希后的 Key

        Returns:
            验证是否通过
        """
        return bcrypt.checkpw(key.encode(), hashed_key.encode())

    @classmethod
    def extract_key_from_header(cls, authorization: str) -> str | None:
        """从 Authorization 头提取 API Key

        支持:
        - Bearer kiki_xxxx
        - kiki_xxxx

        Args:
            authorization: Authorization 请求头

        Returns:
            API Key 字符串，无效返回 None
        """
        if not authorization:
            return None

        parts = authorization.split()

        # Bearer token 格式
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

        # 直接传入 Key
        if len(parts) == 1:
            return parts[0]

        return None


# ============== FastAPI 依赖注入 ==============


class CurrentApiKey(BaseModel):
    """当前 API Key 上下文"""

    id: int
    user_id: int
    key_type: ApiKeyType
    scopes: list[str]
    key_prefix: str


async def verify_api_key(
    authorization: Annotated[str | None, str] = None,
    session: Annotated[AsyncSession, Depends(get_session)] = None,
) -> CurrentApiKey | None:
    """验证 API Key（依赖注入）

    Args:
        authorization: Authorization 请求头
        session: 数据库会话

    Returns:
        CurrentApiKey 对象，验证失败返回 None

    Examples:
        ```python
        @app.get("/api/protected")
        async def protected(
            api_key: Annotated[CurrentApiKey | None, Depends(verify_api_key)] = None,
        ):
            if api_key is None:
                raise HTTPException(status_code=401, detail="Invalid API Key")
            # ...
        ```
    """
    if not authorization:
        return None

    # 提取 API Key
    key = ApiKeyService.extract_key_from_header(authorization)
    if not key:
        return None

    # 查询数据库
    repo = ApiKeyRepository(session)

    # 遍历查找（通过前缀匹配）
    # 提取前缀进行快速查找
    prefix = key.split("_")[0] if "_" in key else key[:8]
    api_keys = await repo.list_by_prefix(prefix)

    for api_key in api_keys:
        if ApiKeyService.verify_key_hash(key, api_key.hashed_key):
            # 检查状态
            if api_key.status != ApiKeyStatus.ACTIVE:
                logger.warning(
                    "api_key_inactive",
                    api_key_id=api_key.id,
                    status=api_key.status.value,
                )
                return None

            # 检查过期时间
            if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
                logger.warning("api_key_expired", api_key_id=api_key.id)
                # 更新状态为过期
                await repo.update_status(api_key.id, ApiKeyStatus.EXPIRED)
                return None

            # 更新最后使用时间
            await repo.update_last_used(api_key.id)

            logger.info(
                "api_key_verified",
                api_key_id=api_key.id,
                user_id=api_key.user_id,
                key_type=api_key.key_type.value,
            )

            return CurrentApiKey(
                id=api_key.id,
                user_id=api_key.user_id,
                key_type=api_key.key_type,
                scopes=api_key.scopes or [],
                key_prefix=api_key.key_prefix,
            )

    logger.warning("api_key_not_found", prefix=prefix)
    return None


async def require_api_key(
    api_key: Annotated[CurrentApiKey | None, Depends(verify_api_key)] = None,
) -> CurrentApiKey:
    """要求有效的 API Key（依赖注入）

    与 verify_api_key 不同，此函数在验证失败时抛出异常。

    Args:
        api_key: API Key 上下文

    Returns:
        CurrentApiKey 对象

    Raises:
        HTTPException: 认证失败时抛出 401

    Examples:
        ```python
        @app.get("/api/protected")
        async def protected(
            api_key: Annotated[CurrentApiKey, Depends(require_api_key)],
        ):
            # api_key 保证有效
            ...
        ```
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={
                "WWW-Authenticate": "Bearer",
                "X-API-Key-Required": "true",
            },
        )

    return api_key


async def verify_api_key_or_token(
    api_key: Annotated[CurrentApiKey | None, Depends(verify_api_key)] = None,
    user_token: Annotated[dict | None, Depends(lambda: None)] = None,
) -> CurrentApiKey | dict:
    """验证 API Key 或 JWT Token（任一即可）

    Args:
        api_key: API Key 上下文
        user_token: JWT Token 上下文

    Returns:
        CurrentApiKey 或用户 Token Payload

    Raises:
        HTTPException: 两者都无效时抛出 401

    Examples:
        ```python
        from app.core.auth import get_current_user

        @app.get("/api/protected")
        async def protected(
            auth: Annotated[CurrentApiKey | dict, Depends(verify_api_key_or_token)],
        ):
            # auth 可能是 CurrentApiKey 或 dict (JWT payload)
            ...
        ```
    """
    from app.core.auth import get_current_user

    # 如果 API Key 有效，直接返回
    if api_key is not None:
        return api_key

    # 尝试验证 JWT Token
    try:
        # 这里需要获取 Authorization 头并验证
        # 由于 FastAPI 依赖的限制，我们简化处理
        # 实际使用时可能需要调整
        return await get_current_user()
    except HTTPException:
        pass

    # 两者都无效
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (API Key or Bearer Token)",
        headers={
            "WWW-Authenticate": "Bearer",
            "X-API-Key-Required": "true",
        },
    )


def require_scope(*required_scopes: str):
    """要求特定权限范围的依赖工厂

    Args:
        *required_scopes: 需要的权限范围

    Returns:
        依赖函数

    Examples:
        ```python
        @app.get("/api/admin")
        async def admin_only(
            auth: Annotated[CurrentApiKey, Depends(require_scope("admin", "write"))],
        ):
            # auth 具有 admin 和 write 权限
            ...
        ```
    """

    async def check_scope(
        api_key: Annotated[CurrentApiKey, Depends(require_api_key)],
    ) -> CurrentApiKey:
        """检查权限范围"""
        key_scopes = set(api_key.scopes or [])
        required = set(required_scopes)

        if not required.issubset(key_scopes):
            logger.warning(
                "api_key_insufficient_scope",
                api_key_id=api_key.id,
                required=list(required),
                has=list(key_scopes),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scope. Required: {', '.join(required_scopes)}",
            )

        return api_key

    return check_scope


# ============== 辅助函数 ==============


def is_mcp_key(api_key: CurrentApiKey) -> bool:
    """检查是否为 MCP 专用 Key"""
    return api_key.key_type == ApiKeyType.MCP


def is_service_key(api_key: CurrentApiKey) -> bool:
    """检查是否为服务间调用 Key"""
    return api_key.key_type == ApiKeyType.SERVICE


def has_scope(api_key: CurrentApiKey, scope: str) -> bool:
    """检查是否具有特定权限"""
    return scope in (api_key.scopes or [])
