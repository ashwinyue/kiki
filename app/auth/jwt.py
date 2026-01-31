"""认证工具模块

提供 JWT Token 认证功能。
"""

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()


class Token(BaseModel):
    """Token 响应模型"""

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime | None = None


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> Token:
    """创建访问 Token

    Args:
        data: 要编码的数据（通常包含 sub、user_id 等）
        expires_delta: 过期时间增量

    Returns:
        Token 对象
    """
    to_encode = data.copy()

    # 设置过期时间
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": f"{to_encode.get('sub', 'unknown')}-{datetime.now(UTC).timestamp()}",
        }
    )

    # 编码 JWT
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )

    logger.info("token_created", sub=to_encode.get("sub"), expires_at=expire.isoformat())

    return Token(access_token=encoded_jwt, expires_at=expire)


def verify_token(token: str) -> dict | None:
    """验证 Token

    Args:
        token: JWT Token 字符串

    Returns:
        解码后的 Payload，验证失败返回 None

    Raises:
        ValueError: Token 格式无效
    """
    if not token or not isinstance(token, str):
        logger.warning("token_invalid_format")
        raise ValueError("Token must be a non-empty string")

    # 基本格式验证
    # JWT 格式: header.payload.signature
    if not re.match(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$", token):
        logger.warning("token_suspicious_format")
        raise ValueError("Token format is invalid - expected JWT format")

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        logger.info("token_verified", sub=payload.get("sub"))
        return payload

    except JWTError as e:
        logger.error("token_verification_failed", error=str(e))
        return None


def decode_token(token: str) -> dict | None:
    """解码 Token（不验证过期时间）

    Args:
        token: JWT Token 字符串

    Returns:
        解码后的 Payload，失败返回 None
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False},
        )
        return payload
    except JWTError:
        return None


def get_token_sub(token: str) -> str | None:
    """从 Token 获取用户标识

    Args:
        token: JWT Token 字符串

    Returns:
        用户标识（sub），验证失败返回 None
    """
    try:
        payload = verify_token(token)
    except ValueError:
        return None

    if payload:
        return payload.get("sub")
    return None


# ============== FastAPI 依赖 ==============

from fastapi import Header, HTTPException, status


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    """获取当前用户（依赖注入）

    Args:
        authorization: Authorization 请求头

    Returns:
        用户信息字典

    Raises:
        HTTPException: 认证失败
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization:
        raise credentials_exception

    # 提取 Bearer Token
    parts = authorization.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        raise credentials_exception

    token = parts[1]
    try:
        payload = verify_token(token)
    except ValueError:
        raise credentials_exception

    if payload is None:
        raise credentials_exception

    return payload


async def get_current_user_id(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> str:
    """获取当前用户 ID（依赖注入）

    Args:
        current_user: 当前用户信息

    Returns:
        用户 ID

    Raises:
        HTTPException: 用户 ID 不存在
    """
    user_id = current_user.get("sub") or current_user.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的用户标识",
        )
    return user_id


# 解决循环导入
if TYPE_CHECKING:
    pass  # Depends 已经在顶部导入
