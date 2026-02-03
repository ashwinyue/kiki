"""认证 API

提供用户注册、登录、会话管理、Token 验证等接口。
使用 Service 层处理业务逻辑，API 层仅负责请求/响应处理。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt import get_token_sub
from app.infra.database import session_scope
from app.models.database import User
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    SessionListItem,
    SessionResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
    TokenResponse,
    TokenValidateRequest,
    TokenValidateResponse,
    UserResponse,
    UserWithTokenResponse,
)
from app.services.core.auth import get_auth_service

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> User:
    """获取当前用户

    验证 JWT Token 并返回用户信息。

    Args:
        credentials: HTTP Bearer 认证凭据

    Returns:
        User: 用户实例

    Raises:
        HTTPException: 认证失败时返回 401
    """
    token = credentials.credentials
    user_id = get_token_sub(token)
    if user_id is None:
        logger.warning("invalid_token", token_prefix=token[:10] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async with session_scope() as session:
        from app.infra.database import user_repository

        repo = user_repository(session)
        # Token 中的 sub 是用户 ID（整数）
        user = await repo.get(int(user_id))
        if user is None:
            logger.error("user_not_found", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # 绑定用户上下文
        from app.observability.logging import bind_context

        bind_context(user_id=user.id)

        return user


async def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> int:
    """获取当前用户 ID

    轻量级认证依赖，仅返回用户 ID 而非完整用户对象。

    Args:
        credentials: HTTP Bearer 认证凭据

    Returns:
        int: 用户 ID

    Raises:
        HTTPException: 认证失败时返回 401
    """
    token = credentials.credentials
    user_id = get_token_sub(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    # 尝试解析为整数
    try:
        return int(user_id)
    except ValueError as err:
        # 如果是 email，查找用户 ID
        async with session_scope() as session:
            from app.infra.database import user_repository

            repo = user_repository(session)
            user = await repo.get_by_email(user_id)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found",
                ) from err
            return user.id


# ============== 注册/登录 ==============


@router.post(
    "/register",
    response_model=UserWithTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册",
    description="创建新用户账号并返回访问令牌。密码需包含至少一个大写字母和一个数字。",
    responses={
        status.HTTP_201_CREATED: {"description": "注册成功"},
        status.HTTP_400_BAD_REQUEST: {"description": "请求参数验证失败"},
        status.HTTP_409_CONFLICT: {"description": "邮箱已被注册"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "请求过于频繁"},
    },
)
@limiter.limit(RateLimit.REGISTER)
async def register(
    request: Request,
    data: RegisterRequest,
) -> UserWithTokenResponse:
    """用户注册

    创建新用户账号，密码需满足复杂度要求：
    - 至少 8 个字符
    - 至少包含一个大写字母
    - 至少包含一个数字
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        return await auth_service.register_user(data)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录（表单）",
    description="使用表单数据登录，返回访问令牌。",
    responses={
        status.HTTP_200_OK: {"description": "登录成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "邮箱或密码错误"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "请求过于频繁"},
    },
)
@limiter.limit(RateLimit.LOGIN)
async def login(
    request: Request,
    username: Annotated[str, Form(description="邮箱")],
    password: Annotated[str, Form(description="密码")],
) -> TokenResponse:
    """用户登录（表单格式）

    适用于传统表单提交的登录方式。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        access_token, expires_at = await auth_service.login_user(username, password)

        return TokenResponse(
            access_token=access_token,
            # ruff: disable=S106 - "bearer" 是 OAuth 2.0 标准 token type，非密码
            token_type="bearer",  # noqa: S106 - OAuth 2.0 标准 token type
            expires_at=expires_at.isoformat() if expires_at else None,
        )


@router.post(
    "/login/json",
    response_model=TokenResponse,
    summary="用户登录（JSON）",
    description="使用 JSON 数据登录，返回访问令牌。",
    responses={
        status.HTTP_200_OK: {"description": "登录成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "邮箱或密码错误"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "请求过于频繁"},
    },
)
@limiter.limit(RateLimit.LOGIN)
async def login_json(
    request: Request,
    data: LoginRequest,
) -> TokenResponse:
    """用户登录（JSON 格式）

    适用于前后端分离的 API 调用。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        access_token, expires_at = await auth_service.login_user(
            data.username,
            data.password,
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",  # noqa: S106 - OAuth 2.0 标准 token type
            expires_at=expires_at.isoformat() if expires_at else None,
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户",
    description="获取当前登录用户的详细信息。",
    responses={
        status.HTTP_200_OK: {"description": "成功返回用户信息"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def get_me(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """获取当前用户信息

    返回当前登录用户的详细信息。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        return await auth_service.get_user_response(current_user)


# ============== 会话管理 ==============


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建会话",
    description="创建新的聊天会话并返回会话令牌。",
    responses={
        status.HTTP_201_CREATED: {"description": "会话创建成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def create_session(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    name: Annotated[str, Form(description="会话名称")] = "",
) -> SessionResponse:
    """创建新会话

    创建一个独立的聊天会话，每个会话有自己的消息历史。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        return await auth_service.create_session(current_user.id, name)


@router.get(
    "/sessions",
    response_model=list[SessionListItem],
    summary="列出会话",
    description="获取当前用户的所有会话列表。",
    responses={
        status.HTTP_200_OK: {"description": "成功返回会话列表"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def list_sessions(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[SessionListItem]:
    """列出用户的所有会话

    返回当前用户的所有聊天会话，包含每个会话的消息数量。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        return await auth_service.list_sessions(current_user.id)


@router.delete(
    "/sessions/{session_id}",
    response_model=dict[str, str],
    summary="删除会话",
    description="删除指定的会话及其关联的所有消息。",
    responses={
        status.HTTP_200_OK: {"description": "会话删除成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权删除其他用户的会话"},
        status.HTTP_404_NOT_FOUND: {"description": "会话不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def delete_session(
    request: Request,
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    """删除会话

    删除指定会话及其关联的所有消息。只能删除自己的会话。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        await auth_service.delete_session(session_id, current_user.id)

        return {"status": "success", "message": "Session deleted"}


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="更新会话名称",
    description="修改指定会话的名称。",
    responses={
        status.HTTP_200_OK: {"description": "会话名称更新成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权修改其他用户的会话"},
        status.HTTP_404_NOT_FOUND: {"description": "会话不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def update_session_name(
    request: Request,
    session_id: str,
    name: Annotated[str, Form(description="新名称")],
    current_user: Annotated[User, Depends(get_current_user)],
) -> SessionResponse:
    """更新会话名称

    修改指定会话的名称。只能修改自己的会话。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        return await auth_service.update_session_name(session_id, current_user.id, name)


# ============== Token 管理 ==============


@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    summary="刷新访问令牌",
    description="使用刷新令牌获取新的访问令牌。",
    responses={
        status.HTTP_200_OK: {"description": "令牌刷新成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "刷新令牌无效"},
    },
)
@limiter.limit(RateLimit.API)
async def refresh_token(
    request: Request,
    data: TokenRefreshRequest,
) -> TokenRefreshResponse:
    """刷新访问令牌

    使用刷新令牌获取新的访问令牌，实现无感续期。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        access_token, expires_at = await auth_service.refresh_token(data.refresh_token)

        return TokenRefreshResponse(
            access_token=access_token,
            token_type="bearer",  # noqa: S106 - OAuth 2.0 标准 token type
            expires_at=expires_at.isoformat() if expires_at else None,
        )


@router.post(
    "/validate",
    response_model=TokenValidateResponse,
    summary="验证 Token",
    description="验证 Token 是否有效并返回相关信息。",
    responses={
        status.HTTP_200_OK: {"description": "验证完成"},
    },
)
@limiter.limit(RateLimit.API)
async def validate_token(
    request: Request,
    data: TokenValidateRequest,
) -> TokenValidateResponse:
    """验证 Token

    检查 Token 是否有效，返回验证结果和用户信息。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        result = await auth_service.validate_token(data.token)

        return TokenValidateResponse(
            valid=result["valid"],
            user_id=result["user_id"],
            expires_at=result["expires_at"],
            message=result["message"],
        )


@router.post(
    "/change-password",
    response_model=dict[str, str],
    summary="修改密码",
    description="修改当前用户的密码，需要提供旧密码进行验证。",
    responses={
        status.HTTP_200_OK: {"description": "密码修改成功"},
        status.HTTP_400_BAD_REQUEST: {"description": "旧密码错误或新密码不符合要求"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def change_password(
    request: Request,
    data: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    """修改密码

    修改当前登录用户的密码，需要验证旧密码。
    """
    async with session_scope() as session:
        auth_service = get_auth_service(session)
        await auth_service.change_password(current_user.id, data)

        logger.info("password_changed", user_id=current_user.id)

        return {"status": "success", "message": "Password changed successfully"}
