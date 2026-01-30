"""认证 API

提供用户注册、登录、会话管理、Token 验证等接口。
"""

import uuid
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from app.core.auth import create_access_token, verify_token, get_token_sub
from app.core.config import get_settings
from app.core.limiter import limiter, RateLimit
from app.core.logging import get_logger, bind_context
from app.models.database import User, Session, SessionCreate
from app.services.database import (
    session_scope,
    user_repository,
    session_repository,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


# ============== Schemas ==============

class RegisterRequest(BaseModel):
    """注册请求"""
    email: EmailStr = Field(..., description="邮箱")
    password: str = Field(..., min_length=8, max_length=100, description="密码")
    full_name: str | None = Field(None, max_length=255, description="全名")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="邮箱")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str
    token_type: str = "bearer"
    expires_at: str | None = None


class UserResponse(BaseModel):
    """用户响应"""
    id: int
    email: str
    full_name: str | None
    is_active: bool
    is_superuser: bool


class UserWithTokenResponse(UserResponse):
    """用户响应（含 Token）"""
    access_token: str
    token_type: str = "bearer"


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str
    name: str
    token: str
    created_at: str


class SessionListItem(BaseModel):
    """会话列表项"""
    session_id: str
    name: str
    created_at: str
    message_count: int = 0


# ============== 认证依赖 ==============

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """获取当前用户

    Args:
        credentials: HTTP Bearer 认证凭据

    Returns:
        User: 用户实例

    Raises:
        HTTPException: 认证失败
    """
    settings = get_settings()
    token = credentials.credentials

    user_id = get_token_sub(token)
    if user_id is None:
        logger.warning("invalid_token", token_prefix=token[:10] + "...")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async with session_scope() as session:
        repo = user_repository(session)
        user = await repo.get_by_email(user_id)
        if user is None:
            logger.error("user_not_found", email=user_id)
            raise HTTPException(status_code=404, detail="User not found")

        # 绑定用户上下文
        bind_context(user_id=user.id)

        return user


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
    """获取当前用户 ID

    Args:
        credentials: HTTP Bearer 认证凭据

    Returns:
        int: 用户 ID

    Raises:
        HTTPException: 认证失败
    """
    settings = get_settings()
    token = credentials.credentials

    user_id = get_token_sub(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 尝试解析为整数
    try:
        return int(user_id)
    except ValueError:
        # 如果是 email，查找用户 ID
        async with session_scope() as session:
            repo = user_repository(session)
            user = await repo.get_by_email(user_id)
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")
            return user.id


# ============== 注册/登录 ==============

@router.post("/register", response_model=UserWithTokenResponse)
@limiter.limit(RateLimit.REGISTER)
async def register(
    request: Request,
    data: RegisterRequest,
) -> UserWithTokenResponse:
    """用户注册

    Args:
        request: FastAPI 请求
        data: 注册数据

    Returns:
        UserWithTokenResponse: 用户信息及 Token
    """
    async with session_scope() as session:
        repo = user_repository(session)

        # 检查邮箱是否已存在
        if await repo.email_exists(data.email):
            raise HTTPException(status_code=400, detail="Email already registered")

        # 创建用户
        from app.models.database import UserCreate

        user_create = UserCreate(
            email=data.email,
            password=data.password,
            full_name=data.full_name,
        )
        user = await repo.create_with_password(user_create)

        # 生成 Token
        token_data = create_access_token(data={"sub": str(user.id)})
        access_token = token_data.access_token

        logger.info("user_registered", user_id=user.id, email=user.email)

        return UserWithTokenResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            access_token=access_token,
        )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(RateLimit.LOGIN)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> TokenResponse:
    """用户登录

    Args:
        request: FastAPI 请求
        username: 邮箱
        password: 密码

    Returns:
        TokenResponse: 访问令牌
    """
    async with session_scope() as session:
        repo = user_repository(session)

        # 验证密码
        user = await repo.verify_password(username, password)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 生成 Token
        token_data = create_access_token(data={"sub": str(user.id)})

        logger.info("user_logged_in", user_id=user.id, email=user.email)

        return TokenResponse(
            access_token=token_data.access_token,
            token_type="bearer",
            expires_at=token_data.expires_at.isoformat() if token_data.expires_at else None,
        )


@router.post("/login/json", response_model=TokenResponse)
@limiter.limit(RateLimit.LOGIN)
async def login_json(
    request: Request,
    data: LoginRequest,
) -> TokenResponse:
    """用户登录（JSON 格式）

    Args:
        request: FastAPI 请求
        data: 登录数据

    Returns:
        TokenResponse: 访问令牌
    """
    async with session_scope() as session:
        repo = user_repository(session)

        user = await repo.verify_password(data.username, data.password)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = create_access_token(data={"sub": str(user.id)})

        logger.info("user_logged_in", user_id=user.id)

        return TokenResponse(
            access_token=token_data.access_token,
            token_type="bearer",
            expires_at=token_data.expires_at.isoformat() if token_data.expires_at else None,
        )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """获取当前用户信息

    Args:
        current_user: 当前用户

    Returns:
        UserResponse: 用户信息
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
    )


# ============== 会话管理 ==============

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: Request,
    current_user: User = Depends(get_current_user),
    name: str = Form(""),
) -> SessionResponse:
    """创建新会话

    Args:
        request: FastAPI 请求
        current_user: 当前用户
        name: 会话名称

    Returns:
        SessionResponse: 会话信息及 Token
    """
    async with session_scope() as session:
        repo = session_repository(session)

        # 创建会话
        session_obj = await repo.create_with_user(
            data=SessionCreate(name=name),
            user_id=current_user.id,
        )

        # 生成 Token（session_id 作为 subject）
        token_data = create_access_token(data={"sub": session_obj.id})

        logger.info(
            "session_created",
            session_id=session_obj.id,
            user_id=current_user.id,
            name=name,
        )

        return SessionResponse(
            session_id=session_obj.id,
            name=session_obj.name,
            token=token_data.access_token,
            created_at=session_obj.created_at.isoformat(),
        )


@router.get("/sessions", response_model=List[SessionListItem])
async def list_sessions(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> List[SessionListItem]:
    """列出用户的所有会话

    Args:
        request: FastAPI 请求
        current_user: 当前用户

    Returns:
        List[SessionListItem]: 会话列表
    """
    async with session_scope() as session:
        repo = session_repository(session)

        from app.repositories.base import PaginationParams

        params = PaginationParams(page=1, size=100)
        result = await repo.list_by_user(current_user.id, params)

        items = []
        for session_obj in result.items:
            # 获取消息数量
            message_count = await repo.get_message_count(session_obj.id)

            items.append(
                SessionListItem(
                    session_id=session_obj.id,
                    name=session_obj.name,
                    created_at=session_obj.created_at.isoformat(),
                    message_count=message_count,
                )
            )

        return items


@router.delete("/sessions/{session_id}")
async def delete_session(
    request: Request,
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """删除会话

    Args:
        request: FastAPI 请求
        session_id: 会话 ID
        current_user: 当前用户

    Returns:
        操作结果
    """
    async with session_scope() as session:
        repo = session_repository(session)

        # 验证会话归属
        session_obj = await repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if session_obj.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Cannot delete other users' sessions")

        # 删除会话关联的消息
        from app.repositories.message import MessageRepository

        message_repo = MessageRepository(session)
        await message_repo.delete_by_session(session_id)

        # 删除会话
        await repo.delete(session_id)

        logger.info("session_deleted", session_id=session_id, user_id=current_user.id)

        return {"status": "success", "message": "Session deleted"}


@router.patch("/sessions/{session_id}")
async def update_session_name(
    request: Request,
    session_id: str,
    name: str = Form(...),
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    """更新会话名称

    Args:
        request: FastAPI 请求
        session_id: 会话 ID
        name: 新名称
        current_user: 当前用户

    Returns:
        SessionResponse: 更新后的会话信息
    """
    async with session_scope() as session:
        repo = session_repository(session)

        session_obj = await repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if session_obj.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Cannot modify other users' sessions")

        # 更新名称
        session_obj.name = name
        await session.commit()
        await session.refresh(session_obj)

        logger.info("session_name_updated", session_id=session_id, name=name)

        return SessionResponse(
            session_id=session_obj.id,
            name=session_obj.name,
            token="",  # 不返回新 token
            created_at=session_obj.created_at.isoformat(),
        )
