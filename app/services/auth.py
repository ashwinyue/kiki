"""认证服务层

提供用户注册、登录、会话管理等业务逻辑。
将业务逻辑从 API 路由中分离出来，便于测试和复用。
"""

from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import create_access_token
from app.models.database import SessionCreate, User
from app.observability.logging import get_logger
from app.repositories.message import MessageRepository
from app.repositories.session import SessionRepository
from app.repositories.user import UserRepository
from app.schemas.auth import (
    RegisterRequest,
    SessionListItem,
    SessionResponse,
    UserResponse,
    UserWithTokenResponse,
)

logger = get_logger(__name__)


class AuthService:
    """认证服务

    提供用户认证相关的业务逻辑，包括注册、登录、会话管理等。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化认证服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self._user_repo: UserRepository | None = None
        self._session_repo: SessionRepository | None = None

    @property
    def user_repo(self) -> UserRepository:
        """获取用户仓储（延迟初始化）"""
        if self._user_repo is None:
            self._user_repo = UserRepository(self.session)
        return self._user_repo

    @property
    def session_repo(self) -> SessionRepository:
        """获取会话仓储（延迟初始化）"""
        if self._session_repo is None:
            self._session_repo = SessionRepository(self.session)
        return self._session_repo

    # ============== 用户注册 ==============

    async def register_user(self, data: RegisterRequest) -> UserWithTokenResponse:
        """注册新用户

        Args:
            data: 注册请求数据

        Returns:
            用户信息及访问令牌

        Raises:
            HTTPException: 邮箱已被注册时抛出 409 错误
        """
        # 检查邮箱是否已存在
        if await self.user_repo.email_exists(data.email):
            logger.warning("registration_email_exists", email=data.email)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

        # 创建用户
        from app.models.database import UserCreate

        user_create = UserCreate(
            email=data.email,
            password=data.password,
            full_name=data.full_name,
        )
        user = await self.user_repo.create_with_password(user_create)

        # 生成访问令牌
        token_data = create_access_token(data={"sub": str(user.id)})

        logger.info("user_registered", user_id=user.id, email=user.email)

        return UserWithTokenResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at,
            access_token=token_data.access_token,
            token_type="bearer",
        )

    # ============== 用户登录 ==============

    async def login_user(self, username: str, password: str) -> tuple[str, datetime | None]:
        """用户登录

        Args:
            username: 邮箱
            password: 密码

        Returns:
            (访问令牌, 过期时间)

        Raises:
            HTTPException: 认证失败时抛出 401 错误
        """
        # 验证密码
        user = await self.user_repo.verify_password(username, password)
        if user is None:
            logger.warning("login_failed", email=username)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 生成访问令牌
        token_data = create_access_token(data={"sub": str(user.id)})

        logger.info("user_logged_in", user_id=user.id, email=user.email)

        return token_data.access_token, token_data.expires_at

    # ============== 用户信息 ==============

    async def get_user_response(self, user: User) -> UserResponse:
        """将用户模型转换为响应对象

        Args:
            user: 用户模型实例

        Returns:
            用户响应对象
        """
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at,
        )

    # ============== 会话管理 ==============

    async def create_session(
        self,
        user_id: int,
        name: str,
    ) -> SessionResponse:
        """创建新会话

        Args:
            user_id: 用户 ID
            name: 会话名称

        Returns:
            会话响应对象
        """
        # 创建会话
        session_obj = await self.session_repo.create_with_user(
            data=SessionCreate(name=name),
            user_id=user_id,
        )

        # 生成会话令牌
        token_data = create_access_token(data={"sub": session_obj.id})

        logger.info(
            "session_created",
            session_id=session_obj.id,
            user_id=user_id,
            name=name,
        )

        return SessionResponse(
            session_id=session_obj.id,
            name=session_obj.name,
            token=token_data.access_token,
            created_at=session_obj.created_at.isoformat(),
        )

    async def list_sessions(
        self,
        user_id: int,
        limit: int = 100,
    ) -> list[SessionListItem]:
        """列出用户的所有会话

        Args:
            user_id: 用户 ID
            limit: 最大返回数量

        Returns:
            会话列表
        """
        from app.repositories.base import PaginationParams

        params = PaginationParams(page=1, size=limit)
        result = await self.session_repo.list_by_user(user_id, params)

        items = []
        for session_obj in result.items:
            # 获取消息数量
            message_count = await self.session_repo.get_message_count(session_obj.id)

            items.append(
                SessionListItem(
                    session_id=session_obj.id,
                    name=session_obj.name,
                    created_at=session_obj.created_at.isoformat(),
                    message_count=message_count,
                )
            )

        return items

    async def delete_session(
        self,
        session_id: str,
        user_id: int,
    ) -> None:
        """删除会话

        Args:
            session_id: 会话 ID
            user_id: 当前用户 ID

        Raises:
            HTTPException: 会话不存在或无权限时抛出
        """
        # 获取会话
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        # 验证权限
        if session_obj.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete other users' sessions",
            )

        # 删除会话关联的消息
        message_repo = MessageRepository(self.session)
        await message_repo.delete_by_session(session_id)

        # 删除会话
        await self.session_repo.delete(session_id)

        logger.info("session_deleted", session_id=session_id, user_id=user_id)

    async def update_session_name(
        self,
        session_id: str,
        user_id: int,
        name: str,
    ) -> SessionResponse:
        """更新会话名称

        Args:
            session_id: 会话 ID
            user_id: 当前用户 ID
            name: 新名称

        Returns:
            更新后的会话响应

        Raises:
            HTTPException: 会话不存在或无权限时抛出
        """
        # 获取会话
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        # 验证权限
        if session_obj.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify other users' sessions",
            )

        # 更新名称
        session_obj.name = name
        await self.session.commit()
        await self.session.refresh(session_obj)

        logger.info("session_name_updated", session_id=session_id, name=name)

        return SessionResponse(
            session_id=session_obj.id,
            name=session_obj.name,
            token="",  # 不返回新 token
            created_at=session_obj.created_at.isoformat(),
        )


# ============== 依赖注入工厂 ==============


def get_auth_service(session: AsyncSession) -> AuthService:
    """创建认证服务实例

    Args:
        session: 数据库会话

    Returns:
        AuthService 实例
    """
    return AuthService(session)
