"""用户仓储

提供用户相关的数据访问操作。
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, UserCreate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


class UserRepository(BaseRepository[User]):
    """用户仓储类

    提供用户的 CRUD 操作和特定查询方法。
    """

    def __init__(self, session: AsyncSession):
        """初始化用户仓储

        Args:
            session: 异步数据库会话
        """
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        """根据邮箱获取用户

        Args:
            email: 用户邮箱

        Returns:
            用户实例或 None
        """
        try:
            statement = select(User).where(User.email == email)
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("user_repository_get_by_email_failed", email=email, error=str(e))
            return None

    async def create_with_password(
        self,
        data: UserCreate,
    ) -> User:
        """创建用户并加密密码

        Args:
            data: 用户创建数据

        Returns:
            创建的用户实例
        """
        try:
            user = User(
                email=data.email,
                full_name=data.full_name,
                is_active=data.is_active,
                is_superuser=data.is_superuser,
                tenant_id=data.tenant_id,
                can_access_all_tenants=data.can_access_all_tenants,
            )
            user.set_password(data.password)

            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)

            logger.info("user_created", user_id=user.id, email=user.email)
            return user

        except Exception as e:
            await self.session.rollback()
            logger.error("user_repository_create_failed", email=data.email, error=str(e))
            raise

    async def update_password(
        self,
        user_id: int,
        new_password: str,
    ) -> bool:
        """更新用户密码

        Args:
            user_id: 用户 ID
            new_password: 新密码

        Returns:
            是否更新成功
        """
        try:
            user = await self.get(user_id)
            if user is None:
                return False

            user.set_password(new_password)
            await self.session.commit()

            logger.info("user_password_updated", user_id=user_id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error("user_repository_update_password_failed", user_id=user_id, error=str(e))
            raise

    async def list_active(
        self,
        params: PaginationParams,
    ) -> PaginatedResult[User]:
        """分页获取活跃用户

        Args:
            params: 分页参数

        Returns:
            分页结果
        """
        return await self.list_paginated(params, is_active=True)

    async def verify_password(
        self,
        email: str,
        password: str,
    ) -> User | None:
        """验证用户密码

        Args:
            email: 用户邮箱
            password: 密码

        Returns:
            验证成功返回用户实例，否则返回 None
        """
        user = await self.get_by_email(email)
        if user and user.verify_password(password):
            return user
        return None

    async def email_exists(
        self,
        email: str,
        exclude_id: int | None = None,
    ) -> bool:
        """检查邮箱是否存在

        Args:
            email: 邮箱
            exclude_id: 排除的用户 ID（用于更新时检查）

        Returns:
            邮箱是否存在
        """
        try:
            statement = select(User).where(User.email == email)
            if exclude_id:
                statement = statement.where(User.id != exclude_id)
            result = await self.session.execute(statement)
            return result.scalar_one_or_none() is not None
        except Exception as e:
            logger.error("user_repository_email_exists_failed", email=email, error=str(e))
            return False
