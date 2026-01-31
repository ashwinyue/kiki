"""会话仓储

提供会话相关的数据访问操作。
"""

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database import ChatSession, SessionCreate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


class SessionRepository(BaseRepository[ChatSession]):
    """会话仓储类"""

    def __init__(self, session: AsyncSession):
        """初始化会话仓储

        Args:
            session: 异步数据库会话
        """
        super().__init__(ChatSession, session)

    async def create_with_user(
        self,
        data: SessionCreate,
        user_id: int | None = None,
    ) -> ChatSession:
        """创建会话

        Args:
            data: 会话创建数据
            user_id: 用户 ID

        Returns:
            创建的会话实例
        """
        import uuid

        session_obj = ChatSession(
            id=str(uuid.uuid4()),
            name=data.name,
            user_id=user_id or data.user_id,
            tenant_id=data.tenant_id,
            agent_id=data.agent_id,
            agent_config=data.agent_config,
            context_config=data.context_config,
        )
        return await self.create(session_obj)

    async def get_with_messages(
        self,
        session_id: str,
    ) -> ChatSession | None:
        """获取会话及其消息

        Args:
            session_id: 会话 ID

        Returns:
            会话实例（包含消息）或 None
        """
        try:
            statement = (
                select(ChatSession)
                .where(ChatSession.id == session_id)
                .options(selectinload(ChatSession.messages))
            )
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "session_repository_get_with_messages_failed", session_id=session_id, error=str(e)
            )
            return None

    async def list_by_user(
        self,
        user_id: int,
        params: PaginationParams,
    ) -> PaginatedResult[ChatSession]:
        """分页获取用户的会话列表

        Args:
            user_id: 用户 ID
            params: 分页参数

        Returns:
            分页结果
        """
        return await self.list_paginated(params, user_id=user_id)

    async def get_message_count(
        self,
        session_id: str,
    ) -> int:
        """获取会话的消息数量

        Args:
            session_id: 会话 ID

        Returns:
            消息数量
        """
        from app.models.database import Message

        try:
            statement = (
                select(func.count()).select_from(Message).where(Message.session_id == session_id)
            )
            result = await self.session.execute(statement)
            return result.scalar() or 0
        except Exception as e:
            logger.error(
                "session_repository_count_messages_failed", session_id=session_id, error=str(e)
            )
            return 0

    async def list_recent(
        self,
        user_id: int | None = None,
        limit: int = 10,
    ) -> list[ChatSession]:
        """获取最近的会话列表

        Args:
            user_id: 用户 ID（可选）
            limit: 限制数量

        Returns:
            会话列表
        """
        try:
            statement = select(ChatSession)
            if user_id:
                statement = statement.where(ChatSession.user_id == user_id)
            statement = statement.order_by(desc(ChatSession.updated_at)).limit(limit)
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("session_repository_list_recent_failed", user_id=user_id, error=str(e))
            return []
