"""消息仓储

提供消息相关的数据访问操作。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Message, MessageCreate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


class MessageRepository(BaseRepository[Message]):
    """消息仓储类"""

    def __init__(self, session: AsyncSession):
        """初始化消息仓储

        Args:
            session: 异步数据库会话
        """
        super().__init__(Message, session)

    async def create_for_session(
        self,
        data: MessageCreate,
        session_id: str,
    ) -> Message:
        """为会话创建消息

        Args:
            data: 消息创建数据
            session_id: 会话 ID

        Returns:
            创建的消息实例
        """
        message = Message(
            role=data.role,
            content=data.content,
            session_id=session_id or data.session_id,
            request_id=data.request_id,
            knowledge_references=data.knowledge_references,
            agent_steps=data.agent_steps,
            is_completed=data.is_completed if data.is_completed is not None else True,
            tool_calls=data.tool_calls,
            extra_data=data.extra_data,
        )
        return await self.create(message)

    async def list_by_session(
        self,
        session_id: str,
        params: PaginationParams,
    ) -> PaginatedResult[Message]:
        """分页获取会话的消息列表

        Args:
            session_id: 会话 ID
            params: 分页参数

        Returns:
            分页结果
        """
        return await self.list_paginated(params, session_id=session_id)

    async def list_by_session_asc(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Message]:
        """按时间升序获取会话的消息列表

        Args:
            session_id: 会话 ID
            limit: 限制数量

        Returns:
            消息列表
        """
        try:
            statement = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(
                "message_repository_list_by_session_asc_failed", session_id=session_id, error=str(e)
            )
            return []

    async def get_last_by_session(
        self,
        session_id: str,
    ) -> Message | None:
        """获取会话的最后一条消息

        Args:
            session_id: 会话 ID

        Returns:
            消息实例或 None
        """
        try:
            statement = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(desc(Message.created_at))
                .limit(1)
            )
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("message_repository_get_last_failed", session_id=session_id, error=str(e))
            return None

    async def list_by_role(
        self,
        session_id: str,
        role: str,
        limit: int = 100,
    ) -> list[Message]:
        """获取会话中指定角色的消息

        Args:
            session_id: 会话 ID
            role: 角色 (user/assistant/system/tool)
            limit: 限制数量

        Returns:
            消息列表
        """
        try:
            statement = (
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.role == role,
                )
                .order_by(Message.created_at.asc())
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(
                "message_repository_list_by_role_failed",
                session_id=session_id,
                role=role,
                error=str(e),
            )
            return []

    async def delete_by_session(
        self,
        session_id: str,
    ) -> int:
        """删除会话的所有消息

        Args:
            session_id: 会话 ID

        Returns:
            删除的消息数量
        """
        try:
            statement = select(Message).where(Message.session_id == session_id)
            result = await self.session.execute(statement)
            messages = result.scalars().all()

            count = 0
            for message in messages:
                await self.session.delete(message)
                count += 1

            await self.session.commit()
            logger.info("message_repository_delete_by_session", session_id=session_id, count=count)
            return count

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "message_repository_delete_by_session_failed", session_id=session_id, error=str(e)
            )
            return 0

    async def count_by_session(
        self,
        session_id: str,
    ) -> int:
        """统计会话的消息数量

        Args:
            session_id: 会话 ID

        Returns:
            消息数量
        """
        return await self.count(session_id=session_id)

    async def load_messages_before(
        self,
        session_id: str,
        before_message_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        """加载指定消息之前的消息（分页加载）

        对齐 WeKnora 的 GetMessagesBySessionBeforeTime 方法。

        Args:
            session_id: 会话 ID
            before_message_id: 锚点消息 ID，加载此消息之前的消息
            limit: 限制数量

        Returns:
            消息列表（按创建时间升序排列）
        """
        try:
            # 获取锚点消息的创建时间
            before_time: datetime | None = None
            if before_message_id:
                anchor_message = await self.get(before_message_id)
                if anchor_message and anchor_message.session_id == session_id:
                    before_time = anchor_message.created_at

            # 构建查询
            statement = select(Message).where(Message.session_id == session_id)

            # 添加时间过滤条件
            if before_time is not None:
                statement = statement.where(Message.created_at < before_time)

            # 按创建时间降序查询，获取最近的消息
            statement = statement.order_by(desc(Message.created_at)).limit(limit)

            result = await self.session.execute(statement)
            messages = list(result.scalars().all())

            # 反转列表，使其按时间升序排列
            messages.reverse()

            logger.info(
                "message_repository_load_messages_before",
                session_id=session_id,
                before_message_id=before_message_id,
                limit=limit,
                count=len(messages),
            )
            return messages

        except Exception as e:
            logger.error(
                "message_repository_load_messages_before_failed",
                session_id=session_id,
                before_message_id=before_message_id,
                error=str(e),
            )
            return []

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[Message]:
        """获取会话的最新消息

        对齐 WeKnora 的 GetRecentMessagesBySession 方法。

        Args:
            session_id: 会话 ID
            limit: 限制数量

        Returns:
            消息列表（按创建时间升序排列）
        """
        try:
            statement = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            result = await self.session.execute(statement)
            messages = list(result.scalars().all())

            # 反转列表，使其按时间升序排列
            messages.reverse()

            logger.info(
                "message_repository_get_recent_messages",
                session_id=session_id,
                limit=limit,
                count=len(messages),
            )
            return messages

        except Exception as e:
            logger.error(
                "message_repository_get_recent_messages_failed",
                session_id=session_id,
                error=str(e),
            )
            return []
