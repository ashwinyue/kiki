"""消息服务

提供消息相关的业务逻辑，包括 CRUD、编辑+重新生成、搜索等。
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import get_agent
from app.agent.message_utils import get_last_user_message
from app.infra.database import message_repository, session_repository
from app.models.database import Message, MessageCreate, MessageUpdate as MessageUpdateModel
from app.observability.logging import get_logger
from app.repositories.base import PaginatedResult, PaginationParams
from app.schemas.message import (
    MessageLoadRequest,
    MessageRegenerateRequest,
    MessageUpdate as MessageUpdateSchema,
)

logger = get_logger(__name__)


class MessageService:
    """消息服务"""

    def __init__(self, session: AsyncSession) -> None:
        """初始化消息服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self.message_repo = message_repository(session)
        self.session_repo = session_repository(session)

    async def get_message(
        self,
        message_id: str,
        session_id: str | None = None,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> Message | None:
        """获取消息

        Args:
            message_id: 消息 ID
            session_id: 会话 ID（可选，用于验证）
            user_id: 用户 ID（可选，用于权限验证）
            tenant_id: 租户 ID（可选，用于权限验证）

        Returns:
            消息对象，不存在或无权访问返回 None
        """
        message = await self.message_repo.get(message_id)
        if message is None:
            return None

        # 验证会话
        if session_id is not None and message.session_id != session_id:
            return None

        # 验证用户权限
        if user_id is not None or tenant_id is not None:
            session_obj = await self.session_repo.get(message.session_id)
            if session_obj is None:
                return None
            if user_id is not None and session_obj.user_id != user_id:
                return None
            if tenant_id is not None and session_obj.tenant_id is not None and session_obj.tenant_id != tenant_id:
                return None

        return message

    async def get_message_or_404(
        self,
        message_id: str,
        session_id: str | None = None,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> Message:
        """获取消息，不存在则抛出 404 异常

        Args:
            message_id: 消息 ID
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            消息对象

        Raises:
            HTTPException: 消息不存在或无权访问
        """
        message = await self.get_message(message_id, session_id, user_id, tenant_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found")
        return message

    async def list_messages(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
        page: int = 1,
        size: int = 50,
    ) -> PaginatedResult[Message]:
        """分页获取会话的消息列表

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID
            page: 页码
            size: 每页数量

        Returns:
            分页结果
        """
        # 验证会话访问权限
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None and session_obj.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session access denied")
        if tenant_id is not None and session_obj.tenant_id is not None and session_obj.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Session tenant mismatch")

        params = PaginationParams(page=page, size=size)
        return await self.message_repo.list_by_session(session_id, params)

    async def update_message(
        self,
        message_id: str,
        data: MessageUpdateSchema,
        regenerate_request: MessageRegenerateRequest | None = None,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> Message:
        """更新消息

        Args:
            message_id: 消息 ID
            data: 更新数据
            regenerate_request: 重新生成请求
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            更新后的消息对象

        Raises:
            HTTPException: 消息不存在或无权访问
        """
        message = await self.get_message_or_404(message_id, user_id=user_id, tenant_id=tenant_id)

        # 只有用户消息可以编辑
        if message.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be edited")

        # 更新消息内容
        message.content = data.content
        message.updated_at = datetime.now(UTC)

        await self.session.commit()
        await self.session.refresh(message)

        logger.info("message_updated", message_id=message_id)

        # 如果需要重新生成
        if regenerate_request and regenerate_request.regenerate:
            await self._regenerate_after(message)

        return message

    async def _regenerate_after(self, message: Message) -> None:
        """删除指定消息之后的所有消息，并触发重新生成

        Args:
            message: 编辑后的消息
        """
        # 删除该消息之后的所有消息
        stmt = delete(Message).where(
            Message.session_id == message.session_id,
            Message.created_at > message.created_at,
        )
        await self.session.execute(stmt)
        await self.session.commit()

        logger.info("messages_deleted_after_edit", session_id=message.session_id, message_id=message.id)

        # TODO: 触发 Agent 重新生成回复
        # 这里需要异步触发，不阻塞当前请求
        # 可以使用消息队列或后台任务

    async def delete_message(
        self,
        message_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> bool:
        """删除消息（软删除）

        Args:
            message_id: 消息 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        message = await self.get_message(message_id, user_id=user_id, tenant_id=tenant_id)
        if message is None:
            return False

        # 软删除
        message.deleted_at = datetime.now(UTC)
        message.updated_at = datetime.now(UTC)

        await self.session.commit()

        logger.info("message_deleted", message_id=message_id)
        return True

    async def search_messages(
        self,
        session_id: str,
        query: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
        limit: int = 20,
    ) -> list[Message]:
        """在会话中搜索消息

        Args:
            session_id: 会话 ID
            query: 搜索关键词
            user_id: 用户 ID
            tenant_id: 租户 ID
            limit: 限制数量

        Returns:
            匹配的消息列表
        """
        # 验证会话访问权限
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None and session_obj.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session access denied")
        if tenant_id is not None and session_obj.tenant_id is not None and session_obj.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Session tenant mismatch")

        # 搜索消息
        try:
            pattern = f"%{query}%"
            stmt = (
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.content.ilike(pattern),
                    Message.deleted_at.is_(None),
                )
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            messages = list(result.scalars().all())

            logger.info("messages_searched", session_id=session_id, query=query, count=len(messages))
            return messages

        except Exception as e:
            logger.error("message_search_failed", session_id=session_id, query=query, error=str(e))
            return []

    async def delete_messages_by_session(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> int:
        """删除会话的所有消息

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            删除的消息数量
        """
        # 验证会话访问权限
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None and session_obj.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session access denied")
        if tenant_id is not None and session_obj.tenant_id is not None and session_obj.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Session tenant mismatch")

        return await self.message_repo.delete_by_session(session_id)

    async def load_messages(
        self,
        session_id: str,
        request: MessageLoadRequest,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> tuple[list[Message], bool]:
        """分页加载会话消息

        对齐 WeKnora 的 LoadMessages API。
        如果提供了 message_id，加载该消息之前的消息；
        否则加载最新的消息。

        Args:
            session_id: 会话 ID
            request: 加载请求参数
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            (消息列表, 是否还有更多消息)
        """
        # 验证会话访问权限
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None and session_obj.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session access denied")
        if tenant_id is not None and session_obj.tenant_id is not None and session_obj.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Session tenant mismatch")

        # 获取消息
        if request.message_id:
            messages = await self.message_repo.load_messages_before(
                session_id,
                before_message_id=request.message_id,
                limit=request.limit,
            )
        else:
            messages = await self.message_repo.get_recent_messages(
                session_id,
                limit=request.limit,
            )

        # 判断是否还有更多消息
        has_more = len(messages) >= request.limit

        logger.info(
            "messages_loaded",
            session_id=session_id,
            message_id=request.message_id,
            limit=request.limit,
            count=len(messages),
            has_more=has_more,
        )

        return messages, has_more
