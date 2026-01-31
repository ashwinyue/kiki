"""会话服务

提供会话相关的业务逻辑，包括 CRUD、标题生成、Thread 自动创建等。
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import ChatSession, SessionCreate, Thread
from app.observability.logging import get_logger
from app.repositories.base import PaginatedResult, PaginationParams
from app.repositories.session import SessionRepository
from app.repositories.thread import ThreadRepository
from app.schemas.session import SessionCreate as SessionCreateSchema, SessionUpdate

logger = get_logger(__name__)


class SessionService:
    """会话服务"""

    def __init__(self, session: AsyncSession) -> None:
        """初始化会话服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self.session_repo = SessionRepository(session)
        self.thread_repo = ThreadRepository(session)

    async def create_session(
        self,
        data: SessionCreateSchema,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> ChatSession:
        """创建会话，同时创建对应的 Thread

        Args:
            data: 会话创建数据
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            创建的会话对象
        """
        import uuid

        logger.info("creating_session", name=data.name, user_id=user_id)

        # 创建会话
        session_data = SessionCreate(
            name=data.name,
            user_id=user_id,
            tenant_id=tenant_id,
            agent_id=data.agent_id,
            agent_config=data.agent_config,
            context_config=data.context_config,
            extra_data=data.extra_data,
        )
        chat_session = await self.session_repo.create_with_user(session_data, user_id)

        # 创建对应的 Thread（用于 LangGraph 状态持久化）
        thread = Thread(
            id=str(uuid.uuid4()),
            name=data.name,
            session_id=chat_session.id,
            user_id=user_id,
            tenant_id=tenant_id,
            status="active",
        )
        await self.thread_repo.create(thread)

        logger.info(
            "session_created",
            session_id=chat_session.id,
            thread_id=thread.id,
            name=data.name,
        )

        return chat_session

    async def get_session(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> ChatSession | None:
        """获取会话

        Args:
            session_id: 会话 ID
            user_id: 用户 ID（可选，用于权限验证）
            tenant_id: 租户 ID（可选，用于权限验证）

        Returns:
            会话对象，不存在或无权访问返回 None
        """
        session_obj = await self.session_repo.get(session_id)
        if session_obj is None:
            return None

        # 权限验证
        if user_id is not None and session_obj.user_id != user_id:
            return None
        if tenant_id is not None and session_obj.tenant_id is not None and session_obj.tenant_id != tenant_id:
            return None

        return session_obj

    async def get_session_or_404(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> ChatSession:
        """获取会话，不存在则抛出 404 异常

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            会话对象

        Raises:
            HTTPException: 会话不存在或无权访问
        """
        session_obj = await self.get_session(session_id, user_id, tenant_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session_obj

    async def validate_session_access(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> ChatSession:
        """验证会话访问权限

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            会话对象

        Raises:
            HTTPException: 会话不存在或无权访问
        """
        return await self.get_session_or_404(session_id, user_id, tenant_id)

    async def update_session(
        self,
        session_id: str,
        data: SessionUpdate,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> ChatSession | None:
        """更新会话

        Args:
            session_id: 会话 ID
            data: 更新数据
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            更新后的会话对象，不存在返回 None
        """
        session_obj = await self.get_session(session_id, user_id, tenant_id)
        if session_obj is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if hasattr(session_obj, key):
                setattr(session_obj, key, value)

        session_obj.updated_at = datetime.now(UTC)
        await self.session.commit()
        await self.session.refresh(session_obj)

        logger.info("session_updated", session_id=session_id)
        return session_obj

    async def delete_session(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> bool:
        """删除会话（软删除）

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        session_obj = await self.get_session(session_id, user_id, tenant_id)
        if session_obj is None:
            return False

        # 软删除会话
        session_obj.deleted_at = datetime.now(UTC)
        session_obj.updated_at = datetime.now(UTC)

        # 同时归档对应的 Thread
        stmt = select(Thread).where(Thread.session_id == session_id)
        result = await self.session.execute(stmt)
        thread = result.scalar_one_or_none()
        if thread:
            thread.status = "archived"

        await self.session.commit()

        logger.info("session_deleted", session_id=session_id)
        return True

    async def list_sessions(
        self,
        user_id: int | None = None,
        tenant_id: int | None = None,
        page: int = 1,
        size: int = 20,
    ) -> PaginatedResult[ChatSession]:
        """分页获取会话列表

        Args:
            user_id: 用户 ID
            tenant_id: 租户 ID
            page: 页码
            size: 每页数量

        Returns:
            分页结果
        """
        params = PaginationParams(page=page, size=size)

        if user_id is not None:
            return await self.session_repo.list_by_user(user_id, params)

        # 如果只有 tenant_id，使用基础分页
        return await self.session_repo.list_paginated(params, tenant_id=tenant_id)

    async def get_message_count(self, session_id: str) -> int:
        """获取会话的消息数量

        Args:
            session_id: 会话 ID

        Returns:
            消息数量
        """
        return await self.session_repo.get_message_count(session_id)

    async def generate_title(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
        model_name: str | None = None,
    ) -> str:
        """生成会话标题

        基于会话的前几条消息自动生成标题。

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID
            model_name: 使用的模型名称

        Returns:
            生成的标题
        """
        from app.llm import get_llm_service
        from app.repositories.message import MessageRepository

        session_obj = await self.get_session_or_404(session_id, user_id, tenant_id)

        # 获取会话的前几条用户消息
        message_repo = MessageRepository(self.session)
        messages = await message_repo.list_by_role(session_id, "user", limit=3)

        if not messages:
            return "新对话"

        # 提取用户消息内容
        user_input = " ".join([msg.content[:100] for msg in messages])

        try:
            llm_service = get_llm_service()
            prompt = f"""请为以下对话生成一个简短的标题（不超过20个字）：
{user_input}

只返回标题，不要其他内容。"""

            response = await llm_service.ainvoke(prompt, model_name=model_name)
            title = response.strip()[:50] or "新对话"

            # 更新会话标题
            session_obj.name = title
            session_obj.updated_at = datetime.now(UTC)
            await self.session.commit()

            logger.info("session_title_generated", session_id=session_id, title=title)
            return title

        except Exception as e:
            logger.error("generate_title_failed", session_id=session_id, error=str(e))
            return "新对话"


def resolve_effective_user_id(
    request_user_id: str | None,
    state_user_id: str | None,
) -> str | None:
    """解析有效的用户 ID

    优先使用 request 中的 user_id，如果没有则使用 state 中的 user_id。
    如果两者都存在但不匹配，则抛出异常。

    Args:
        request_user_id: 请求中的用户 ID
        state_user_id: 请求 state 中的用户 ID

    Returns:
        有效的用户 ID

    Raises:
        HTTPException: 如果用户 ID 不匹配
    """
    if (
        state_user_id is not None
        and request_user_id is not None
        and str(state_user_id) != str(request_user_id)
    ):
        raise HTTPException(status_code=403, detail="User mismatch")
    if state_user_id is not None and request_user_id is None:
        return str(state_user_id)
    return request_user_id
