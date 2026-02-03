"""消息服务

提供消息的业务逻辑封装。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.repositories.message import MessageRepository

logger = get_logger(__name__)


class MessageService:
    """消息服务"""

    def __init__(self, db: AsyncSession) -> None:
        """初始化消息服务

        Args:
            db: 数据库会话
        """
        self.db = db
        self.repository = MessageRepository(db)

    async def list_messages(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
        page: int = 1,
        size: int = 50,
    ) -> dict:
        """获取消息列表

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID
            page: 页码
            size: 每页数量

        Returns:
            消息列表
        """
        from app.repositories.base import PaginationParams

        params = PaginationParams(page=page, size=size)
        result = await self.repository.list_by_session(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            params=params,
        )

        return {
            "items": result.items,
            "total": result.total,
            "page": page,
            "size": size,
        }

    async def get_message(
        self,
        message_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ):
        """获取消息详情

        Args:
            message_id: 消息 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            消息对象
        """
        return await self.repository.get_by_id(
            message_id, user_id=user_id, tenant_id=tenant_id
        )

    async def update_message(
        self,
        message_id: str,
        content: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ):
        """更新消息内容

        Args:
            message_id: 消息 ID
            content: 新内容
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            更新后的消息对象
        """
        return await self.repository.update_content(
            message_id=message_id,
            content=content,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def delete_message(
        self,
        message_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> bool:
        """删除消息

        Args:
            message_id: 消息 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        return await self.repository.delete(
            message_id, user_id=user_id, tenant_id=tenant_id
        )
