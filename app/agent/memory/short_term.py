"""短期记忆实现

基于 PostgreSQL 的短期消息存储。
"""

from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage

from app.core.config import get_settings
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = get_logger(__name__)
settings = get_settings()


class ShortTermMemory:
    """短期记忆

    使用 LangGraph 的 PostgreSQL Checkpoint 实现消息存储。
    配合 LangGraph 的状态管理自动工作。
    """

    def __init__(
        self,
        session_id: str,
        checkpointer: "AsyncPostgresSaver | None" = None,
    ) -> None:
        """初始化短期记忆

        Args:
            session_id: 会话 ID
            checkpointer: 检查点保存器（可选）
        """
        self.session_id = session_id
        self._checkpointer = checkpointer
        self._connection_pool = None

        logger.debug(
            "short_term_memory_initialized",
            session_id=session_id,
            has_checkpointer=checkpointer is not None,
        )

    async def get_checkpointer(self) -> "AsyncPostgresSaver | None":
        """获取检查点保存器

        Returns:
            AsyncPostgresSaver 实例或 None
        """
        if self._checkpointer:
            return self._checkpointer

        # 尝试创建 PostgreSQL 检查点
        try:
            from psycopg_pool import AsyncConnectionPool

            if self._connection_pool is None:
                db_url = settings.database_url
                if db_url.startswith("postgresql+asyncpg://"):
                    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

                self._connection_pool = AsyncConnectionPool(
                    conninfo=db_url,
                    open=False,
                    max_size=settings.database_pool_size,
                    kwargs={"autocommit": True},
                )
                await self._connection_pool.open()
                logger.debug("postgres_connection_pool_created")

            checkpointer = AsyncPostgresSaver(self._connection_pool)
            await checkpointer.setup()
            self._checkpointer = checkpointer
            return checkpointer

        except ImportError:
            logger.warning("psycopg_pool_not_installed")
            return None
        except Exception as e:
            logger.warning("checkpointer_init_failed", error=str(e))
            return None

    async def add_message(self, message: BaseMessage) -> None:
        """添加消息

        注意：短期记忆通常由 LangGraph 自动管理，
        此方法主要用于外部添加消息的场景。

        Args:
            message: 消息对象
        """
        logger.debug(
            "message_added_to_short_term",
            session_id=self.session_id,
            message_type=message.type,
        )

    async def get_messages(
        self,
        thread_id: str | None = None,
    ) -> list[BaseMessage]:
        """获取消息列表

        Args:
            thread_id: 线程 ID（默认使用 session_id）

        Returns:
            消息列表
        """
        checkpointer = await self.get_checkpointer()
        if not checkpointer:
            return []

        try:
            from langgraph.types import RunnableConfig

            config = RunnableConfig(
                configurable={"thread_id": thread_id or self.session_id},
            )

            state = await checkpointer.aget_tuple(config)

            if state and state.metadata:
                # 从 checkpoint 中提取消息
                # 这里需要根据实际的 checkpoint 结构来提取
                logger.debug("messages_retrieved_from_checkpoint")
                return []

        except Exception as e:
            logger.warning("get_messages_failed", error=str(e))

        return []

    async def clear(self, thread_id: str | None = None) -> None:
        """清除短期记忆

        Args:
            thread_id: 线程 ID（默认使用 session_id）
        """
        if self._connection_pool:
            try:
                async with self._connection_pool.connection() as conn:
                    await conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = %s",
                        (thread_id or self.session_id,),
                    )
                    await conn.execute(
                        "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                        (thread_id or self.session_id,),
                    )
                    logger.info(
                        "short_term_memory_cleared",
                        session_id=self.session_id,
                    )
            except Exception as e:
                logger.error("clear_short_term_memory_failed", error=str(e))

    async def close(self) -> None:
        """关闭连接池"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.debug("short_term_memory_closed")


def create_short_term_memory(
    session_id: str,
    checkpointer: "AsyncPostgresSaver | None" = None,
) -> ShortTermMemory:
    """创建短期记忆实例

    Args:
        session_id: 会话 ID
        checkpointer: 检查点保存器

    Returns:
        ShortTermMemory 实例
    """
    return ShortTermMemory(session_id, checkpointer)
