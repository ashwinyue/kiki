"""会话状态管理服务

提供会话状态的跟踪和管理功能，支持：
- 跟踪活跃会话
- 停止信号管理
- 状态查询

使用 Redis 存储状态，支持分布式部署。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.observability.logging import get_logger

logger = get_logger(__name__)


class SessionState(str, Enum):
    """会话状态枚举"""

    IDLE = "idle"  # 空闲
    RUNNING = "running"  # 运行中
    STOPPING = "stopping"  # 停止中
    STOPPED = "stopped"  # 已停止
    ERROR = "error"  # 错误


class SessionStateInfo(BaseModel):
    """会话状态信息"""

    session_id: str = Field(..., description="会话 ID")
    state: SessionState = Field(..., description="会话状态")
    user_id: int | None = Field(None, description="用户 ID")
    tenant_id: int | None = Field(None, description="租户 ID")
    started_at: datetime | None = Field(None, description="开始时间")
    updated_at: datetime | None = Field(None, description="更新时间")
    metadata: dict[str, str | int | bool] | None = Field(None, description="额外元数据")


class SessionStateManager:
    """会话状态管理器

    使用 Redis 存储会话状态，支持分布式部署。

    Examples:
        ```python
        manager = SessionStateManager()

        # 标记会话运行中
        await manager.set_running("session-123", user_id=1)

        # 检查会话是否运行中
        is_running = await manager.is_running("session-123")

        # 请求停止会话
        await manager.request_stop("session-123")

        # 检查是否已请求停止
        is_stopping = await manager.is_stopping("session-123")

        # 清除会话状态
        await manager.clear("session-123")
        ```
    """

    def __init__(self) -> None:
        """初始化会话状态管理器"""
        from app.infra.redis import get_cache

        self._cache = get_cache(key_prefix="kiki:session:state:")
        self._stop_cache = get_cache(key_prefix="kiki:session:stop:")

    def _make_state_key(self, session_id: str) -> str:
        return f"state:{session_id}"

    def _make_stop_key(self, session_id: str) -> str:
        return f"stopped:{session_id}"

    async def get_state(self, session_id: str) -> SessionStateInfo | None:
        """获取会话状态

        Args:
            session_id: 会话 ID

        Returns:
            会话状态信息，不存在返回 None
        """
        try:
            import json

            key = self._make_state_key(session_id)
            value = await self._cache.get(key)

            if value is None:
                return None

            data = json.loads(value) if isinstance(value, str) else value
            return SessionStateInfo(**data)

        except Exception as e:
            logger.error("get_session_state_failed", session_id=session_id, error=str(e))
            return None

    async def set_state(
        self,
        session_id: str,
        state: SessionState,
        user_id: int | None = None,
        tenant_id: int | None = None,
        metadata: dict[str, str | int | bool] | None = None,
        ttl: int = 3600,
    ) -> bool:
        """设置会话状态

        Args:
            session_id: 会话 ID
            state: 会话状态
            user_id: 用户 ID
            tenant_id: 租户 ID
            metadata: 额外元数据
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        try:
            now = datetime.now(UTC)

            info = SessionStateInfo(
                session_id=session_id,
                state=state,
                user_id=user_id,
                tenant_id=tenant_id,
                started_at=now if state == SessionState.RUNNING else None,
                updated_at=now,
                metadata=metadata,
            )

            key = self._make_state_key(session_id)
            value = info.model_dump_json()

            success = await self._cache.set(key, value, ttl=ttl)

            if success:
                logger.debug(
                    "session_state_set",
                    session_id=session_id,
                    state=state.value,
                )

            return success

        except Exception as e:
            logger.error("set_session_state_failed", session_id=session_id, error=str(e))
            return False

    async def set_running(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> bool:
        """标记会话运行中

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            是否设置成功
        """
        return await self.set_state(
            session_id,
            SessionState.RUNNING,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def set_idle(self, session_id: str) -> bool:
        """标记会话空闲

        Args:
            session_id: 会话 ID

        Returns:
            是否设置成功
        """
        state_info = await self.get_state(session_id)
        user_id = state_info.user_id if state_info else None
        tenant_id = state_info.tenant_id if state_info else None

        return await self.set_state(
            session_id,
            SessionState.IDLE,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def set_stopping(self, session_id: str) -> bool:
        """标记会话停止中

        Args:
            session_id: 会话 ID

        Returns:
            是否设置成功
        """
        state_info = await self.get_state(session_id)
        user_id = state_info.user_id if state_info else None
        tenant_id = state_info.tenant_id if state_info else None

        return await self.set_state(
            session_id,
            SessionState.STOPPING,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def set_stopped(self, session_id: str) -> bool:
        """标记会话已停止

        Args:
            session_id: 会话 ID

        Returns:
            是否设置成功
        """
        state_info = await self.get_state(session_id)
        user_id = state_info.user_id if state_info else None
        tenant_id = state_info.tenant_id if state_info else None

        return await self.set_state(
            session_id,
            SessionState.STOPPED,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    async def is_running(self, session_id: str) -> bool:
        """检查会话是否运行中

        Args:
            session_id: 会话 ID

        Returns:
            True 表示运行中
        """
        state_info = await self.get_state(session_id)
        return state_info is not None and state_info.state == SessionState.RUNNING

    async def is_stopping(self, session_id: str) -> bool:
        """检查会话是否停止中

        Args:
            session_id: 会话 ID

        Returns:
            True 表示停止中
        """
        state_info = await self.get_state(session_id)
        return state_info is not None and state_info.state in (
            SessionState.STOPPING,
            SessionState.STOPPED,
        )

    async def request_stop(self, session_id: str) -> bool:
        """请求停止会话

        设置停止标记和状态为 STOPPING。

        Args:
            session_id: 会话 ID

        Returns:
            是否成功设置停止标记
        """
        try:
            stop_key = self._make_stop_key(session_id)
            stop_success = await self._stop_cache.set(stop_key, "1", ttl=300)

            state_success = await self.set_stopping(session_id)

            if stop_success or state_success:
                logger.info("session_stop_requested", session_id=session_id)

            return stop_success

        except Exception as e:
            logger.error("request_stop_failed", session_id=session_id, error=str(e))
            return False

    async def is_stop_requested(self, session_id: str) -> bool:
        """检查是否已请求停止

        Args:
            session_id: 会话 ID

        Returns:
            True 表示已请求停止
        """
        try:
            stop_key = self._make_stop_key(session_id)
            return await self._stop_cache.exists(stop_key)
        except Exception as e:
            logger.error("check_stop_requested_failed", session_id=session_id, error=str(e))
            return False

    async def clear_stop(self, session_id: str) -> bool:
        """清除停止标记

        Args:
            session_id: 会话 ID

        Returns:
            是否清除成功
        """
        try:
            stop_key = self._make_stop_key(session_id)
            success = await self._stop_cache.delete(stop_key)

            await self.set_idle(session_id)

            if success:
                logger.info("session_stop_cleared", session_id=session_id)

            return success

        except Exception as e:
            logger.error("clear_stop_failed", session_id=session_id, error=str(e))
            return False

    async def clear(self, session_id: str) -> bool:
        """清除会话状态

        Args:
            session_id: 会话 ID

        Returns:
            是否清除成功
        """
        try:
            state_key = self._make_state_key(session_id)
            stop_key = self._make_stop_key(session_id)

            await self._cache.delete(state_key)
            await self._stop_cache.delete(stop_key)

            logger.debug("session_state_cleared", session_id=session_id)
            return True

        except Exception as e:
            logger.error("clear_state_failed", session_id=session_id, error=str(e))
            return False

    async def list_running_sessions(
        self,
        tenant_id: int | None = None,
        user_id: int | None = None,
    ) -> list[SessionStateInfo]:
        """列出运行中的会话

        注意：此方法在大量会话时可能性能较差，仅用于管理接口。

        Args:
            tenant_id: 租户 ID 筛选
            user_id: 用户 ID 筛选

        Returns:
            运行中的会话列表
        """
        # 由于 Redis 的键扫描限制，这里返回空列表
        # 实际应用中可以考虑维护一个活跃会话集合
        logger.warning(
            "list_running_sessions_not_implemented",
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return []


# 全局实例
_session_state_manager: SessionStateManager | None = None


def get_session_state_manager() -> SessionStateManager:
    """获取会话状态管理器实例（单例）

    Returns:
        SessionStateManager 实例
    """
    global _session_state_manager
    if _session_state_manager is None:
        _session_state_manager = SessionStateManager()
    return _session_state_manager


__all__ = [
    "SessionState",
    "SessionStateInfo",
    "SessionStateManager",
    "get_session_state_manager",
]
