"""LangGraph Store 持久化

实现 LangGraph 的 Store 接口，用于跨会话的长期记忆存储。
支持用户偏好、对话历史等持久化数据。
"""

import logging
from typing import Any, Iterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============== Store 接口实现 ==============

class PostgresStore:
    """PostgreSQL 实现的 LangGraph Store

    用于存储跨会话的长期记忆，如用户偏好、对话摘要等。

    存储结构：
    - namespace: 命名空间，如 "user:123", "agent:456", "session:789"
    - key: 键名，如 "preferences", "summary", "profile"
    - value: JSONB 值
    """

    def __init__(self, session: AsyncSession):
        """初始化 Store

        Args:
            session: SQLAlchemy 异步会话
        """
        self.session = session

    async def aget(self, namespace: str, key: str) -> Any | None:
        """获取存储值

        Args:
            namespace: 命名空间
            key: 键名

        Returns:
            存储的值，不存在返回 None
        """
        try:
            query = text(
                """
                SELECT value FROM memories
                WHERE namespace = :namespace AND key = :key
                LIMIT 1
                """
            )
            result = await self.session.execute(
                query, {"namespace": namespace, "key": key}
            )
            row = result.fetchone()
            if row:
                logger.debug("store_get_found", namespace=namespace, key=key)
                return row[0]
            logger.debug("store_get_not_found", namespace=namespace, key=key)
            return None

        except Exception as e:
            logger.error("store_get_failed", namespace=namespace, key=key, error=str(e))
            return None

    async def aput(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """存储值

        Args:
            namespace: 命名空间
            key: 键名
            value: 要存储的值（会被序列化为 JSON）
        """
        import json

        try:
            # 使用 UPSERT 语法
            query = text(
                """
                INSERT INTO memories (namespace, key, value)
                VALUES (:namespace, :key, :value::jsonb)
                ON CONFLICT (namespace, key)
                DO UPDATE SET value = :value::jsonb, updated_at = NOW()
                """
            )
            await self.session.execute(
                query,
                {
                    "namespace": namespace,
                    "key": key,
                    "value": json.dumps(value),
                },
            )
            await self.session.commit()
            logger.debug("store_put_success", namespace=namespace, key=key)

        except Exception as e:
            logger.error("store_put_failed", namespace=namespace, key=key, error=str(e))
            await self.session.rollback()

    async def adelete(self, namespace: str, key: str) -> None:
        """删除存储值

        Args:
            namespace: 命名空间
            key: 键名
        """
        try:
            query = text(
                """
                DELETE FROM memories
                WHERE namespace = :namespace AND key = :key
                """
            )
            await self.session.execute(
                query, {"namespace": namespace, "key": key}
            )
            await self.session.commit()
            logger.debug("store_delete_success", namespace=namespace, key=key)

        except Exception as e:
            logger.error("store_delete_failed", namespace=namespace, key=key, error=str(e))
            await self.session.rollback()

    async def asearch(
        self,
        namespace_prefix: str,
        limit: int = 10,
    ) -> Iterator[tuple[str, str, Any]]:
        """搜索存储值

        Args:
            namespace_prefix: 命名空间前缀
            limit: 最大返回数量

        Yields:
            (namespace, key, value) 元组
        """
        try:
            query = text(
                """
                SELECT namespace, key, value
                FROM memories
                WHERE namespace LIKE :prefix || '%'
                ORDER BY updated_at DESC
                LIMIT :limit
                """
            )
            result = await self.session.execute(
                query, {"prefix": namespace_prefix, "limit": limit}
            )
            for row in result:
                yield (row[0], row[1], row[2])

        except Exception as e:
            logger.error(
                "store_search_failed", namespace_prefix=namespace_prefix, error=str(e)
            )

    async def alist_namespaces(self, prefix: str = "", limit: int = 100) -> list[str]:
        """列出所有命名空间

        Args:
            prefix: 命名空间前缀过滤
            limit: 最大返回数量

        Returns:
            命名空间列表
        """
        try:
            query = text(
                """
                SELECT DISTINCT namespace
                FROM memories
                WHERE namespace LIKE :prefix || '%'
                ORDER BY namespace
                LIMIT :limit
                """
            )
            result = await self.session.execute(
                query, {"prefix": prefix, "limit": limit}
            )
            return [row[0] for row in result]

        except Exception as e:
            logger.error("store_list_namespaces_failed", prefix=prefix, error=str(e))
            return []

    async def akeys(self, namespace: str) -> list[str]:
        """列出命名空间下的所有键

        Args:
            namespace: 命名空间

        Returns:
            键列表
        """
        try:
            query = text(
                """
                SELECT key FROM memories
                WHERE namespace = :namespace
                ORDER BY key
                """
            )
            result = await self.session.execute(query, {"namespace": namespace})
            return [row[0] for row in result]

        except Exception as e:
            logger.error("store_keys_failed", namespace=namespace, error=str(e))
            return []


# ============== 工厂函数 ==============

async def create_store(session: AsyncSession) -> PostgresStore:
    """创建 Store 实例

    Args:
        session: SQLAlchemy 异步会话

    Returns:
        PostgresStore 实例
    """
    return PostgresStore(session)


# ============== 内存 Store（开发用）=============

class InMemoryStore:
    """内存存储实现

    用于开发测试，数据不持久化。
    """

    def __init__(self):
        self._data: dict[str, dict[str, Any]] = {}

    async def aget(self, namespace: str, key: str) -> Any | None:
        """获取值"""
        return self._data.get(namespace, {}).get(key)

    async def aput(self, namespace: str, key: str, value: Any) -> None:
        """存储值"""
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = value

    async def adelete(self, namespace: str, key: str) -> None:
        """删除值"""
        if namespace in self._data:
            self._data[namespace].pop(key, None)

    async def asearch(
        self, namespace_prefix: str, limit: int = 10
    ) -> Iterator[tuple[str, str, Any]]:
        """搜索值"""
        count = 0
        for ns, data in self._data.items():
            if ns.startswith(namespace_prefix):
                for key, value in data.items():
                    if count >= limit:
                        return
                    yield (ns, key, value)
                    count += 1

    async def alist_namespaces(self, prefix: str = "", limit: int = 100) -> list[str]:
        """列出命名空间"""
        return [
            ns for ns in self._data.keys()
            if ns.startswith(prefix)][:limit]

    async def akeys(self, namespace: str) -> list[str]:
        """列出键"""
        return list(self._data.get(namespace, {}).keys())


# ============== 辅助函数 ==============

def user_namespace(user_id: int | str) -> str:
    """生成用户命名空间

    Args:
        user_id: 用户 ID

    Returns:
        命名空间字符串，如 "user:123"
    """
    return f"user:{user_id}"


def session_namespace(session_id: str) -> str:
    """生成会话命名空间

    Args:
        session_id: 会话 ID

    Returns:
        命名空间字符串，如 "session:abc-123"
    """
    return f"session:{session_id}"


def agent_namespace(agent_id: str) -> str:
    """生成 Agent 命名空间

    Args:
        agent_id: Agent ID

    Returns:
        命名空间字符串，如 "agent:agent-1"
    """
    return f"agent:{agent_id}"


# ============== 预定义键名 ==============

class StoreKeys:
    """预定义的存储键名"""

    # 用户相关
    USER_PREFERENCES = "preferences"  # 用户偏好设置
    USER_PROFILE = "profile"  # 用户资料
    USER_HISTORY = "history"  # 用户历史摘要

    # 会话相关
    SESSION_SUMMARY = "summary"  # 会话摘要
    SESSION_CONTEXT = "context"  # 会话上下文
    SESSION_METADATA = "metadata"  # 会话元数据

    # Agent 相关
    AGENT_CONFIG = "config"  # Agent 配置
    AGENT_STATE = "state"  # Agent 状态
    AGENT_MEMORY = "memory"  # Agent 记忆
