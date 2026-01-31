"""API 依赖注入模块

提供 FastAPI 路由的依赖注入函数。
"""

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 数据库依赖 ==============


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（依赖注入）

    这是 get_session 的别名，用于 API 路由。

    Yields:
        异步会话实例
    """
    from app.infra.database import get_session

    async for session in get_session():
        yield session


# ============== 租户依赖 ==============


async def get_tenant_id(request: Request) -> int | None:
    """获取租户 ID（依赖注入）

    从中间件设置的请求状态中获取租户 ID。

    Args:
        request: FastAPI 请求对象

    Returns:
        租户 ID 或 None
    """
    return getattr(request.state, "tenant_id", None)


# 重新导出其他有用的依赖
__all__ = [
    "get_db",
    "get_tenant_id",
]
