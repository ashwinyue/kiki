"""API v1 统一依赖注入模块

提供 FastAPI v1 路由的依赖注入函数和类型别名。
采用链式依赖注入模式，提高代码可维护性。
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.config.dependencies import get_llm_service_dep, get_memory_manager_dep
from app.middleware.auth import (
    get_current_tenant_id,
    get_current_user_dep,
    require_current_user,
    require_tenant,
)
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.agent.memory.manager import MemoryManager
    from app.llm import LLMService
    from app.services.llm.model_service import ModelService

logger = get_logger(__name__)

# 数据库会话依赖
DbDep = Annotated[AsyncSession, Depends(get_db)]

# 租户 ID 依赖（可选）
TenantIdDep = Annotated[int | None, Depends(get_current_tenant_id)]

# 租户 ID 依赖（必选）
RequiredTenantIdDep = Annotated[int, Depends(require_tenant)]

# 用户 ID 依赖（可选）
UserIdDep = Annotated[str | None, Depends(get_current_user_dep)]

# 用户 ID 依赖（必选）
RequiredUserIdDep = Annotated[str, Depends(require_current_user)]

# LLM 服务依赖
LlmServiceDep = Annotated["LLMService", Depends(get_llm_service_dep)]

# Memory Manager 依赖
MemoryManagerDep = Annotated["MemoryManager", Depends(get_memory_manager_dep)]


# ============== 服务类依赖 ==============


async def get_model_service_dep(
    db: DbDep,
    tenant_id: TenantIdDep,
) -> AsyncIterator["ModelService"]:
    """获取模型服务（链式依赖注入）

    依赖: db + tenant_id → ModelService

    Args:
        db: 数据库会话
        tenant_id: 租户 ID

    Yields:
        ModelService 实例
    """
    from app.services.llm.model_service import ModelService

    service = ModelService(db, tenant_id)
    try:
        yield service
    finally:
        pass


# ============== 辅助函数 ==============


async def validate_session_access_dep(
    session_id: str,
    user_id: str | None,
    tenant_id: int | None,
    db: DbDep,
) -> None:
    """验证会话访问权限（依赖注入）

    Args:
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID
        db: 数据库会话

    Raises:
        HTTPException: 会话不存在或无权访问
    """
    from app.services.core.session_service import SessionService

    service = SessionService(db)
    await service.validate_session_access(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )


async def resolve_effective_user_id_dep(
    user_id: UserIdDep,
    tenant_id: TenantIdDep,
) -> str:
    """解析有效的用户 ID（依赖注入）

    处理匿名用户和认证用户的情况。

    Args:
        user_id: 用户 ID（可能为 None）
        tenant_id: 租户 ID（可能为 None）

    Returns:
        有效的用户 ID（匿名用户生成临时 ID）
    """
    from app.services.core.session_service import resolve_effective_user_id

    return resolve_effective_user_id(user_id, tenant_id)


# ============== 导出 ==============
# 推荐使用类型别名，而不是直接使用 Depends


__all__ = [
    # 类型别名
    "DbDep",
    "TenantIdDep",
    "RequiredTenantIdDep",
    "UserIdDep",
    "RequiredUserIdDep",
    "LlmServiceDep",
    "MemoryManagerDep",
    # 服务类依赖
    "get_model_service_dep",
    # 辅助函数
    "validate_session_access_dep",
    "resolve_effective_user_id_dep",
]
