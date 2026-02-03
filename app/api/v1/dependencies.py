"""API v1 统一依赖注入模块

提供 FastAPI v1 路由的依赖注入函数和类型别名。
采用链式依赖注入模式，提高代码可维护性。

参考设计模式：
- FastAPI 标准依赖注入模式
- 外部项目的链式 Depends 模式

使用示例:
    ```python
    from app.api.v1.dependencies import DbDep, TenantIdDep, AgentDep

    @router.get("/items/{id}")
    async def get_item(
        id: str,
        db: DbDep,
        tenant_id: TenantIdDep,
    ):
        # ...
    ```
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import ChatAgent  # 使用 ChatAgent 替代已废弃的 LangGraphAgent
from app.api.dependencies import get_db
from app.config.dependencies import (
    get_agent_dep,
    get_context_manager_dep,
    get_llm_service_dep,
    get_memory_manager_dep,
)
from app.middleware.auth import (
    get_current_tenant_id,
    get_current_user_dep,
    require_current_user,
    require_tenant,
)
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.agent.memory.context import ContextManager
    from app.agent.memory.manager import MemoryManager
    from app.llm import LLMService

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

# Agent 依赖
AgentDep = Annotated[ChatAgent, Depends(get_agent_dep)]

# LLM 服务依赖
LlmServiceDep = Annotated["LLMService", Depends(get_llm_service_dep)]

# Memory Manager 依赖
MemoryManagerDep = Annotated["MemoryManager", Depends(get_memory_manager_dep)]

# Context Manager 依赖（已废弃，Agent 现在使用 PostgreSQL Checkpointer）
# ContextManagerDep = Annotated["ContextManager", Depends(get_context_manager_dep)]


# ============== 链式依赖注入函数 ==============
# 按照外部项目的模式，提供链式依赖注入


async def get_session_service_dep(
    db: DbDep,
) -> AsyncIterator["SessionService"]:
    """获取会话服务（链式依赖注入）

    依赖: db → SessionService

    Args:
        db: 数据库会话

    Yields:
        SessionService 实例
    """
    from app.services.core.session_service import SessionService

    service = SessionService(db)
    try:
        yield service
    finally:
        pass  # SessionService 不需要清理


async def get_agent_with_memory_dep(
    db: DbDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    session_id: str | None = None,
) -> AsyncIterator[ChatAgent]:
    """获取带记忆的 Agent（链式依赖注入）

    依赖: db + tenant_id + user_id → Agent

    Args:
        db: 数据库会话
        tenant_id: 租户 ID
        user_id: 用户 ID
        session_id: 会话 ID（可选）

    Yields:
        ChatAgent 实例
    """
    agent = await get_agent_dep(session_id, user_id)
    try:
        yield agent
    except Exception as e:
        logger.error("agent_with_memory_error", error=str(e))
        raise


async def get_chat_graph_dep(
    system_prompt: str | None = None,
    llm_service: LlmServiceDep = None,
) -> "CompiledStateGraph":
    """获取聊天图（链式依赖注入）

    依赖: llm_service → CompiledStateGraph

    Args:
        system_prompt: 系统提示词
        llm_service: LLM 服务

    Returns:
        编译后的聊天图
    """

    from app.agent.graph.builder import build_chat_graph

    # 这里可以添加缓存逻辑
    return build_chat_graph(llm_service, system_prompt)


# ============== 服务类依赖 ==============


async def get_knowledge_service_dep(
    db: DbDep,
    tenant_id: TenantIdDep,
) -> AsyncIterator["KnowledgeService"]:
    """获取知识库服务（链式依赖注入）

    ⚠️ 已废弃：知识库相关模块已移除

    依赖: db + tenant_id → KnowledgeService

    Args:
        db: 数据库会话
        tenant_id: 租户 ID

    Yields:
        KnowledgeService 实例

    Raises:
        NotImplementedError: 此服务已移除
    """
    raise NotImplementedError("KnowledgeService 已移除（RAG 相关功能已删除）")


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


async def get_task_service_dep(
    db: DbDep,
    tenant_id: TenantIdDep,
) -> AsyncIterator["TaskService"]:
    """获取任务服务（链式依赖注入）

    ⚠️ 已废弃：Celery 任务队列模块已移除

    依赖: db + tenant_id → TaskService

    Args:
        db: 数据库会话
        tenant_id: 租户 ID

    Yields:
        TaskService 实例

    Raises:
        NotImplementedError: 此服务已移除
    """
    raise NotImplementedError("TaskService 已移除（Celery 任务队列已删除）")


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
    "AgentDep",
    "LlmServiceDep",
    "MemoryManagerDep",
    "ContextManagerDep",
    # 链式依赖注入函数
    "get_session_service_dep",
    "get_agent_with_memory_dep",
    "get_chat_graph_dep",
    "get_knowledge_service_dep",
    "get_model_service_dep",
    "get_task_service_dep",
    # 辅助函数
    "validate_session_access_dep",
    "resolve_effective_user_id_dep",
]
