"""多 Agent 协作 API

提供 Router、Supervisor、Swarm 等 Agent 模式的创建和交互接口。
同时也提供单个 Agent 的 CRUD 操作。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request as StarletteRequest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.agents.common import (
    delete_agent_system,
    get_agent_system,
    list_agent_systems,
)
from app.api.v1.agents.router import router as router_router
from app.api.v1.agents.schemas import (
    AgentConfig,
    AgentSystemResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Message,
    RouterAgentRequest,
    SupervisorAgentRequest,
    SwarmAgentRequest,
    AgentDetailResponse,
    AgentListResponse,
    AgentPublic as AgentPublicSchema,
    AgentRequest,
    AgentStatsResponse,
    ExecutionListResponse,
)
from app.api.v1.agents.supervisor import router as supervisor_router
from app.api.v1.agents.swarm import router as swarm_router
from app.models.agent import AgentType, AgentStatus
from app.rate_limit.limiter import RateLimit, limiter
from app.auth.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.repositories.agent_async import (
    AgentExecutionRepositoryAsync,
    AgentRepositoryAsync,
)
from app.repositories.base import PaginationParams
from app.services.database import get_session

router = APIRouter(prefix="/agents", tags=["agents"])
logger = get_logger(__name__)


# ==================== Agent 系统 API ====================

# 列出所有 Agent 系统
@router.get("/systems", response_model=list[dict])
@limiter.limit(RateLimit.API)
async def get_systems(request: StarletteRequest):
    """列出所有 Agent 系统"""
    return list_agent_systems()


# 获取单个 Agent 系统
@router.get("/systems/{system_id}", response_model=dict)
@limiter.limit(RateLimit.API)
async def get_system(request: StarletteRequest, system_id: str):
    """获取单个 Agent 系统"""
    system_key = system_id
    system = get_agent_system(system_key)
    return {
        "id": system_id,
        "type": system["type"],
        "name": system["name"],
        "agents": system["agents"],
    }


# 删除 Agent 系统
@router.delete("/systems/{system_id}", response_model=dict)
@limiter.limit(RateLimit.API)
async def delete_system(request: StarletteRequest, system_id: str):
    """删除 Agent 系统"""
    system_key = system_id
    deleted = delete_agent_system(system_key)
    return {"deleted": deleted}


# ==================== 单 Agent CRUD API ====================

# 依赖注入
async def get_agent_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AgentRepositoryAsync:
    """获取 Agent 仓储"""
    return AgentRepositoryAsync(session)


async def get_execution_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AgentExecutionRepositoryAsync:
    """获取执行历史仓储"""
    return AgentExecutionRepositoryAsync(session)


@router.get(
    "/list",
    response_model=AgentListResponse,
    summary="列出所有 Agent",
    description="获取所有 Agent 的列表，支持按类型和状态筛选",
)
@limiter.limit(RateLimit.API)
async def list_agents(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    *,
    tenant_id: TenantIdDep = None,
    agent_type: AgentType | None = Query(None, description="筛选 Agent 类型"),
    status: AgentStatus | None = Query(None, description="筛选 Agent 状态"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
) -> AgentListResponse:
    """列出所有 Agent"""
    params = PaginationParams(page=page, size=size)
    result = await repository.list_by_type_and_status(agent_type, status, params, tenant_id)

    # 转换为公开模型
    agents = [
        AgentPublicSchema(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            agent_type=agent.agent_type.value,
            status=agent.status.value,
            model_name=agent.model_name,
            system_prompt=agent.system_prompt,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens or 0,
            config=agent.config or {},
            created_at=agent.created_at.isoformat(),
        )
        for agent in result.items
    ]

    return AgentListResponse(
        items=agents,
        total=result.total,
        page=result.page,
        size=result.size,
        pages=result.pages,
    )


@router.get(
    "/stats",
    response_model=AgentStatsResponse,
    summary="获取 Agent 统计",
    description="获取 Agent 的统计信息",
)
@limiter.limit(RateLimit.API)
async def get_agent_stats(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    tenant_id: TenantIdDep = None,
) -> AgentStatsResponse:
    """获取 Agent 统计"""
    active_count = (
        await repository.get_active_count_by_tenant(tenant_id)
        if tenant_id is not None
        else await repository.get_active_count()
    )

    # 获取各类型统计
    all_params = PaginationParams(page=1, size=1000)
    all_result = await repository.list_by_type_and_status(None, None, all_params, tenant_id)

    type_counts: dict[str, int] = {}
    for agent in all_result.items:
        agent_type = agent.agent_type.value
        type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

    return AgentStatsResponse(
        total_agents=all_result.total,
        active_agents=active_count,
        agents_by_type=type_counts,
    )


@router.get(
    "/executions",
    response_model=ExecutionListResponse,
    summary="获取执行历史",
    description="获取 Agent 执行历史记录",
)
@limiter.limit(RateLimit.API)
async def list_executions(
    request: StarletteRequest,
    repository: Annotated[AgentExecutionRepositoryAsync, Depends(get_execution_repository)],
    agent_repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    tenant_id: TenantIdDep = None,
    agent_id: int | None = Query(None, description="筛选 Agent ID"),
    limit: int = Query(20, ge=1, le=100, description="限制数量"),
) -> ExecutionListResponse:
    """获取执行历史"""
    if tenant_id is not None and agent_id is not None:
        agent = await agent_repository.get(agent_id)
        if agent is None or agent.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        executions = await repository.list_by_agent(agent_id, limit)
    elif tenant_id is not None:
        agent_ids = await agent_repository.list_ids_by_tenant(tenant_id)
        executions = await repository.list_by_agents(agent_ids, limit)
    elif agent_id:
        executions = await repository.list_by_agent(agent_id, limit)
    else:
        executions = await repository.list_recent(limit)

    items = [
        {
            "id": exc.id,
            "thread_id": exc.thread_id,
            "agent_id": exc.agent_id,
            "status": exc.status,
            "tokens_used": exc.tokens_used or 0,
            "duration_ms": exc.duration_ms or 0,
            "created_at": exc.created_at.isoformat(),
        }
        for exc in executions
    ]

    return ExecutionListResponse(items=items)


@router.get(
    "/{agent_id}",
    response_model=AgentDetailResponse,
    summary="获取 Agent 详情",
    description="根据 ID 获取 Agent 的详细信息",
)
@limiter.limit(RateLimit.API)
async def get_agent(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    agent_id: int,
    tenant_id: TenantIdDep = None,
) -> AgentDetailResponse:
    """获取 Agent 详情"""
    agent = await repository.get(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )
    if tenant_id is not None and agent.tenant_id is not None and agent.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )

    return AgentDetailResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        agent_type=agent.agent_type.value,
        status=agent.status.value,
        model_name=agent.model_name,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens or 0,
        config=agent.config or {},
        created_at=agent.created_at.isoformat(),
    )


@router.post(
    "",
    response_model=AgentDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建 Agent",
    description="创建一个新的 Agent",
)
@limiter.limit(RateLimit.API)
async def create_agent(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    data: AgentRequest,
    *,
    tenant_id: TenantIdDep = None,
) -> AgentDetailResponse:
    """创建 Agent"""
    from app.models.agent import AgentCreate

    # 转换请求格式
    agent_create = AgentCreate(
        name=data.name,
        description=data.description,
        agent_type=data.agent_type,
        model_name=data.model_name,
        system_prompt=data.system_prompt or "",
        temperature=data.temperature,
        max_tokens=data.max_tokens if data.max_tokens > 0 else None,
        config=data.config,
        tenant_id=tenant_id,
        created_by_user_id=getattr(request.state, "user_id", None),
    )

    agent = await repository.create_with_tools(agent_create)

    return AgentDetailResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        agent_type=agent.agent_type.value,
        status=agent.status.value,
        model_name=agent.model_name,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens or 0,
        config=agent.config or {},
        created_at=agent.created_at.isoformat(),
    )


@router.patch(
    "/{agent_id}",
    response_model=AgentDetailResponse,
    summary="更新 Agent",
    description="更新 Agent 的配置",
)
@limiter.limit(RateLimit.API)
async def update_agent(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    agent_id: int,
    *,
    tenant_id: TenantIdDep = None,
    data: AgentRequest,
) -> AgentDetailResponse:
    """更新 Agent"""
    from app.models.agent import AgentUpdate

    # 构建更新数据
    update_data = AgentUpdate(
        name=data.name,
        description=data.description,
        agent_type=data.agent_type,
        model_name=data.model_name,
        system_prompt=data.system_prompt,
        temperature=data.temperature,
        max_tokens=data.max_tokens if data.max_tokens > 0 else None,
        config=data.config,
    )

    existing = await repository.get(agent_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )
    if tenant_id is not None and existing.tenant_id is not None and existing.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )

    agent = await repository.update_agent(agent_id, update_data)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )

    return AgentDetailResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        agent_type=agent.agent_type.value,
        status=agent.status.value,
        model_name=agent.model_name,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens or 0,
        config=agent.config or {},
        created_at=agent.created_at.isoformat(),
    )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除 Agent",
    description="软删除指定的 Agent",
)
@limiter.limit(RateLimit.API)
async def delete_agent(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    agent_id: int,
    *,
    tenant_id: TenantIdDep = None,
) -> None:
    """删除 Agent"""
    existing = await repository.get(agent_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )
    if tenant_id is not None and existing.tenant_id is not None and existing.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )

    success = await repository.soft_delete(agent_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )


# ==================== 包含多 Agent 子路由 ====================

# 包含子路由（必须在单 Agent CRUD 之后，避免路径冲突）
router.include_router(router_router)
router.include_router(supervisor_router)
router.include_router(swarm_router)

__all__ = [
    "router",
    "AgentConfig",
    "RouterAgentRequest",
    "SupervisorAgentRequest",
    "SwarmAgentRequest",
    "ChatRequest",
    "ChatResponse",
    "ChatHistoryResponse",
    "Message",
    "AgentSystemResponse",
    "get_agent_system",
    "list_agent_systems",
    "delete_agent_system",
]
