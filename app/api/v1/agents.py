"""多 Agent 协作 API

提供 Router、Supervisor、Swarm 等 Agent 模式的创建和交互接口。
同时也提供单个 Agent 的 CRUD 操作。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request as StarletteRequest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import get_agent
from app.agent.factory import create_api_handoff_agent, create_api_agent_node
from app.agent.message_utils import extract_ai_content
from app.agent.multi_agent import RouterAgent, SupervisorAgent, create_swarm
from app.agent.state import create_state_from_input
from app.infra.database import get_session
from app.llm import get_llm_service
from app.models.agent import AgentCreate, AgentStatus, AgentType, AgentUpdate
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.repositories.agent_async import (
    AgentExecutionRepositoryAsync,
    AgentRepositoryAsync,
)
from app.repositories.base import PaginationParams
from app.schemas.agent import (
    AgentConfig,
    AgentDetailResponse,
    AgentListResponse,
    AgentPublic,
    AgentRequest,
    AgentStatsResponse,
    AgentSystemResponse,
    ExecutionItem,
    ExecutionListResponse,
    MultiAgentChatRequest as ChatRequest,
    MultiAgentChatResponse as ChatResponse,
    RouterAgentRequest,
    SupervisorAgentRequest,
    SwarmAgentRequest,
)
from app.auth.middleware import TenantIdDep
from app.services.agent_system_service import get_agent_system_service
from app.services.session_service import resolve_effective_user_id
from app.infra.database import session_scope

router = APIRouter(prefix="/agents", tags=["agents"])
logger = get_logger(__name__)

# Agent 系统服务
_agent_system_service = get_agent_system_service()


# ============== 辅助函数 ==============


async def _validate_session_access(
    session_id: str,
    user_id: str | None,
    tenant_id: int | None,
) -> None:
    """验证会话访问权限

    Args:
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID

    Raises:
        HTTPException: 会话不存在或无权访问
    """
    from app.services.session_service import SessionService

    async with session_scope() as db_session:
        service = SessionService(db_session)
        await service.validate_session_access(
            session_id,
            user_id=int(user_id) if user_id else None,
            tenant_id=tenant_id,
        )


# ============== Agent 系统 CRUD ==============


@router.get("/systems", response_model=list[dict])
@limiter.limit(RateLimit.API)
async def get_systems(request: StarletteRequest):
    """列出所有 Agent 系统"""
    return _agent_system_service.list_systems()


@router.get("/systems/{system_id}", response_model=dict)
@limiter.limit(RateLimit.API)
async def get_system(request: StarletteRequest, system_id: str):
    """获取单个 Agent 系统"""
    system = _agent_system_service.get_system(system_id)
    return {
        "id": system_id,
        "type": system["type"],
        "name": system["name"],
        "agents": system["agents"],
    }


@router.delete("/systems/{system_id}", response_model=dict)
@limiter.limit(RateLimit.API)
async def delete_system(request: StarletteRequest, system_id: str):
    """删除 Agent 系统"""
    deleted = _agent_system_service.delete_system(system_id)
    return {"deleted": deleted}


# ============== Router Agent ==============


@router.post("/router", response_model=AgentSystemResponse)
@limiter.limit(RateLimit.API)
async def create_router_agent(
    request: StarletteRequest,
    data: RouterAgentRequest,
) -> AgentSystemResponse:
    """创建路由 Agent 系统

    路由 Agent 根据用户意图将请求路由到不同的子 Agent。
    """
    success = False
    try:
        logger.info(
            "router_agent_create_start",
            name=data.name,
            agent_count=len(data.agents),
        )
        llm_service = get_llm_service()

        # 构建子 Agent 字典
        agents = {}
        for config in data.agents:
            agents[config.name] = create_api_agent_node(config)

        # 创建 Router Agent
        router_agent = RouterAgent(
            llm_service=llm_service,
            agents=agents,
            router_prompt=data.router_prompt,
        )

        graph = router_agent.compile()

        # 存储系统
        system_id = _agent_system_service.store_system(
            system_type="router",
            name=data.name,
            graph=graph,
            agents=list(agents.keys()),
        )

        success = True
        return AgentSystemResponse(
            name=data.name,
            type="router",
            agents=list(agents.keys()),
            session_id=system_id,
        )

    except Exception as e:
        logger.exception("create_router_agent_failed", name=data.name)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        logger.info(
            "router_agent_create_complete",
            name=data.name,
            success=success,
        )


@router.post("/router/{system_id}/chat", response_model=ChatResponse)
@limiter.limit(RateLimit.API)
async def router_chat(
    request: StarletteRequest,
    system_id: str,
    data: ChatRequest,
    tenant_id: TenantIdDep = None,
) -> ChatResponse:
    """使用路由 Agent 进行对话"""
    try:
        state_user_id = getattr(request.state, "user_id", None)
        effective_user_id = resolve_effective_user_id(data.user_id, state_user_id)

        await _validate_session_access(data.session_id, effective_user_id, tenant_id)

        content, session_id = await _agent_system_service.execute_chat(
            system_id=system_id,
            message=data.message,
            session_id=data.session_id,
            user_id=effective_user_id,
            tenant_id=tenant_id,
        )

        return ChatResponse(
            content=content,
            session_id=session_id,
            agent_name=system_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("router_chat_failed", system_id=system_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============== Supervisor Agent ==============


@router.post("/supervisor", response_model=AgentSystemResponse)
@limiter.limit(RateLimit.API)
async def create_supervisor_agent(
    request: StarletteRequest,
    data: SupervisorAgentRequest,
) -> AgentSystemResponse:
    """创建监督 Agent 系统

    监督 Agent 管理多个 Worker Agent，协调它们完成任务。
    """
    success = False
    try:
        logger.info(
            "supervisor_agent_create_start",
            name=data.name,
            worker_count=len(data.workers),
        )
        llm_service = get_llm_service()

        # 构建 Worker Agent 字典
        workers = {}
        for config in data.workers:
            workers[config.name] = create_api_agent_node(config)

        # 创建 Supervisor Agent
        supervisor = SupervisorAgent(
            llm_service=llm_service,
            workers=workers,
            supervisor_prompt=data.supervisor_prompt,
        )

        graph = supervisor.compile()

        # 存储系统
        system_id = _agent_system_service.store_system(
            system_type="supervisor",
            name=data.name,
            graph=graph,
            agents=list(workers.keys()),
        )

        success = True
        return AgentSystemResponse(
            name=data.name,
            type="supervisor",
            agents=list(workers.keys()),
            session_id=system_id,
        )

    except Exception as e:
        logger.exception("create_supervisor_agent_failed", name=data.name)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        logger.info(
            "supervisor_agent_create_complete",
            name=data.name,
            success=success,
        )


@router.post("/supervisor/{system_id}/chat", response_model=ChatResponse)
@limiter.limit(RateLimit.API)
async def supervisor_chat(
    request: StarletteRequest,
    system_id: str,
    data: ChatRequest,
    tenant_id: TenantIdDep = None,
) -> ChatResponse:
    """使用监督 Agent 进行对话"""
    try:
        state_user_id = getattr(request.state, "user_id", None)
        effective_user_id = resolve_effective_user_id(data.user_id, state_user_id)

        await _validate_session_access(data.session_id, effective_user_id, tenant_id)

        content, session_id = await _agent_system_service.execute_chat(
            system_id=system_id,
            message=data.message,
            session_id=data.session_id,
            user_id=effective_user_id,
            tenant_id=tenant_id,
        )

        return ChatResponse(
            content=content,
            session_id=session_id,
            agent_name=system_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("supervisor_chat_failed", system_id=system_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============== Swarm Agent ==============


@router.post("/swarm", response_model=AgentSystemResponse)
@limiter.limit(RateLimit.API)
async def create_swarm_agent(
    request: StarletteRequest,
    data: SwarmAgentRequest,
) -> AgentSystemResponse:
    """创建 Swarm Agent 系统

    Swarm Agent 支持多个 Agent 之间动态切换控制权。
    """
    success = False
    try:
        logger.info(
            "swarm_agent_create_start",
            name=data.name,
            agent_count=len(data.agents),
        )
        # 构建 HandoffAgent 列表
        agents = []
        agent_names = []

        for config in data.agents:
            handoff_targets = data.handoff_mapping.get(config.name, [])
            agent = create_api_handoff_agent(config, handoff_targets)
            agents.append(agent)
            agent_names.append(config.name)

        # 创建 Swarm
        graph = create_swarm(agents=agents, default_agent=data.default_agent)

        # 存储系统
        system_id = _agent_system_service.store_system(
            system_type="swarm",
            name=data.name,
            graph=graph,
            agents=agent_names,
        )

        success = True
        return AgentSystemResponse(
            name=data.name,
            type="swarm",
            agents=agent_names,
            session_id=system_id,
        )

    except Exception as e:
        logger.exception("create_swarm_agent_failed", name=data.name)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        logger.info(
            "swarm_agent_create_complete",
            name=data.name,
            success=success,
        )


@router.post("/swarm/{system_id}/chat", response_model=ChatResponse)
@limiter.limit(RateLimit.API)
async def swarm_chat(
    request: StarletteRequest,
    system_id: str,
    data: ChatRequest,
    tenant_id: TenantIdDep = None,
) -> ChatResponse:
    """使用 Swarm Agent 进行对话"""
    try:
        state_user_id = getattr(request.state, "user_id", None)
        effective_user_id = resolve_effective_user_id(data.user_id, state_user_id)

        await _validate_session_access(data.session_id, effective_user_id, tenant_id)

        content, session_id = await _agent_system_service.execute_chat(
            system_id=system_id,
            message=data.message,
            session_id=data.session_id,
            user_id=effective_user_id,
            tenant_id=tenant_id,
        )

        return ChatResponse(
            content=content,
            session_id=session_id,
            agent_name=system_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("swarm_chat_failed", system_id=system_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============== 单 Agent CRUD ==============


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
        AgentPublic(
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
        ExecutionItem(
            id=exc.id,
            thread_id=exc.thread_id,
            agent_id=exc.agent_id,
            status=exc.status,
            tokens_used=exc.tokens_used or 0,
            duration_ms=exc.duration_ms or 0,
            created_at=exc.created_at.isoformat(),
        )
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
async def get_agent_endpoint(
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
async def create_agent_endpoint(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    data: AgentRequest,
    *,
    tenant_id: TenantIdDep = None,
) -> AgentDetailResponse:
    """创建 Agent"""
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
async def update_agent_endpoint(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    agent_id: int,
    *,
    tenant_id: TenantIdDep = None,
    data: AgentRequest,
) -> AgentDetailResponse:
    """更新 Agent"""
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
async def delete_agent_endpoint(
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


__all__ = [
    "router",
    "AgentConfig",
    "RouterAgentRequest",
    "SupervisorAgentRequest",
    "SwarmAgentRequest",
    "ChatRequest",
    "ChatResponse",
    "ChatHistoryResponse",
    "AgentMessage",
    "AgentSystemResponse",
    "get_agent_system",
    "list_agent_systems",
    "delete_agent_system",
]
