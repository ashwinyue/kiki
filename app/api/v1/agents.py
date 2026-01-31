"""Agent 管理 API

提供单个 Agent 的 CRUD 操作和复制功能。
对齐 WeKnora99 CustomAgent 模型和 Copy API。
"""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request as StarletteRequest
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.repositories.agent_async import AgentRepositoryAsync
from app.repositories.base import PaginationParams
from app.schemas.agent import (
    AgentCopyRequest,
    AgentCopyResponse,
    AgentDetailResponse,
    AgentListResponse,
    AgentPublic,
    AgentRequest,
    BatchAgentCopyRequest,
    BatchAgentCopyResponse,
    PlaceholderCreate,
    PlaceholderListResponse,
    PlaceholderPreviewRequest,
    PlaceholderPreviewResponse,
    PlaceholderSchema,
    PlaceholderUpdate,
)
from app.services.agent_clone import AgentCloner
from app.services.placeholder_service import PlaceholderService

router = APIRouter(prefix="/agents", tags=["agents"])
logger = get_logger(__name__)


# ============== 单 Agent CRUD ==============


async def get_agent_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AgentRepositoryAsync:
    """获取 Agent 仓储"""
    return AgentRepositoryAsync(session)


@router.get(
    "/list",
    response_model=AgentListResponse,
    summary="列出所有 Agent",
    description="获取所有 Agent 的列表",
)
@limiter.limit(RateLimit.API)
async def list_agents(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    *,
    tenant_id: TenantIdDep = None,
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
) -> AgentListResponse:
    """列出所有 Agent"""
    params = PaginationParams(page=page, size=size)
    result = await repository.list_by_tenant(tenant_id, params)

    # 转换为公开模型
    agents = [
        AgentPublic(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            agent_type="custom",
            status="active" if agent.deleted_at is None else "deleted",
            model_name="",
            system_prompt="",
            temperature=0.0,
            max_tokens=0,
            config=agent.config or {},
            created_at=agent.created_at.isoformat() if agent.created_at else "",
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
    "/{agent_id}",
    response_model=AgentDetailResponse,
    summary="获取 Agent 详情",
    description="根据 ID 获取 Agent 的详细信息",
)
@limiter.limit(RateLimit.API)
async def get_agent_endpoint(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    agent_id: str,
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
        id=str(agent.id),
        name=agent.name,
        description=agent.description,
        agent_type="custom",
        status="active" if agent.deleted_at is None else "deleted",
        model_name="",
        system_prompt="",
        temperature=0.0,
        max_tokens=0,
        config=agent.config or {},
        created_at=agent.created_at.isoformat() if agent.created_at else "",
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
    if tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id 必填",
        )

    # 构建数据
    agent_data = {
        "id": str(uuid4()),
        "name": data.name,
        "description": data.description,
        "tenant_id": tenant_id,
        "created_by": str(getattr(request.state, "user_id", "")) or None,
        "config": data.config,
    }

    agent = await repository.create_with_tools(agent_data)

    return AgentDetailResponse(
        id=str(agent.id),
        name=agent.name,
        description=agent.description,
        agent_type="custom",
        status="active" if agent.deleted_at is None else "deleted",
        model_name="",
        system_prompt="",
        temperature=0.0,
        max_tokens=0,
        config=agent.config or {},
        created_at=agent.created_at.isoformat() if agent.created_at else "",
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
    agent_id: str,
    *,
    tenant_id: TenantIdDep = None,
    data: AgentRequest,
) -> AgentDetailResponse:
    """更新 Agent"""
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

    # 构建更新数据
    update_data = {
        "name": data.name,
        "description": data.description,
        "config": data.config,
    }

    agent = await repository.update_agent(agent_id, update_data)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} 不存在",
        )

    return AgentDetailResponse(
        id=str(agent.id),
        name=agent.name,
        description=agent.description,
        agent_type="custom",
        status="active" if agent.deleted_at is None else "deleted",
        model_name="",
        system_prompt="",
        temperature=0.0,
        max_tokens=0,
        config=agent.config or {},
        created_at=agent.created_at.isoformat() if agent.created_at else "",
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
    agent_id: str,
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


# ============== Agent 复制 ==============


@router.post(
    "/{agent_id}/copy",
    response_model=AgentCopyResponse,
    summary="复制 Agent",
    description="复制指定 Agent，可选择是否复制配置、工具、知识库等",
)
@limiter.limit(RateLimit.API)
async def copy_agent_endpoint(
    request: StarletteRequest,
    repository: Annotated[AgentRepositoryAsync, Depends(get_agent_repository)],
    session: Annotated[AsyncSession, Depends(get_session)],
    agent_id: str,
    *,
    tenant_id: TenantIdDep = None,
    data: AgentCopyRequest,
) -> AgentCopyResponse:
    """复制 Agent

    对齐 WeKnora 的 POST /agents/:id/copy API。

    复制选项：
    - copy_config: 是否复制配置（默认 true）
    - copy_tools: 是否复制工具关联（默认 true）
    - copy_knowledge: 是否复制知识库关联（默认 false）
    """
    # 验证源 Agent 存在
    source_agent = await repository.get(agent_id)
    if source_agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"源 Agent {agent_id} 不存在",
        )

    # 租户权限检查
    if tenant_id is not None and source_agent.tenant_id is not None and source_agent.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权复制此 Agent",
        )

    # 获取创建人
    created_by = str(getattr(request.state, "user_id", "")) or None

    # 执行复制
    cloner = AgentCloner(session)
    try:
        result = await cloner.copy_agent(agent_id, data, created_by)
        logger.info(
            "agent_copy_success",
            source_agent_id=agent_id,
            new_agent_id=result.new_agent_id,
            name=result.name,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("agent_copy_failed", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"复制 Agent 失败: {str(e)}",
        )


@router.post(
    "/batch/copy",
    response_model=BatchAgentCopyResponse,
    summary="批量复制 Agent",
    description="批量复制多个 Agent",
)
@limiter.limit(RateLimit.API)
async def batch_copy_agents_endpoint(
    request: StarletteRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    *,
    tenant_id: TenantIdDep = None,
    data: BatchAgentCopyRequest,
) -> BatchAgentCopyResponse:
    """批量复制 Agent

    对齐 WeKnora 的批量复制功能。
    """
    # 获取创建人
    created_by = str(getattr(request.state, "user_id", "")) or None

    # 执行批量复制
    cloner = AgentCloner(session)
    try:
        result = await cloner.batch_copy(
            agent_ids=data.agent_ids,
            request=data,
            tenant_id=tenant_id,
            created_by=created_by,
        )
        logger.info(
            "batch_agent_copy_complete",
            success_count=result.success_count,
            failed_count=result.failed_count,
        )
        return result
    except Exception as e:
        logger.error("batch_agent_copy_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量复制失败: {str(e)}",
        )


__all__ = [
    "router",
    "AgentRequest",
    "AgentCopyRequest",
    "AgentCopyResponse",
    "BatchAgentCopyRequest",
    "BatchAgentCopyResponse",
]


# ============== 占位符管理 ==============


async def get_placeholder_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PlaceholderService:
    """获取占位符服务"""
    return PlaceholderService(session)


@router.get(
    "/placeholders",
    response_model=PlaceholderListResponse,
    summary="获取所有占位符",
    description="获取当前租户的所有占位符定义（包括全局占位符）",
)
@limiter.limit(RateLimit.API)
async def list_placeholders(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    *,
    tenant_id: TenantIdDep = None,
    agent_id: str | None = Query(None, description="过滤 Agent ID"),
    category: str | None = Query(None, description="过滤分类"),
    is_enabled: bool | None = Query(None, description="过滤是否启用"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
) -> PlaceholderListResponse:
    """列出所有占位符"""
    if tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id 必填",
        )

    items, total = await service.list_placeholders(
        tenant_id,
        agent_id=agent_id,
        category=category,
        is_enabled=is_enabled,
        page=page,
        size=size,
    )

    return PlaceholderListResponse(items=items, total=total)


@router.get(
    "/placeholders/{placeholder_id}",
    response_model=PlaceholderSchema,
    summary="获取占位符详情",
    description="根据 ID 获取占位符的详细信息",
)
@limiter.limit(RateLimit.API)
async def get_placeholder(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    placeholder_id: str,
    tenant_id: TenantIdDep = None,
) -> PlaceholderSchema:
    """获取占位符详情"""
    placeholder = await service.get_placeholder(placeholder_id)
    if placeholder is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"占位符 {placeholder_id} 不存在",
        )
    return placeholder


@router.post(
    "/placeholders",
    response_model=PlaceholderSchema,
    status_code=status.HTTP_201_CREATED,
    summary="创建占位符",
    description="创建一个新的占位符定义",
)
@limiter.limit(RateLimit.API)
async def create_placeholder(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    data: PlaceholderCreate,
    *,
    tenant_id: TenantIdDep = None,
) -> PlaceholderSchema:
    """创建占位符"""
    if tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id 必填",
        )

    created_by = str(getattr(request.state, "user_id", "")) or None
    placeholder = await service.create_placeholder(data, tenant_id, created_by)

    logger.info(
        "placeholder_created",
        placeholder_id=placeholder.id,
        name=placeholder.name,
        agent_id=placeholder.agent_id,
    )
    return placeholder


@router.put(
    "/placeholders/{placeholder_id}",
    response_model=PlaceholderSchema,
    summary="更新占位符",
    description="更新占位符的配置",
)
@limiter.limit(RateLimit.API)
async def update_placeholder(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    placeholder_id: str,
    *,
    tenant_id: TenantIdDep = None,
    data: PlaceholderUpdate,
) -> PlaceholderSchema:
    """更新占位符"""
    placeholder = await service.update_placeholder(placeholder_id, data, tenant_id)
    if placeholder is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"占位符 {placeholder_id} 不存在",
        )

    logger.info("placeholder_updated", placeholder_id=placeholder_id)
    return placeholder


@router.delete(
    "/placeholders/{placeholder_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除占位符",
    description="软删除指定的占位符",
)
@limiter.limit(RateLimit.API)
async def delete_placeholder(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    placeholder_id: str,
    *,
    tenant_id: TenantIdDep = None,
) -> None:
    """删除占位符"""
    success = await service.delete_placeholder(placeholder_id, tenant_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"占位符 {placeholder_id} 不存在",
        )

    logger.info("placeholder_deleted", placeholder_id=placeholder_id)


@router.get(
    "/{agent_id}/placeholders",
    response_model=list[PlaceholderSchema],
    summary="获取 Agent 的占位符",
    description="获取指定 Agent 的所有占位符（包括全局占位符）",
)
@limiter.limit(RateLimit.API)
async def get_agent_placeholders(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    agent_id: str,
    *,
    tenant_id: TenantIdDep = None,
) -> list[PlaceholderSchema]:
    """获取 Agent 的占位符"""
    if tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id 必填",
        )

    return await service.get_placeholders_for_agent(agent_id, tenant_id)


@router.post(
    "/preview",
    response_model=PlaceholderPreviewResponse,
    summary="预览占位符替换",
    description="预览模板中的占位符替换结果",
)
@limiter.limit(RateLimit.API)
async def preview_placeholders(
    request: StarletteRequest,
    service: Annotated[PlaceholderService, Depends(get_placeholder_service)],
    data: PlaceholderPreviewRequest,
    *,
    tenant_id: TenantIdDep = None,
) -> PlaceholderPreviewResponse:
    """预览占位符替换

    对齐 WeKnora 的 POST /agents/preview API。

    支持两种占位符格式：
    - {{ variable_name }} - Jinja2 格式
    - ${variable_name} - 简单格式
    """
    result = await service.preview(data, tenant_id)
    return result
