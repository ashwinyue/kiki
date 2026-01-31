"""Supervisor Agent API 路由

提供监督 Agent 的创建和交互接口。
"""

from fastapi import APIRouter, HTTPException
from starlette.requests import Request as StarletteRequest

from app.agent import get_agent
from app.agent.multi_agent import SupervisorAgent
from app.api.v1.agents.common import (
    create_agent_node,
    create_input_state,
    create_run_config,
    extract_ai_content,
    get_agent_system,
    store_agent_system,
    validate_session_access,
)
from app.api.v1.agents.schemas import (
    AgentSystemResponse,
    ChatRequest,
    ChatResponse,
    SupervisorAgentRequest,
)
from app.core.limiter import RateLimit, limiter
from app.core.tenant_middleware import TenantIdDep
from app.llm import get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/supervisor", tags=["supervisor-agents"])


@router.post("", response_model=AgentSystemResponse)
@limiter.limit(RateLimit.API)
async def create_supervisor_agent(
    request: StarletteRequest,
    data: SupervisorAgentRequest,
) -> AgentSystemResponse:
    """创建监督 Agent 系统

    监督 Agent 管理多个 Worker Agent，协调它们完成任务。

    Args:
        request: HTTP 请求
        data: 监督 Agent 配置

    Returns:
        AgentSystemResponse: 创建的系统信息
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
            workers[config.name] = create_agent_node(config)

        # 创建 Supervisor Agent
        supervisor = SupervisorAgent(
            llm_service=llm_service,
            workers=workers,
            supervisor_prompt=data.supervisor_prompt,
        )

        graph = supervisor.compile()

        # 存储系统
        system_id = store_agent_system(
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


@router.post("/{system_id}/chat", response_model=ChatResponse)
@limiter.limit(RateLimit.API)
async def supervisor_chat(
    request: StarletteRequest,
    system_id: str,
    data: ChatRequest,
    tenant_id: TenantIdDep = None,
) -> ChatResponse:
    """使用监督 Agent 进行对话

    Args:
        system_id: 系统 ID
        message: 用户消息
        session_id: 会话 ID
        user_id: 用户 ID

    Returns:
        ChatResponse: 聊天响应
    """
    try:
        effective_user_id = data.user_id
        state_user_id = getattr(request.state, "user_id", None)
        if (
            state_user_id is not None
            and effective_user_id is not None
            and str(state_user_id) != str(effective_user_id)
        ):
            raise HTTPException(status_code=403, detail="User mismatch")
        if state_user_id is not None and effective_user_id is None:
            effective_user_id = str(state_user_id)

        await validate_session_access(data.session_id, effective_user_id, tenant_id)

        system = get_agent_system(system_id)
        graph = system["graph"]

        input_data = create_input_state(data.message, effective_user_id)
        config = create_run_config(
            data.session_id,
            effective_user_id,
            tenant_id=tenant_id,
        )

        result = await graph.ainvoke(input_data, config)

        content = extract_ai_content(result.get("messages", []))

        agent = await get_agent()
        await agent.persist_interaction(data.session_id, data.message, content)

        return ChatResponse(
            content=content,
            session_id=data.session_id,
            agent_name=system_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("supervisor_chat_failed", system_id=system_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
