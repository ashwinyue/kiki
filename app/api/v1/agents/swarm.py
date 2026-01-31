"""Swarm Agent API 路由

提供 Swarm (Handoff) Agent 的创建和交互接口。
"""

from fastapi import APIRouter, HTTPException
from starlette.requests import Request as StarletteRequest

from app.agent import get_agent
from app.agent.multi_agent import HandoffAgent, create_swarm
from app.agent.tools import list_tools
from app.api.v1.agents.common import (
    create_input_state,
    create_run_config,
    extract_ai_content,
    get_agent_system,
    store_agent_system,
    validate_session_access,
)
from app.api.v1.agents.schemas import (
    AgentConfig,
    AgentSystemResponse,
    ChatRequest,
    ChatResponse,
    SwarmAgentRequest,
)
from app.core.limiter import RateLimit, limiter
from app.core.tenant_middleware import TenantIdDep
from app.llm import get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/swarm", tags=["swarm-agents"])


def _create_handoff_agent(config: AgentConfig, handoff_targets: list[str]) -> HandoffAgent:
    """创建可切换的 Handoff Agent

    Args:
        config: Agent 配置
        handoff_targets: 可切换的目标列表

    Returns:
        HandoffAgent 实例
    """
    llm_service = get_llm_service()

    # 获取指定的工具
    all_tools = {t.name: t for t in list_tools()}
    agent_tools = [all_tools[name] for name in config.tools if name in all_tools]

    return HandoffAgent(
        name=config.name,
        llm_service=llm_service,
        tools=agent_tools,
        handoff_targets=handoff_targets,
        system_prompt=config.system_prompt or None,
    )


@router.post("", response_model=AgentSystemResponse)
@limiter.limit(RateLimit.API)
async def create_swarm_agent(
    request: StarletteRequest,
    data: SwarmAgentRequest,
) -> AgentSystemResponse:
    """创建 Swarm Agent 系统

    Swarm Agent 支持多个 Agent 之间动态切换控制权。

    Args:
        request: HTTP 请求
        data: Swarm Agent 配置

    Returns:
        AgentSystemResponse: 创建的系统信息
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
            agent = _create_handoff_agent(config, handoff_targets)
            agents.append(agent)
            agent_names.append(config.name)

        # 创建 Swarm
        graph = create_swarm(agents=agents, default_agent=data.default_agent)

        # 存储系统
        system_id = store_agent_system(
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


@router.post("/{system_id}/chat", response_model=ChatResponse)
@limiter.limit(RateLimit.API)
async def swarm_chat(
    request: StarletteRequest,
    system_id: str,
    data: ChatRequest,
    tenant_id: TenantIdDep = None,
) -> ChatResponse:
    """使用 Swarm Agent 进行对话

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
        logger.exception("swarm_chat_failed", system_id=system_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
