"""多 Agent API 通用函数和系统管理"""

from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import RunnableConfig

from app.agent.graphs import create_react_agent
from app.agent.state import AgentState
from app.agent.tools import list_tools
from app.api.v1.agents.schemas import AgentConfig
from app.config.settings import get_settings
from app.llm import get_llm_service
from app.observability.logging import get_logger
from app.infra.database import session_repository, session_scope

logger = get_logger(__name__)

# 存储已创建的 Agent 系统
_agent_systems: dict[str, dict[str, Any]] = {}


def create_agent_node(agent_config: AgentConfig) -> callable:
    """根据配置创建 Agent 节点函数

    Args:
        agent_config: Agent 配置

    Returns:
        Agent 节点函数
    """
    llm_service = get_llm_service()

    # 获取指定的工具
    all_tools = {t.name: t for t in list_tools()}
    agent_tools = [all_tools[name] for name in agent_config.tools if name in all_tools]

    # 使用 create_react_agent 创建 Agent
    react_agent = create_react_agent(
        llm_service=llm_service,
        tools=agent_tools,
        system_prompt=agent_config.system_prompt or None,
    )

    async def agent_node(state: AgentState, config) -> dict:
        """Agent 节点包装函数"""
        # 获取最后一条用户消息
        user_message = ""
        for msg in reversed(state["messages"]):
            if msg.type == "human":
                user_message = msg.content
                break

        # 调用 Agent
        messages = await react_agent.get_response(
            message=user_message,
            session_id=config.get("configurable", {}).get("thread_id", "default"),
            user_id=state.get("user_id"),
        )

        # 获取响应
        content = ""
        for msg in reversed(messages):
            if msg.type == "ai":
                content = msg.content
                break

        return {"messages": [AIMessage(content=content)]}

    return agent_node


async def validate_session_access(
    session_id: str,
    user_id: str | None,
    tenant_id: int | None,
) -> None:
    """验证会话是否存在，并可选校验用户/租户归属"""
    async with session_scope() as session:
        repo = session_repository(session)
        session_obj = await repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None:
            try:
                user_id_int = int(user_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid user_id") from exc

            if session_obj.user_id != user_id_int:
                raise HTTPException(status_code=403, detail="Session access denied")

        if tenant_id is not None and session_obj.tenant_id is not None:
            if session_obj.tenant_id != tenant_id:
                raise HTTPException(status_code=403, detail="Session tenant mismatch")


def get_agent_system(system_key: str) -> dict[str, Any]:
    """获取 Agent 系统

    Args:
        system_key: 系统键

    Returns:
        系统配置

    Raises:
        HTTPException: 如果系统不存在
    """
    if system_key not in _agent_systems:
        raise HTTPException(status_code=404, detail=f"Agent 系统 '{system_key}' 不存在")
    return _agent_systems[system_key]


def create_input_state(message: str, user_id: str | None = None) -> dict:
    """创建输入状态

    Args:
        message: 用户消息
        user_id: 用户 ID

    Returns:
        输入状态字典
    """
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "session_id": None,
        "_next_agent": None,
        "_next_worker": None,
        "_handoff_target": None,
    }


def create_run_config(
    session_id: str,
    user_id: str | None = None,
    tenant_id: int | None = None,
) -> RunnableConfig:
    """创建运行配置

    Args:
        session_id: 会话 ID
        user_id: 用户 ID

    Returns:
        RunnableConfig
    """
    callbacks = []
    try:
        from app.agent.callbacks import KikiCallbackHandler

        settings = get_settings()
        callbacks.append(
            KikiCallbackHandler(
                session_id=session_id,
                user_id=user_id,
                enable_langfuse=settings.langfuse_enabled,
                enable_metrics=True,
            )
        )
    except Exception:
        pass

    return RunnableConfig(
        configurable={"thread_id": session_id},
        metadata={"user_id": user_id, "session_id": session_id, "tenant_id": tenant_id},
        callbacks=callbacks or None,
    )


def extract_ai_content(messages: list) -> str:
    """从消息列表中提取 AI 响应内容

    Args:
        messages: 消息列表

    Returns:
        AI 响应内容
    """
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
        elif hasattr(msg, "content"):
            return str(msg.content)
    return ""


def store_agent_system(
    system_type: str,
    name: str,
    graph,
    agents: list[str],
) -> str:
    """存储 Agent 系统

    Args:
        system_type: 系统类型
        name: 系统名称
        graph: 编译后的图
        agents: Agent 名称列表

    Returns:
        系统 ID
    """
    system_id = f"{system_type}_{name}"
    _agent_systems[system_id] = {
        "type": system_type,
        "name": name,
        "graph": graph,
        "agents": agents,
    }
    logger.info(
        f"{system_type}_agent_created",
        name=name,
        agents=agents,
    )
    return system_id


def list_agent_systems() -> list[dict[str, Any]]:
    """列出所有 Agent 系统

    Returns:
        系统列表
    """
    return [
        {
            "id": key,
            "type": value["type"],
            "name": value["name"],
            "agents": value["agents"],
        }
        for key, value in _agent_systems.items()
    ]


def delete_agent_system(system_key: str) -> bool:
    """删除 Agent 系统

    Args:
        system_key: 系统键

    Returns:
        是否成功删除
    """
    if system_key in _agent_systems:
        del _agent_systems[system_key]
        logger.info("agent_system_deleted", system_key=system_key)
        return True
    return False
