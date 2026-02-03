"""Agent 配置文件

参考 DeerFlow 设计，集中定义所有 Agent 配置。
每个 Agent 包含：
- agent_type: 决定使用的 LLM 类型
- prompt_template: 提示词模板名称
- tools: 工具列表（延迟加载）

使用方式：
    ```python
    from app.agent.graph.agents import AGENT_REGISTRY, get_agent_config

    # 获取所有 agent 配置
    for agent_id, config in AGENT_REGISTRY.items():
        print(f"{agent_id}: {config['description']}")

    # 创建 agent
    from app.agent.graph.agent_factory import create_agent
    config = get_agent_config("planner")
    planner = create_agent(
        agent_name="planner",
        agent_type=config["agent_type"],
        tools=config["tools"](),
        prompt_template=config["prompt_template"],
    )
    ```
"""

from typing import Any

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ========== 工具加载函数 ==========

def get_research_tools():
    """获取研究工具"""
    from app.agent.tools import get_tools as get_all_tools
    tools = get_all_tools()
    # 过滤出搜索相关工具
    return [t for t in tools if "search" in t.name.lower() or "tavily" in t.name.lower()]


def get_code_tools():
    """获取代码工具"""
    from app.agent.tools import get_tools as get_all_tools
    tools = get_all_tools()
    # 过滤出代码相关工具
    return [t for t in tools if "python" in t.name.lower() or "repl" in t.name.lower()]


def get_chat_tools():
    """获取聊天工具"""
    from app.agent.tools import get_tools as get_all_tools
    return get_all_tools()


# ========== Agent 配置注册表 ==========

AGENT_REGISTRY: dict[str, dict[str, Any]] = {
    # ========== 专门化角色（DeerFlow 风格）==========
    "coordinator": {
        "agent_type": "coordinator",
        "prompt_template": "chat",
        "description": "协调者 - 处理入口任务",
        "tools": lambda: [],
    },
    "planner": {
        "agent_type": "planner",
        "prompt_template": "planner",
        "description": "规划者 - 任务分解和计划生成",
        "tools": lambda: [],
    },
    "researcher": {
        "agent_type": "researcher",
        "prompt_template": "researcher",
        "description": "研究员 - 信息检索和验证",
        "tools": lambda: get_research_tools(),
    },
    "analyst": {
        "agent_type": "analyst",
        "prompt_template": "analyst",
        "description": "分析师 - 数据分析和推理",
        "tools": lambda: [],
    },
    "coder": {
        "agent_type": "coder",
        "prompt_template": "coder",
        "description": "代码专家 - 代码生成和优化",
        "tools": lambda: get_code_tools(),
    },
    "reporter": {
        "agent_type": "reporter",
        "prompt_template": "reporter",
        "description": "报告员 - 聚合结果生成报告",
        "tools": lambda: [],
    },
    # ========== Supervisor 模式 ==========
    "supervisor": {
        "agent_type": "supervisor",
        "prompt_template": "supervisor",
        "description": "监督者 - 协调多个 Worker Agent",
        "tools": lambda: [],
    },
    "router": {
        "agent_type": "router",
        "prompt_template": "router",
        "description": "路由器 - 根据意图分发请求",
        "tools": lambda: [],
    },
    # ========== 通用角色 ==========
    "chat": {
        "agent_type": "chat",
        "prompt_template": "chat",
        "description": "通用对话助手",
        "tools": lambda: get_chat_tools(),
    },
    "worker": {
        "agent_type": "worker",
        "prompt_template": "chat",
        "description": "通用 Worker",
        "tools": lambda: [],
    },
}


# ========== 便捷函数 ==========

def get_agent_config(agent_id: str) -> dict[str, Any]:
    """获取 Agent 配置

    Args:
        agent_id: Agent ID

    Returns:
        Agent 配置字典

    Raises:
        KeyError: 如果 Agent 不存在
    """
    if agent_id not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(
            f"Agent '{agent_id}' 不存在。可用: {available}"
        )
    return AGENT_REGISTRY[agent_id]


def list_agents() -> list[str]:
    """列出所有已注册的 Agent

    Returns:
        Agent ID 列表
    """
    return list(AGENT_REGISTRY.keys())


def list_agents_by_type(agent_type: str) -> list[str]:
    """按类型列出 Agent

    Args:
        agent_type: Agent 类型

    Returns:
        Agent ID 列表
    """
    return [
        agent_id
        for agent_id, config in AGENT_REGISTRY.items()
        if config["agent_type"] == agent_type
    ]


# ========== DeerFlow 风格的预配置 Agent 组 ==========

# 研究团队（DeerFlow 深度研究架构）
RESEARCH_TEAM = ["planner", "researcher", "analyst", "coder", "reporter"]

# 通用对话团队
CHAT_TEAM = ["chat"]

# Supervisor 模式团队
SUPERVISOR_TEAM = ["supervisor", "researcher", "coder"]


__all__ = [
    "AGENT_REGISTRY",
    "get_agent_config",
    "list_agents",
    "list_agents_by_type",
    "RESEARCH_TEAM",
    "CHAT_TEAM",
    "SUPERVISOR_TEAM",
]
