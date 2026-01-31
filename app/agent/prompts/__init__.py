"""Prompt 模板管理模块

提供统一的 Prompt 模板注册、获取和管理功能。

使用示例:
```python
from app.agent.prompts import get_prompt, register_prompt

# 获取内置 Prompt
router_prompt = get_prompt("router")

# 注册自定义 Prompt
register_prompt("custom", "你是一个自定义助手...")
```
"""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agent.prompts.chat import (
    CHAT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT_WITH_TOOLS,
)
from app.agent.prompts.router import ROUTER_SYSTEM_PROMPT
from app.agent.prompts.supervisor import (
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT_WITH_CONTEXT,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


# Prompt 注册表
_PROMPT_REGISTRY: dict[str, str | ChatPromptTemplate] = {
    "chat": CHAT_SYSTEM_PROMPT,
    "chat_with_tools": CHAT_SYSTEM_PROMPT_WITH_TOOLS,
    "router": ROUTER_SYSTEM_PROMPT,
    "supervisor": SUPERVISOR_SYSTEM_PROMPT,
    "supervisor_with_context": SUPERVISOR_SYSTEM_PROMPT_WITH_CONTEXT,
}


def register_prompt(name: str, template: str | ChatPromptTemplate) -> None:
    """注册 Prompt 模板

    Args:
        name: Prompt 名称
        template: Prompt 模板（字符串或 ChatPromptTemplate）

    Examples:
        ```python
        from app.agent.prompts import register_prompt

        register_prompt(
            "custom_assistant",
            "你是一个专业的助手，专门处理 X、Y、Z 问题。"
        )
        ```
    """
    _PROMPT_REGISTRY[name] = template
    logger.info("prompt_registered", name=name, type=type(template).__name__)


def get_prompt(
    name: str,
    default: str | ChatPromptTemplate | None = None,
    **variables: Any,
) -> str | ChatPromptTemplate:
    """获取 Prompt 模板

    Args:
        name: Prompt 名称
        default: 默认值（如果 Prompt 不存在）
        **variables: 模板变量（用于格式化字符串模板）

    Returns:
        Prompt 模板

    Raises:
        KeyError: 如果 Prompt 不存在且未提供默认值

    Examples:
        ```python
        from app.agent.prompts import get_prompt

        # 获取原始 Prompt
        prompt = get_prompt("router")

        # 获取并格式化 Prompt
        prompt = get_prompt("router", agents="Sales, Support")
        ```
    """
    if name not in _PROMPT_REGISTRY:
        if default is not None:
            logger.warning("prompt_not_found_using_default", name=name)
            return default
        raise KeyError(f"Prompt '{name}' 未注册")

    template = _PROMPT_REGISTRY[name]

    # 如果是字符串且提供了变量，进行格式化
    if isinstance(template, str) and variables:
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning("prompt_format_failed", missing=str(e))

    return template


def list_prompts() -> list[str]:
    """列出所有已注册的 Prompt 名称

    Returns:
        Prompt 名称列表
    """
    return list(_PROMPT_REGISTRY.keys())


def create_chat_prompt(
    system_prompt: str | None = None,
    with_tools: bool = False,
) -> ChatPromptTemplate:
    """创建聊天 Prompt 模板

    Args:
        system_prompt: 自定义系统提示词
        with_tools: 是否包含工具说明

    Returns:
        ChatPromptTemplate 实例
    """
    if system_prompt is None:
        system_prompt = get_prompt("chat_with_tools" if with_tools else "chat")

    messages = [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
    return ChatPromptTemplate.from_messages(messages)


def create_router_prompt(
    agent_list: str | list[str] | None = None,
    custom_prompt: str | None = None,
) -> str:
    """创建路由 Agent Prompt

    Args:
        agent_list: 可用的 Agent 列表
        custom_prompt: 自定义 Prompt

    Returns:
        Prompt 字符串
    """
    if custom_prompt:
        return custom_prompt

    base_prompt = get_prompt("router")

    if agent_list:
        if isinstance(agent_list, list):
            agent_list = ", ".join(agent_list)
        return base_prompt.replace("{agents}", agent_list)

    return base_prompt


def create_supervisor_prompt(
    worker_list: str | list[str] | None = None,
    custom_prompt: str | None = None,
    with_context: bool = True,
) -> str:
    """创建监督 Agent Prompt

    Args:
        worker_list: 可用的 Worker 列表
        custom_prompt: 自定义 Prompt
        with_context: 是否包含上下文说明

    Returns:
        Prompt 字符串
    """
    if custom_prompt:
        return custom_prompt

    prompt_key = "supervisor_with_context" if with_context else "supervisor"
    base_prompt = get_prompt(prompt_key)

    if worker_list:
        if isinstance(worker_list, list):
            worker_list = ", ".join(worker_list)
        base_prompt = base_prompt.replace("{workers}", worker_list)

    return base_prompt


__all__ = [
    # 注册函数（旧版，兼容性保留）
    "register_prompt",
    "get_prompt",
    "list_prompts",
    # 创建函数
    "create_chat_prompt",
    "create_router_prompt",
    "create_supervisor_prompt",
    # 内置 Prompt
    "CHAT_SYSTEM_PROMPT",
    "CHAT_SYSTEM_PROMPT_WITH_TOOLS",
    "ROUTER_SYSTEM_PROMPT",
    "SUPERVISOR_SYSTEM_PROMPT",
    "SUPERVISOR_SYSTEM_PROMPT_WITH_CONTEXT",
    # Jinja2 模板系统（新版）
    "render_prompt",
    "render_template_string",
    "get_template",
    "register_template",
    "register_template_file",
    "list_templates",
    "delete_template",
    "create_langchain_prompt",
    "create_structured_prompt",
    "load_templates_from_dir",
    "export_templates_to_dir",
    "render",  # render_prompt 的简写别名
]

# 导出 Jinja2 模板系统
from app.agent.prompts.template import (  # noqa: E402, F401
    create_langchain_prompt,
    create_structured_prompt,
    delete_template,
    export_templates_to_dir,
    get_template,
    list_templates,
    load_templates_from_dir,
    register_template,
    register_template_file,
    render_prompt,
    render_template_string,
)

# 简化导出
render = render_prompt  # type: ignore
