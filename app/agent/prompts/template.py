"""Prompt 模板系统

使用 Jinja2 实现多语言 Prompt 模板管理，支持变量注入和回退机制。
参考 DeerFlow 的模板设计。

依赖安装:
    uv add jinja2

使用示例:
```python
from app.agent.prompts.template import (
    render_prompt,
    register_template,
    get_template,
)

# 渲染模板（带变量）
prompt = render_prompt("chat", name="用户", locale="zh-CN")

# 注册自定义模板
register_template("custom", "你好 {{name}}！")

# 获取模板对象
template = get_template("router")
```
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateSyntaxError,
)
from langchain_core.prompts import ChatPromptTemplate

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认语言
_DEFAULT_LOCALE = "zh-CN"

# 支持的语言
_SUPPORTED_LOCALES = ["zh-CN", "en-US", "ja-JP"]

# 模板目录
_TEMPLATE_DIR = Path(__file__).parent / "templates"

# 全局 Jinja2 环境
_jinja_env: Environment | None = None


def _get_jinja_env() -> Environment:
    """获取 Jinja2 环境

    Returns:
        Jinja2 Environment 实例
    """
    global _jinja_env
    if _jinja_env is None:
        if _TEMPLATE_DIR.exists():
            loader = FileSystemLoader(_TEMPLATE_DIR)
            logger.info("using_filesystem_loader", template_dir=str(_TEMPLATE_DIR))
        else:
            loader = BaseLoader()
            logger.debug("using_base_loader", template_dir=str(_TEMPLATE_DIR))

        _jinja_env = Environment(
            loader=loader,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,  # Prompt 不需要 HTML 转义
        )

        _jinja_env.globals.update(
            {
                "now": datetime.now,
                "today": datetime.now().date,
                "env": os.getenv,
            }
        )

    return _jinja_env


_BUILTIN_TEMPLATES: dict[str, dict[str, str]] = {
    "chat": {
        "zh-CN": """你是一个专业、友好的 AI 助手，名叫 Kiki。

你的职责：
- 准确理解用户意图
- 提供有帮助的回答
- 使用工具获取最新信息时，注明信息来源

当前时间：{{ now().strftime('%Y-%m-%d %H:%M') }}
用户语言：{{ locale }}
""",
        "en-US": """You are a professional and friendly AI assistant named Kiki.

Your responsibilities:
- Accurately understand user intent
- Provide helpful responses
- When using tools to get latest information, cite the source

Current time: {{ now().strftime('%Y-%m-%d %H:%M') }}
User locale: {{ locale }}
""",
    },
    "chat_with_tools": {
        "zh-CN": """你是一个专业、友好的 AI 助手，名叫 Kiki。

你可以使用以下工具来帮助用户：
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

使用工具时请遵循以下原则：
1. 优先使用工具获取最新信息
2. 工具参数要准确完整
3. 解释工具调用的结果

当前时间：{{ now().strftime('%Y-%m-%d %H:%M') }}
""",
        "en-US": """You are a professional and friendly AI assistant named Kiki.

You have access to the following tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

When using tools, follow these principles:
1. Prioritize using tools to get latest information
2. Provide accurate and complete tool parameters
3. Explain the results of tool calls

Current time: {{ now().strftime('%Y-%m-%d %H:%M') }}
""",
    },
    "router": {
        "zh-CN": """你是一个智能路由器，负责将用户请求分配给最合适的 Agent。

可用的 Agent：
{% for agent in agents %}
- {{ agent.name }}: {{ agent.description }}
{% endfor %}

根据用户请求的内容和意图，选择最合适的 Agent 处理。
只需要返回 Agent 的名称即可。

用户请求：{{ input }}""",
        "en-US": """You are an intelligent router responsible for directing user requests to the most suitable Agent.

Available Agents:
{% for agent in agents %}
- {{ agent.name }}: {{ agent.description }}
{% endfor %}

Based on the user's request content and intent, select the most suitable Agent to handle it.
Only return the Agent's name.

User request: {{ input }}""",
    },
    "supervisor": {
        "zh-CN": """你是一个任务监督者，负责协调多个 Worker Agent 完成复杂任务。

可用的 Workers：
{% for worker in workers %}
- {{ worker }}: {{ workers[worker] }}
{% endfor %}

你的职责：
1. 分析任务需求
2. 将任务分解为子任务
3. 分配子任务给合适的 Worker
4. 汇总结果，给出最终答案

任务：{{ task }}""",
        "en-US": """You are a task supervisor responsible for coordinating multiple Worker Agents to complete complex tasks.

Available Workers:
{% for worker in workers %}
- {{ worker }}: {{ workers[worker] }}
{% endfor %}

Your responsibilities:
1. Analyze task requirements
2. Break down the task into subtasks
3. Assign subtasks to appropriate Workers
4. Summarize results and provide the final answer

Task: {{ task }}""",
    },
    "clarification": {
        "zh-CN": """为了更好地帮助你，我需要更多信息。

{{ question }}

请提供详细信息，以便我能够准确理解你的需求。

历史对话：
{% for item in history %}
- {{ item }}
{% endfor %}""",
        "en-US": """To better assist you, I need more information.

{{ question }}

Please provide details so I can accurately understand your needs.

Conversation history:
{% for item in history %}
- {{ item }}
{% endfor %}""",
    },
    "tool_error": {
        "zh-CN": """工具调用遇到错误，请尝试其他方式。

错误信息：{{ error }}

工具：{{ tool_name }}
参数：{{ tool_args }}

请分析错误原因并给出建议。""",
        "en-US": """Tool call encountered an error, please try another approach.

Error message: {{ error }}
Tool: {{ tool_name }}
Arguments: {{ tool_args }}

Please analyze the error and provide suggestions.""",
    },
}


# ============== 模板注册表 ==============

_template_registry: dict[str, dict[str, str]] = {}
# 复制内置模板
for name, locales in _BUILTIN_TEMPLATES.items():
    _template_registry[name] = locales.copy()


def register_template(
    name: str,
    template: str,
    locale: str = "zh-CN",
) -> None:
    """注册模板

    Args:
        name: 模板名称
        template: 模板内容（Jinja2 格式）
        locale: 语言代码

    Examples:
        ```python
        register_template("greeting", "你好，{{ name }}！", locale="zh-CN")
        register_template("greeting", "Hello, {{ name }}!", locale="en-US")
        ```
    """
    if name not in _template_registry:
        _template_registry[name] = {}

    _template_registry[name][locale] = template
    logger.info("template_registered", name=name, locale=locale)


def register_template_file(
    name: str,
    file_path: str | Path,
    locale: str = "zh-CN",
) -> None:
    """从文件注册模板

    Args:
        name: 模板名称
        file_path: 模板文件路径
        locale: 语言代码

    Examples:
        ```python
        register_template_file("custom", "/path/to/custom.jinja2")
        ```
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("template_file_not_found", path=str(file_path))
        raise FileNotFoundError(f"模板文件不存在: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    register_template(name, content, locale)
    logger.info("template_file_registered", name=name, path=str(file_path))


def get_template(name: str, locale: str = "zh-CN") -> str:
    """获取模板内容

    Args:
        name: 模板名称
        locale: 语言代码

    Returns:
        模板内容字符串

    Raises:
        KeyError: 如果模板不存在
    """
    if name not in _template_registry:
        raise KeyError(f"模板 '{name}' 不存在")

    locales = _template_registry[name]

    # 尝试获取指定语言
    if locale in locales:
        return locales[locale]

    # 回退到默认语言
    if _DEFAULT_LOCALE in locales:
        logger.warning(
            "template_locale_fallback",
            name=name,
            requested=locale,
            fallback=_DEFAULT_LOCALE,
        )
        return locales[_DEFAULT_LOCALE]

    # 使用第一个可用语言
    first_locale = next(iter(locales))
    logger.warning(
        "template_locale_using_first",
        name=name,
        requested=locale,
        using=first_locale,
    )
    return locales[first_locale]


def list_templates() -> list[str]:
    """列出所有已注册的模板名称

    Returns:
        模板名称列表
    """
    return list(_template_registry.keys())


def delete_template(name: str, locale: str | None = None) -> bool:
    """删除模板

    Args:
        name: 模板名称
        locale: 语言代码（如果为 None，删除所有语言）

    Returns:
        是否成功删除
    """
    if name not in _template_registry:
        return False

    if locale is None:
        del _template_registry[name]
        logger.info("template_deleted_all", name=name)
        return True

    if locale in _template_registry[name]:
        del _template_registry[name][locale]
        logger.info("template_deleted_locale", name=name, locale=locale)

        # 如果没有语言了，删除整个模板
        if not _template_registry[name]:
            del _template_registry[name]

        return True

    return False


# ============== 模板渲染 ==============


def render_prompt(
    name: str,
    locale: str = "zh-CN",
    **variables: Any,
) -> str:
    """渲染模板

    Args:
        name: 模板名称
        locale: 语言代码
        **variables: 模板变量

    Returns:
        渲染后的字符串

    Raises:
        KeyError: 如果模板不存在
        TemplateSyntaxError: 如果模板语法错误

    Examples:
        ```python
        # 简单渲染
        prompt = render_prompt("chat", name="用户")

        # 带多个变量
        prompt = render_prompt(
            "router",
            agents=[{"name": "search", "description": "搜索"}],
            input="查询天气"
        )
        ```
    """
    # 添加默认变量
    default_vars = {
        "locale": locale,
        "now": datetime.now,
        "today": datetime.now().date,
        "env": os.getenv,
    }
    variables = {**default_vars, **variables}

    # 获取模板内容
    template_str = get_template(name, locale)

    # 创建模板并渲染
    try:
        env = _get_jinja_env()
        template = env.from_string(template_str)
        result = template.render(**variables)
        logger.info("template_rendered", name=name, locale=locale)
        return result
    except TemplateSyntaxError as e:
        logger.error("template_syntax_error", name=name, error=str(e))
        raise
    except Exception as e:
        logger.error("template_render_error", name=name, error=str(e))
        # 返回原始模板而不是抛出异常
        return template_str


def render_template_string(
    template_str: str,
    **variables: Any,
) -> str:
    """渲染模板字符串

    Args:
        template_str: 模板字符串
        **variables: 模板变量

    Returns:
        渲染后的字符串

    Examples:
        ```python
        result = render_template_string("Hello {{ name }}!", name="World")
        # => "Hello World!"
        ```
    """
    try:
        env = _get_jinja_env()
        template = env.from_string(template_str)
        return template.render(**variables)
    except TemplateSyntaxError as e:
        logger.error("template_string_syntax_error", error=str(e))
        raise
    except Exception as e:
        logger.error("template_string_render_error", error=str(e))
        return template_str


# ============== LangChain 集成 ==============


def create_langchain_prompt(
    name: str,
    locale: str = "zh-CN",
    **variables: Any,
) -> ChatPromptTemplate:
    """创建 LangChain ChatPromptTemplate

    Args:
        name: 模板名称
        locale: 语言代码
        **variables: 默认变量值

    Returns:
        ChatPromptTemplate 实例

    Examples:
        ```python
        from langchain_core.messages import HumanMessage

        prompt_template = create_langchain_prompt("chat")
        messages = prompt_template.format_messages(
            messages=[HumanMessage(content="你好")]
        )
        ```
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    template_str = render_prompt(name, locale, **variables)

    return ChatPromptTemplate.from_messages(
        [
            ("system", template_str),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def create_structured_prompt(
    system_template: str,
    locale: str = "zh-CN",
) -> ChatPromptTemplate:
    """创建结构化 Prompt 模板

    Args:
        system_template: 系统提示词模板（Jinja2 格式）
        locale: 语言代码

    Returns:
        ChatPromptTemplate 实例
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 渲染系统模板
    rendered = render_template_string(system_template, locale=locale)

    return ChatPromptTemplate.from_messages(
        [
            ("system", rendered),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


# ============== 批量操作 ==============


def load_templates_from_dir(directory: str | Path) -> int:
    """从目录加载所有模板

    目录结构:
        directory/
            zh-CN/
                chat.jinja2
                router.jinja2
            en-US/
                chat.jinja2
                router.jinja2

    Args:
        directory: 模板目录路径

    Returns:
        加载的模板数量
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning("template_dir_not_found", path=str(directory))
        return 0

    count = 0
    for locale_dir in directory.iterdir():
        if not locale_dir.is_dir():
            continue

        locale = locale_dir.name
        if locale not in _SUPPORTED_LOCALES:
            logger.debug("unsupported_locale", locale=locale)
            continue

        for template_file in locale_dir.glob("*.jinja2"):
            try:
                name = template_file.stem
                content = template_file.read_text(encoding="utf-8")
                register_template(name, content, locale)
                count += 1
            except Exception as e:
                logger.error(
                    "template_file_load_failed",
                    path=str(template_file),
                    error=str(e),
                )

    logger.info("templates_loaded_from_dir", count=count, directory=str(directory))
    return count


def export_templates_to_dir(directory: str | Path) -> int:
    """导出所有模板到目录

    Args:
        directory: 目标目录路径

    Returns:
        导出的模板数量
    """
    directory = Path(directory)
    count = 0

    for name, locales in _template_registry.items():
        for locale, content in locales.items():
            # 创建语言目录
            locale_dir = directory / locale
            locale_dir.mkdir(parents=True, exist_ok=True)

            # 写入模板文件
            template_file = locale_dir / f"{name}.jinja2"
            template_file.write_text(content, encoding="utf-8")
            count += 1

    logger.info("templates_exported_to_dir", count=count, directory=str(directory))
    return count


__all__ = [
    # 模板注册
    "register_template",
    "register_template_file",
    "get_template",
    "list_templates",
    "delete_template",
    # 模板渲染
    "render_prompt",
    "render_template_string",
    # LangChain 集成
    "create_langchain_prompt",
    "create_structured_prompt",
    # 批量操作
    "load_templates_from_dir",
    "export_templates_to_dir",
]
