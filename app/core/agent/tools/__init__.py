"""工具系统模块

提供 LangChain 工具的注册、管理和执行。

使用示例:
    ```python
    from app.core.agent.tools import register_tool, list_tools, get_tool_node

    # 注册自定义工具
    @register_tool
    async def my_tool(query: str) -> str:
        \"\"\"我的自定义工具\"\"\"
        return f"结果: {query}"

    # 获取工具节点
    tool_node = get_tool_node()
    ```
"""

from app.core.agent.tools.builtin.calculation import calculate

# 导出内置示例工具
from app.core.agent.tools.builtin.database import search_database
from app.core.agent.tools.builtin.weather import get_weather
from app.core.agent.tools.registry import (
    BaseToolRegistry,
    ToolRegistry,
    get_tool,
    get_tool_node,
    list_tools,
    register_tool,
)

__all__ = [
    # 工具注册
    "register_tool",
    "get_tool",
    "list_tools",
    "get_tool_node",
    "ToolRegistry",
    "BaseToolRegistry",
    # 内置示例工具
    "search_database",
    "get_weather",
    "calculate",
]

# 自动注册内置示例工具
for tool_obj in [search_database, get_weather, calculate]:
    register_tool(tool_obj)
