"""工具注册系统

提供工具的注册、查询和 ToolNode 创建功能。
"""


from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import ToolNode

from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseToolRegistry:
    """工具注册表抽象基类"""

    def register(self, tool_obj: BaseTool) -> None:
        """注册工具"""
        raise NotImplementedError

    def get(self, name: str) -> BaseTool | None:
        """获取工具"""
        raise NotImplementedError

    def list_all(self) -> list[BaseTool]:
        """列出所有工具"""
        raise NotImplementedError

    def create_tool_node(self) -> ToolNode:
        """创建 ToolNode"""
        raise NotImplementedError


class ToolRegistry(BaseToolRegistry):
    """工具注册表

    管理所有已注册的 LangChain 工具。
    """

    def __init__(self) -> None:
        self._registry: dict[str, BaseTool] = {}

    def register(self, tool_obj: BaseTool) -> None:
        """注册工具到注册表

        Args:
            tool_obj: LangChain 工具实例
        """
        self._registry[tool_obj.name] = tool_obj
        logger.info(
            "tool_registered",
            tool_name=tool_obj.name,
            tool_type=type(tool_obj).__name__,
        )

    def get(self, name: str) -> BaseTool | None:
        """获取工具

        Args:
            name: 工具名称

        Returns:
            工具实例或 None
        """
        return self._registry.get(name)

    def list_all(self) -> list[BaseTool]:
        """列出所有已注册的工具

        Returns:
            工具列表
        """
        return list(self._registry.values())

    def create_tool_node(self) -> ToolNode:
        """获取包含所有已注册工具的 ToolNode

        Returns:
            ToolNode 实例
        """
        tools = self.list_all()
        logger.debug("creating_tool_node", tool_count=len(tools))
        return ToolNode(tools)


# 全局工具注册表实例
_global_registry = ToolRegistry()


def register_tool(tool_obj: BaseTool) -> None:
    """注册工具到全局注册表

    Args:
        tool_obj: LangChain 工具实例

    Examples:
        ```python
        from langchain_core.tools import tool

        @tool
        async def my_tool(query: str) -> str:
            \"\"\"我的工具\"\"\"
            return f"结果: {query}"

        register_tool(my_tool)
        ```
    """
    _global_registry.register(tool_obj)


def get_tool(name: str) -> BaseTool | None:
    """获取工具

    Args:
        name: 工具名称

    Returns:
        工具实例或 None
    """
    return _global_registry.get(name)


def list_tools() -> list[BaseTool]:
    """列出所有已注册的工具

    Returns:
        工具列表
    """
    return _global_registry.list_all()


def get_tool_node() -> ToolNode:
    """获取包含所有已注册工具的 ToolNode

    Returns:
        ToolNode 实例
    """
    return _global_registry.create_tool_node()


def tool(*args, **kwargs):
    """工具装饰器

    便捷的装饰器，自动注册工具。

    Examples:
        ```python
        from app.core.agent.tools import tool

        @tool
        async def my_tool(query: str) -> str:
            \"\"\"我的工具描述\"\"\"
            return f"结果: {query}"
        ```
    """
    def decorator(func):
        tool_obj = lc_tool(*args, **kwargs)(func)
        register_tool(tool_obj)
        return tool_obj

    # 支持 @tool 和 @tool() 两种形式
    if args and callable(args[0]):
        # @tool 形式
        func = args[0]
        tool_obj = lc_tool(func)
        register_tool(tool_obj)
        return tool_obj  # 返回工具对象，而不是原始函数
    else:
        # @tool() 形式
        return decorator
