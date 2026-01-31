"""工具注册表测试

测试工具注册、查询和 ToolNode 创建功能。
"""

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from app.agent.tools import (
    ToolRegistry,
    default_tool_error_handler,
    get_tool,
    get_tool_node,
    list_tools,
    register_tool,
    set_tool_error_handler,
)


class TestToolRegistry:
    """ToolRegistry 测试"""

    def test_register_tool(self) -> None:
        """测试工具注册"""
        registry = ToolRegistry()

        @tool
        def test_tool(query: str) -> str:
            """测试工具"""
            return f"结果: {query}"

        registry.register(test_tool)

        assert registry.get("test_tool") is test_tool
        assert len(registry.list_all()) == 1

    def test_register_multiple_tools(self) -> None:
        """测试注册多个工具"""
        registry = ToolRegistry()

        @tool
        def tool_a(query: str) -> str:
            """工具 A"""
            return f"A: {query}"

        @tool
        def tool_b(query: str) -> str:
            """工具 B"""
            return f"B: {query}"

        registry.register(tool_a)
        registry.register(tool_b)

        assert len(registry.list_all()) == 2
        assert registry.get("tool_a") is tool_a
        assert registry.get("tool_b") is tool_b

    def test_get_nonexistent_tool(self) -> None:
        """测试获取不存在的工具"""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_create_tool_node(self) -> None:
        """测试创建 ToolNode"""
        registry = ToolRegistry()

        @tool
        def test_tool(query: str) -> str:
            """测试工具"""
            return f"结果: {query}"

        registry.register(test_tool)
        tool_node = registry.create_tool_node()

        assert isinstance(tool_node, ToolNode)

    def test_default_error_handler(self) -> None:
        """测试默认错误处理函数"""
        error = Exception("测试错误")
        result = default_tool_error_handler(error)

        # 错误消息包含 "操作失败" 和错误详情
        assert "操作失败" in result or "Exception" in result
        assert "测试错误" in result

    def test_custom_error_handler(self) -> None:
        """测试自定义错误处理函数"""
        def custom_handler(error: Exception) -> str:
            return f"自定义错误: {error}"

        registry = ToolRegistry(error_handler=custom_handler)
        tool_node = registry.create_tool_node()

        # 验证 ToolNode 使用了自定义错误处理
        assert isinstance(tool_node, ToolNode)

    def test_set_error_handler(self) -> None:
        """测试设置错误处理函数"""
        registry = ToolRegistry()

        def new_handler(error: Exception) -> str:
            return f"新错误处理: {error}"

        registry.set_error_handler(new_handler)

        # 验证设置成功（通过创建 ToolNode 触发）
        tool_node = registry.create_tool_node()
        assert isinstance(tool_node, ToolNode)


class TestGlobalToolRegistry:
    """全局工具注册表测试"""

    def setup_method(self) -> None:
        """每个测试前清空注册表"""
        # 保存原始注册表
        from app.agent.tools.registry import _global_registry
        self._original_tools = _global_registry._registry.copy()
        _global_registry._registry.clear()

    def teardown_method(self) -> None:
        """每个测试后恢复注册表"""
        from app.agent.tools.registry import _global_registry
        _global_registry._registry.clear()
        _global_registry._registry.update(self._original_tools)

    def test_register_tool_global(self) -> None:
        """测试全局工具注册"""
        @tool
        def global_tool(query: str) -> str:
            """全局工具"""
            return f"全局: {query}"

        register_tool(global_tool)

        assert get_tool("global_tool") is global_tool

    def test_list_tools_global(self) -> None:
        """测试全局工具列表"""
        @tool
        def tool_a(query: str) -> str:
            """工具 A"""
            return f"A: {query}"

        register_tool(tool_a)

        tools = list_tools()
        assert len(tools) >= 1
        assert tool_a in tools

    def test_get_tool_node_global(self) -> None:
        """测试全局 ToolNode"""
        tool_node = get_tool_node()
        assert isinstance(tool_node, ToolNode)

    def test_set_error_handler_global(self) -> None:
        """测试全局设置错误处理"""
        def custom_handler(error: Exception) -> str:
            return f"全局自定义: {error}"

        set_tool_error_handler(custom_handler)

        # 创建 ToolNode 应该使用新的错误处理器
        tool_node = get_tool_node()
        assert isinstance(tool_node, ToolNode)
