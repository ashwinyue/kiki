"""MCP 工具函数单元测试

参考 DeerFlow 的 MCP 测试实现。
由于 MCP SDK 是条件导入的，使用简化测试策略。

Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.agent.mcp.utils import (
    filter_mcp_tools,
    mcp_tool_to_dict,
)


def test_mcp_tool_to_dict():
    """测试 MCP 工具转字典"""
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {"type": "object"}

    result = mcp_tool_to_dict(mock_tool)
    assert result == {
        "name": "test_tool",
        "description": "Test tool description",
        "input_schema": {"type": "object"},
    }


def test_filter_mcp_tools_no_filter():
    """测试不过滤工具"""
    tools = ["tool1", "tool2", "tool3"]
    result = filter_mcp_tools(tools)
    assert result == tools


def test_filter_mcp_tools_with_filter():
    """测试过滤工具"""
    tools = ["tool1", "tool2", "tool3"]

    # 创建模拟工具对象
    mock_tools = []
    for name in tools:
        mock_tool = MagicMock()
        mock_tool.name = name
        mock_tools.append(mock_tool)

    result = filter_mcp_tools(mock_tools, enabled_tools=["tool1", "tool3"])
    assert len(result) == 2
    assert result[0].name == "tool1"
    assert result[1].name == "tool3"


# 集成测试 - 需要安装 MCP SDK
@pytest.mark.asyncio
async def test_load_mcp_tools_stdio_missing_command():
    """测试 stdio 类型缺少命令参数"""
    from app.agent.mcp.utils import load_mcp_tools

    with pytest.raises(HTTPException) as exc:
        await load_mcp_tools(server_type="stdio")
    assert exc.value.status_code == 400
    assert "Command is required" in exc.value.detail


@pytest.mark.asyncio
async def test_load_mcp_tools_sse_missing_url():
    """测试 sse 类型缺少 URL 参数"""
    from app.agent.mcp.utils import load_mcp_tools

    with pytest.raises(HTTPException) as exc:
        await load_mcp_tools(server_type="sse")
    assert exc.value.status_code == 400
    assert "URL is required" in exc.value.detail


@pytest.mark.asyncio
async def test_load_mcp_tools_unsupported_type():
    """测试不支持的传输类型"""
    from app.agent.mcp.utils import load_mcp_tools

    with pytest.raises(HTTPException) as exc:
        await load_mcp_tools(server_type="unknown")
    assert exc.value.status_code == 400
    assert "Unsupported server type" in exc.value.detail
