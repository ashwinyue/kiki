"""MCP 请求数据模型单元测试

参考 DeerFlow 的 MCP 请求模型测试。
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import pytest
from pydantic import ValidationError

from app.agent.mcp.request import (
    MCPServerConfig,
    MCPServerMetadataRequest,
    MCPServerMetadataResponse,
    MCPSettings,
    MCPToolDefinition,
)


def test_mcp_server_metadata_request_stdio():
    """测试 stdio 类型请求模型"""
    request = MCPServerMetadataRequest(
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/allowed/path"],
        env={"API_KEY": "xxx"},
        timeout_seconds=60,
    )
    assert request.transport == "stdio"
    assert request.command == "uvx"
    assert request.args == ["mcp-server-filesystem", "/allowed/path"]
    assert request.env == {"API_KEY": "xxx"}
    assert request.timeout_seconds == 60


def test_mcp_server_metadata_request_sse():
    """测试 sse 类型请求模型"""
    request = MCPServerMetadataRequest(
        transport="sse",
        url="http://localhost:3000/sse",
        headers={"Authorization": "Bearer xxx"},
    )
    assert request.transport == "sse"
    assert request.url == "http://localhost:3000/sse"
    assert request.headers == {"Authorization": "Bearer xxx"}


def test_mcp_server_metadata_request_streamable_http():
    """测试 streamable_http 类型请求模型"""
    request = MCPServerMetadataRequest(
        transport="streamable_http",
        url="http://localhost:3000/mcp",
        headers={"API-Key": "xxx"},
    )
    assert request.transport == "streamable_http"
    assert request.url == "http://localhost:3000/mcp"


def test_mcp_server_metadata_request_missing_required():
    """测试缺少必填字段"""
    with pytest.raises(ValidationError):
        MCPServerMetadataRequest()


def test_mcp_server_metadata_response():
    """测试响应模型"""
    response = MCPServerMetadataResponse(
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem"],
        tools=[
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object"},
            }
        ],
    )
    assert response.transport == "stdio"
    assert len(response.tools) == 1
    assert response.tools[0]["name"] == "read_file"


def test_mcp_tool_definition():
    """测试工具定义模型"""
    tool = MCPToolDefinition(
        name="search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert tool.name == "search"
    assert tool.description == "Search the web"
    assert "query" in tool.input_schema["properties"]


def test_mcp_server_config():
    """测试 MCP 服务器配置模型"""
    config = MCPServerConfig(
        transport="stdio",
        command="uvx",
        args=["mcp-server-weather"],
        enabled_tools=["get_weather", "get_forecast"],
        add_to_agents=["researcher"],
    )
    assert config.transport == "stdio"
    assert config.enabled_tools == ["get_weather", "get_forecast"]
    assert config.add_to_agents == ["researcher"]


def test_mcp_settings():
    """测试 MCP 设置模型"""
    settings = MCPSettings(
        servers={
            "weather": MCPServerConfig(
                transport="stdio",
                command="uvx",
                args=["mcp-server-weather"],
                enabled_tools=["get_weather"],
            ),
            "github": MCPServerConfig(
                transport="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
            ),
        }
    )
    assert len(settings.servers) == 2
    assert "weather" in settings.servers
    assert "github" in settings.servers
    assert settings.servers["weather"].enabled_tools == ["get_weather"]


def test_mcp_settings_empty_servers():
    """测试空服务器配置"""
    settings = MCPSettings()
    assert settings.servers == {}


def test_mcp_server_config_optional_fields():
    """测试可选字段"""
    config = MCPServerConfig(transport="stdio", command="uvx", args=["test"])
    assert config.enabled_tools is None
    assert config.add_to_agents is None
    assert config.url is None
    assert config.headers is None
