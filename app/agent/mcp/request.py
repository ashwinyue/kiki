"""MCP 请求/响应模型

参考 DeerFlow 的 MCP 数据模型设计。
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPServerMetadataRequest(BaseModel):
    """MCP 服务器元数据请求模型"""

    transport: str = Field(
        ...,
        description="MCP 服务器连接类型 (stdio, sse, streamable_http)",
    )
    command: Optional[str] = Field(
        None, description="启动命令 (用于 stdio 类型)"
    )
    args: Optional[List[str]] = Field(
        None, description="命令参数 (用于 stdio 类型)"
    )
    url: Optional[str] = Field(
        None, description="SSE 服务器 URL (用于 sse/streamable_http 类型)"
    )
    env: Optional[Dict[str, str]] = Field(
        None, description="环境变量 (用于 stdio 类型)"
    )
    headers: Optional[Dict[str, str]] = Field(
        None, description="HTTP 请求头 (用于 sse/streamable_http 类型)"
    )
    timeout_seconds: Optional[int] = Field(
        None, description="操作超时时间（秒）"
    )


class MCPServerMetadataResponse(BaseModel):
    """MCP 服务器元数据响应模型"""

    transport: str = Field(
        ...,
        description="MCP 服务器连接类型 (stdio, sse, streamable_http)",
    )
    command: Optional[str] = Field(
        None, description="启动命令 (用于 stdio 类型)"
    )
    args: Optional[List[str]] = Field(
        None, description="命令参数 (用于 stdio 类型)"
    )
    url: Optional[str] = Field(
        None, description="SSE 服务器 URL (用于 sse/streamable_http 类型)"
    )
    env: Optional[Dict[str, str]] = Field(
        None, description="环境变量 (用于 stdio 类型)"
    )
    headers: Optional[Dict[str, str]] = Field(
        None, description="HTTP 请求头 (用于 sse/streamable_http 类型)"
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="从 MCP 服务器获取的可用工具列表",
    )


class MCPToolDefinition(BaseModel):
    """MCP 工具定义模型"""

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    input_schema: Dict[str, Any] = Field(..., description="工具输入参数 JSON Schema")


class MCPServerConfig(BaseModel):
    """MCP 服务器配置模型

    用于在聊天请求中配置 MCP 服务器。
    参考 DeerFlow 的 mcp_settings 结构。
    """

    transport: str = Field(
        ...,
        description="MCP 服务器连接类型",
    )
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    enabled_tools: Optional[List[str]] = Field(
        None,
        description="启用的工具名称列表，为空表示启用所有工具",
    )
    add_to_agents: Optional[List[str]] = Field(
        None,
        description="添加到哪些 Agent 的工具列表",
    )


class MCPSettings(BaseModel):
    """MCP 设置模型

    用于聊天流请求中配置多个 MCP 服务器。
    """

    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP 服务器配置字典，key 为服务器名称",
    )
