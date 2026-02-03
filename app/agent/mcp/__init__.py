"""MCP (Model Context Protocol) 客户端模块

参考 DeerFlow 的 MCP 架构设计，支持动态加载和发现 MCP 服务器工具。

传输类型支持:
- stdio: 标准输入输出通信
- sse: Server-Sent Events
- streamable_http: HTTP 流式传输

使用示例:
    ```python
    from app.agent.mcp import load_mcp_tools, mcp_tool_to_dict

    # 加载 stdio 类型 MCP 服务器
    tools = await load_mcp_tools(
        server_type="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/allowed/path"],
    )

    # 加载 sse 类型 MCP 服务器
    tools = await load_mcp_tools(
        server_type="sse",
        url="http://localhost:3000/sse",
        headers={"Authorization": "Bearer xxx"},
    )
    ```
"""

from app.agent.mcp.request import (
    MCPServerConfig,
    MCPServerMetadataRequest,
    MCPServerMetadataResponse,
    MCPSettings,
    MCPToolDefinition,
)
from app.agent.mcp.utils import (
    filter_mcp_tools,
    load_mcp_tools,
    mcp_tool_to_dict,
)

__all__ = [
    # 数据模型
    "MCPServerMetadataRequest",
    "MCPServerMetadataResponse",
    "MCPToolDefinition",
    "MCPServerConfig",
    "MCPSettings",
    # 工具函数
    "load_mcp_tools",
    "mcp_tool_to_dict",
    "filter_mcp_tools",
]
