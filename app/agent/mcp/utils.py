"""MCP 工具加载核心逻辑

参考 DeerFlow 的 MCP 工具加载实现。
支持 stdio、sse、streamable_http 三种传输类型。

Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import logging
from datetime import timedelta
from typing import Any, List, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def _get_tools_from_client_session(
    client_context_manager: Any, timeout_seconds: int = 10
) -> List:
    """从客户端会话获取工具列表

    Args:
        client_context_manager: 返回 (read, write) 函数的上下文管理器
        timeout_seconds: 读取操作超时时间（秒）

    Returns:
        MCP 服务器可用工具列表

    Raises:
        Exception: 过程中发生错误
    """
    try:
        from mcp import ClientSession
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="MCP SDK not installed. Install with: uv add mcp",
        )

    async with client_context_manager as context_result:
        # 按索引访问以保持安全
        read = context_result[0]
        write = context_result[1]
        # 忽略任何额外值

        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=timeout_seconds)
        ) as session:
            # 初始化连接
            await session.initialize()
            # 列出可用工具
            listed_tools = await session.list_tools()
            return listed_tools.tools


async def load_mcp_tools(
    server_type: str,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    url: Optional[str] = None,
    env: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout_seconds: int = 60,  # 首次执行使用更长默认超时
) -> List:
    """从 MCP 服务器加载工具

    Args:
        server_type: MCP 服务器连接类型 (stdio, sse, streamable_http)
        command: 执行命令 (用于 stdio 类型)
        args: 命令参数 (用于 stdio 类型)
        url: SSE/HTTP 服务器的 URL (用于 sse/streamable_http 类型)
        env: 环境变量 (用于 stdio 类型)
        headers: HTTP 请求头 (用于 sse/streamable_http 类型)
        timeout_seconds: 超时时间（秒），默认 60 秒用于首次执行

    Returns:
        MCP 服务器可用工具列表

    Raises:
        HTTPException: 加载工具时发生错误
    """
    try:
        if server_type == "stdio":
            if not command:
                raise HTTPException(
                    status_code=400, detail="Command is required for stdio type"
                )

            try:
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="MCP SDK not installed. Install with: uv add mcp",
                )

            server_params = StdioServerParameters(
                command=command,  # 可执行文件
                args=args,  # 可选命令行参数
                env=env,  # 可选环境变量
            )

            return await _get_tools_from_client_session(
                stdio_client(server_params), timeout_seconds
            )

        elif server_type == "sse":
            if not url:
                raise HTTPException(
                    status_code=400, detail="URL is required for sse type"
                )

            try:
                from mcp.client.sse import sse_client
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="MCP SDK not installed. Install with: uv add mcp",
                )

            return await _get_tools_from_client_session(
                sse_client(url=url, headers=headers, timeout=timeout_seconds),
                timeout_seconds,
            )

        elif server_type == "streamable_http":
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail="URL is required for streamable_http type",
                )

            try:
                from mcp.client.streamable_http import streamablehttp_client
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="MCP SDK not installed. Install with: uv add mcp",
                )

            return await _get_tools_from_client_session(
                streamablehttp_client(
                    url=url, headers=headers, timeout=timeout_seconds
                ),
                timeout_seconds,
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported server type: {server_type}"
            )

    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.exception(f"Error loading MCP tools: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        raise


def mcp_tool_to_dict(mcp_tool: Any) -> dict:
    """将 MCP 工具对象转换为字典

    Args:
        mcp_tool: MCP 工具对象

    Returns:
        工具信息字典
    """
    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description,
        "input_schema": mcp_tool.inputSchema,
    }


def filter_mcp_tools(
    tools: List[Any],
    enabled_tools: Optional[List[str]] = None,
) -> List[Any]:
    """过滤 MCP 工具列表

    Args:
        tools: 所有可用工具列表
        enabled_tools: 启用的工具名称列表，为空表示启用所有

    Returns:
        过滤后的工具列表
    """
    if not enabled_tools:
        return tools

    enabled_set = set(enabled_tools)
    return [t for t in tools if t.name in enabled_set]
