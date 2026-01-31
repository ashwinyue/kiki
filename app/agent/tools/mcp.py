"""MCP (Model Context Protocol) 工具集成

支持从 MCP 服务器动态加载工具。
"""

import json
from typing import Any

from langchain_core.tools import StructuredTool

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== MCP 客户端 ==============


class MCPClient:
    """MCP 客户端

    通过 stdio 或 HTTP 与 MCP 服务器通信，获取工具列表。
    """

    def __init__(
        self,
        name: str,
        tenant_id: int | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        transport: str = "stdio",  # stdio, http, sse
        url: str | None = None,
    ):
        """初始化 MCP 客户端

        Args:
            name: MCP 服务器名称
            command: 启动命令（用于 stdio transport）
            args: 命令参数（用于 stdio transport）
            transport: 传输方式
            url: 服务器 URL（用于 http/sse transport）
        """
        self.name = name
        self.tenant_id = tenant_id
        self.command = command
        self.args = args or []
        self.transport = transport
        self.url = url
        self._tools: dict[str, StructuredTool] = {}
        self._process: Any = None
        self._initialized = False

    async def initialize(self) -> bool:
        """初始化 MCP 连接

        Returns:
            是否成功
        """
        if self._initialized:
            return True

        try:
            if self.transport == "stdio":
                success = await self._initialize_stdio()
            elif self.transport in ("http", "sse"):
                success = await self._initialize_http()
            else:
                logger.error("unknown_transport", transport=self.transport)
                return False

            if success:
                self._initialized = True
                logger.info(
                    "mcp_client_initialized",
                    name=self.name,
                    transport=self.transport,
                    tool_count=len(self._tools),
                )

            return success

        except Exception as e:
            logger.error("mcp_init_failed", name=self.name, error=str(e))
            return False

    async def _initialize_stdio(self) -> bool:
        """初始化 stdio 传输

        Returns:
            是否成功
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
            )

            self._session = stdio_client(server_params)

            await self._session.__aenter__()

            # 列出可用工具
            response = await self._session.list_tools()

            # 转换为 LangChain 工具
            for tool in response.tools:
                self._tools[tool.name] = self._mcp_tool_to_langchain(tool)

            logger.info("mcp_stdio_tools_loaded", count=len(self._tools))
            return True

        except ImportError:
            logger.warning("mcp_not_installed", transport="stdio")
            return False
        except Exception as e:
            logger.error("mcp_stdio_init_failed", error=str(e))
            return False

    async def _initialize_http(self) -> bool:
        """初始化 HTTP/SSE 传输

        Returns:
            是否成功
        """
        try:
            from mcp import ClientSession
            from mcp.client.http import http_client

            if not self.url:
                logger.error("mcp_http_url_not_set")
                return False

            self._session = http_client(self.url)
            await self._session.__aenter__()

            # 列出可用工具
            response = await self._session.list_tools()

            for tool in response.tools:
                self._tools[tool.name] = self._mcp_tool_to_langchain(tool)

            logger.info("mcp_http_tools_loaded", count=len(self._tools))
            return True

        except ImportError:
            logger.warning("mcp_not_installed", transport="http")
            return False
        except Exception as e:
            logger.error("mcp_http_init_failed", error=str(e))
            return False

    def _mcp_tool_to_langchain(self, mcp_tool: Any) -> StructuredTool:
        """将 MCP 工具转换为 LangChain 工具

        Args:
            mcp_tool: MCP 工具对象

        Returns:
            LangChain 工具
        """

        async def invoke_tool(**kwargs) -> str:
            """调用 MCP 工具"""
            try:
                response = await self._session.call_tool(
                    mcp_tool.name,
                    arguments=kwargs,
                )
                # 处理响应内容
                if hasattr(response, "content"):
                    if isinstance(response.content, list):
                        return json.dumps([c.dict() for c in response.content])
                    return str(response.content)
                return str(response)
            except Exception as e:
                logger.error(
                    "mcp_tool_call_failed",
                    tool=mcp_tool.name,
                    error=str(e),
                )
                return f"工具调用失败: {str(e)}"

        # 构建 input_schema
        input_schema = mcp_tool.inputSchema

        return StructuredTool.from_function(
            func=invoke_tool,
            name=mcp_tool.name,
            description=mcp_tool.description,
            args_schema=input_schema,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> Any:
        """调用 MCP 工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if not self._initialized:
            await self.initialize()

        if tool_name not in self._tools:
            raise ValueError(f"工具不存在: {tool_name}")

        return await self._session.call_tool(tool_name, arguments)

    def list_tools(self) -> list[str]:
        """列出可用工具

        Returns:
            工具名称列表
        """
        return list(self._tools.keys())

    def get_langchain_tools(self) -> list[StructuredTool]:
        """获取所有 LangChain 格式的工具

        Returns:
            工具列表
        """
        return list(self._tools.values())

    async def close(self) -> None:
        """关闭 MCP 连接"""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._initialized = False
            logger.info("mcp_client_closed", name=self.name)


# ============== MCP 服务器注册表 ==============


class MCPRegistry:
    """MCP 服务器注册表

    管理多个 MCP 服务器连接。
    """

    _clients: dict[str, MCPClient] = {}

    @classmethod
    def register(
        cls,
        name: str,
        tenant_id: int | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        transport: str = "stdio",
        url: str | None = None,
    ) -> MCPClient:
        """注册 MCP 服务器

        Args:
            name: 服务器名称
            command: 启动命令
            args: 命令参数
            transport: 传输方式
            url: 服务器 URL

        Returns:
            MCPClient 实例

        Examples:
            ```python
            # 注册 uvx 启动的服务器
            MCPRegistry.register(
                name="weather",
                command="uvx",
                args=["mcp-server-weather"],
            )

            # 注册 HTTP 服务器
            MCPRegistry.register(
                name="custom",
                transport="http",
                url="http://localhost:3000/mcp",
            )
            ```
        """
        client = MCPClient(
            name=name,
            tenant_id=tenant_id,
            command=command,
            args=args,
            transport=transport,
            url=url,
        )
        cls._clients[name] = client
        logger.info("mcp_server_registered", name=name, transport=transport)
        return client

    @classmethod
    def get(cls, name: str) -> MCPClient | None:
        """获取 MCP 客户端

        Args:
            name: 服务器名称

        Returns:
            MCPClient 实例或 None
        """
        return cls._clients.get(name)

    @classmethod
    async def initialize_all(
        cls,
        tenant_id: int | None = None,
        include_global: bool = True,
    ) -> None:
        """初始化所有注册的服务器"""
        for client in cls._clients.values():
            if tenant_id is None:
                if client.tenant_id is None:
                    await client.initialize()
                continue
            if client.tenant_id == tenant_id or (include_global and client.tenant_id is None):
                await client.initialize()

    @classmethod
    def get_all_tools(
        cls,
        tenant_id: int | None = None,
        include_global: bool = True,
    ) -> list[StructuredTool]:
        """获取所有服务器的工具

        Returns:
            工具列表
        """
        all_tools = []
        for client in cls._clients.values():
            if not client._initialized:
                continue
            if tenant_id is None:
                if client.tenant_id is None:
                    all_tools.extend(client.get_langchain_tools())
                continue
            if client.tenant_id == tenant_id or (include_global and client.tenant_id is None):
                all_tools.extend(client.get_langchain_tools())
        return all_tools

    @classmethod
    async def close_all(cls) -> None:
        """关闭所有连接"""
        for client in cls._clients.values():
            await client.close()


# ============== 便捷函数 ==============


async def register_mcp_from_config(config: dict) -> MCPClient:
    """从配置注册 MCP 服务器

    Args:
        config: 配置字典

    Returns:
        MCPClient 实例

    Examples:
        ```python
        config = {
            "name": "filesystem",
            "command": "uvx",
            "args": ["mcp-server-filesystem", "/path/to/allowed"],
        }
        client = await register_mcp_from_config(config)
        ```
    """
    client = MCPRegistry.register(
        name=config["name"],
        tenant_id=config.get("tenant_id"),
        command=config.get("command"),
        args=config.get("args"),
        transport=config.get("transport", "stdio"),
        url=config.get("url"),
    )

    await client.initialize()
    return client


def register_mcp_servers(servers: list[dict]) -> None:
    """批量注册 MCP 服务器

    Args:
        servers: 服务器配置列表

    Examples:
        ```python
        servers = [
            {
                "name": "weather",
                "command": "uvx",
                "args": ["mcp-server-weather"],
            },
            {
                "name": "github",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            },
        ]
        register_mcp_servers(servers)
        ```
    """
    for server_config in servers:
        MCPRegistry.register(**server_config)


async def load_mcp_tools(
    tenant_id: int | None = None,
    include_global: bool = True,
) -> list[StructuredTool]:
    """加载所有 MCP 工具

    Returns:
        工具列表
    """
    await MCPRegistry.initialize_all(tenant_id=tenant_id, include_global=include_global)
    return MCPRegistry.get_all_tools(tenant_id=tenant_id, include_global=include_global)


# ============== 预定义的 MCP 服务器配置 ==============

# 常用的 MCP 服务器
PREDEFINED_MCP_SERVERS = {
    "filesystem": {
        "name": "filesystem",
        "command": "uvx",
        "args": ["mcp-server-filesystem", "/allowed/path"],
        "description": "文件系统访问",
    },
    "github": {
        "name": "github",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "description": "GitHub 仓库操作",
    },
    "postgres": {
        "name": "postgres",
        "command": "uvx",
        "args": ["mcp-server-postgres", "--connection-string", "postgresql://..."],
        "description": "PostgreSQL 数据库查询",
    },
    "brave-search": {
        "name": "brave-search",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "description": "Brave 搜索引擎",
    },
    "fetch": {
        "name": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "网页抓取",
    },
}


def get_predefined_mcp_server(name: str) -> dict | None:
    """获取预定义的 MCP 服务器配置

    Args:
        name: 服务器名称

    Returns:
        配置字典或 None
    """
    return PREDEFINED_MCP_SERVERS.get(name)


def list_predefined_mcp_servers() -> list[str]:
    """列出所有预定义的 MCP 服务器

    Returns:
        服务器名称列表
    """
    return list(PREDEFINED_MCP_SERVERS.keys())
