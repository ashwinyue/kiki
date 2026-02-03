"""工具系统模块

提供 LangChain 工具的注册、管理和执行。
集成 MCP (Model Context Protocol) 工具支持。

使用示例:
    ```python
    from app.agent.tools import register_tool, list_tools, get_tool_node

    # 注册自定义工具
    @register_tool
    async def my_tool(query: str) -> str:
        \"\"\"我的自定义工具\"\"\"
        return f"结果: {query}"

    # 获取工具节点（同步方式，不包含 MCP 工具）
    tool_node = get_tool_node()

    # 获取工具节点（异步方式，包含 MCP 工具）- 推荐
    tool_node = await aget_tool_node()
    ```
"""

# 导出内置工具
from app.agent.tools.builtin import (
    calculate,
    close_crawler,
    crawl_multiple_urls,
    crawl_url,
    get_weather,
    python_repl,
    search_database,
    search_web,
    search_web_tavily,
    web_fetch,
)
from app.agent.tools.decorators import (
    LoggedToolMixin,
    create_logged_tool,
    log_io,
    log_io_async,
    track_tool_metrics,
)

# 导出工具注册系统
from app.agent.tools.registry import (
    BaseToolRegistry,
    ToolRegistry,
    aget_tool_node,
    alist_tools,
    clear_mcp_tools_cache,
    default_tool_error_handler,
    get_tool,
    get_tool_node,
    list_tools,
    register_tool,
    set_tool_error_handler,
    tool,
)
from app.agent.tools.search_postprocessor import (
    SearchResultPostProcessor,
    extract_domain,
    is_pdf_url,
    is_valid_url,
    normalize_url,
)

# 导出 MCP 工具集成（可选依赖）
try:
    from app.agent.tools.mcp import (
        PREDEFINED_MCP_SERVERS,
        MCPClient,
        MCPRegistry,
        get_predefined_mcp_server,
        list_predefined_mcp_servers,
        load_mcp_tools,
        register_mcp_from_config,
        register_mcp_servers,
    )

    _mcp_available = True
except ImportError:
    _mcp_available = False

    # 创建占位符，避免导入错误
    def _not_implemented(*args, **kwargs):  # type: ignore
        raise ImportError("MCP 功能需要安装 mcp 包: uv add mcp")

    MCPClient = _not_implemented  # type: ignore
    MCPRegistry = _not_implemented  # type: ignore
    PREDEFINED_MCP_SERVERS = {}
    get_predefined_mcp_server = _not_implemented  # type: ignore
    list_predefined_mcp_servers = _not_implemented  # type: ignore
    load_mcp_tools = _not_implemented  # type: ignore
    register_mcp_from_config = _not_implemented  # type: ignore
    register_mcp_servers = _not_implemented  # type: ignore

__all__ = [
    # 工具注册
    "register_tool",
    "get_tool",
    "list_tools",
    "get_tool_node",
    "tool",
    "ToolRegistry",
    "BaseToolRegistry",
    # 异步工具（包含 MCP 工具）
    "alist_tools",
    "aget_tool_node",
    "clear_mcp_tools_cache",
    # 错误处理
    "default_tool_error_handler",
    "set_tool_error_handler",
    # 工具装饰器
    "log_io",
    "log_io_async",
    "create_logged_tool",
    "LoggedToolMixin",
    "track_tool_metrics",
    # 搜索结果处理
    "SearchResultPostProcessor",
    "is_pdf_url",
    "is_valid_url",
    "extract_domain",
    "normalize_url",
    # 内置搜索工具
    "search_web",
    "search_web_tavily",
    # 内置爬虫工具
    "crawl_url",
    "crawl_multiple_urls",
    "close_crawler",
    # 网页内容提取
    "web_fetch",
    # 代码执行
    "python_repl",
    # 其他内置工具
    "search_database",
    "get_weather",
    "calculate",
    # MCP 工具集成
    "MCPClient",
    "MCPRegistry",
    "PREDEFINED_MCP_SERVERS",
    "get_predefined_mcp_server",
    "list_predefined_mcp_servers",
    "load_mcp_tools",
    "register_mcp_from_config",
    "register_mcp_servers",
]

# 自动注册内置工具
_builtin_tools = [
    search_web,
    search_database,
    get_weather,
    calculate,
    crawl_url,
    web_fetch,
    python_repl,
]
for tool_obj in _builtin_tools:
    register_tool(tool_obj)
