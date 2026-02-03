"""工具注册系统

提供工具的注册、查询和 ToolNode 创建功能。
支持自定义工具错误处理。
线程安全/异步安全的全局注册表。
集成 MCP (Model Context Protocol) 工具支持。
"""

from asyncio import Lock
from collections.abc import Callable
from threading import RLock

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import ToolNode

from app.config.errors import handle_tool_error
from app.observability.logging import get_logger

# 尝试导入 MCP 模块（可选依赖）
try:
    from app.agent.tools.mcp import load_mcp_tools

    _mcp_available = True
except ImportError:
    _mcp_available = False

logger = get_logger(__name__)

# 线程锁（保护全局注册表）
_thread_lock = RLock()

# 异步锁（保护异步操作）
_async_lock = Lock()

# MCP 工具缓存（按租户）
_mcp_tools_cache_by_tenant: dict[int | None, list[BaseTool]] = {}
_mcp_tools_loaded_by_tenant: set[int | None] = set()


async def _register_mcp_services_from_db(tenant_id: int | None) -> None:
    """从数据库注册 MCP 服务"""
    try:
        from app.agent.tools.mcp import register_mcp_from_config
        from app.infra.database import mcp_service_repository, session_scope

        async with session_scope() as session:
            repo = mcp_service_repository(session)
            services = await repo.list_enabled(tenant_id=tenant_id, include_global=True)

        for service in services:
            stdio = service.stdio_config or {}
            config = {
                "name": service.name,
                "tenant_id": service.tenant_id,
                "transport": service.transport_type,
                "url": service.url,
                "command": stdio.get("command"),
                "args": stdio.get("args"),
            }
            await register_mcp_from_config(config)
    except Exception as e:
        logger.warning("mcp_services_register_failed", error=str(e))


async def _ensure_mcp_tools_loaded(tenant_id: int | None) -> list[BaseTool]:
    """确保 MCP 工具已加载

    Returns:
        MCP 工具列表
    """
    global _mcp_tools_cache_by_tenant, _mcp_tools_loaded_by_tenant

    if _mcp_available and tenant_id not in _mcp_tools_loaded_by_tenant:
        try:
            await _register_mcp_services_from_db(tenant_id)
            mcp_tools = await load_mcp_tools(tenant_id=tenant_id, include_global=True)
            _mcp_tools_cache_by_tenant[tenant_id] = mcp_tools  # type: ignore
            _mcp_tools_loaded_by_tenant.add(tenant_id)
            logger.info("mcp_tools_loaded", count=len(mcp_tools))
        except Exception as e:
            logger.warning("mcp_tools_load_failed", error=str(e))

    return _mcp_tools_cache_by_tenant.get(tenant_id, [])


def clear_mcp_tools_cache() -> None:
    """清除 MCP 工具缓存

    用于重新加载 MCP 工具的场景。
    """
    global _mcp_tools_cache_by_tenant, _mcp_tools_loaded_by_tenant
    _mcp_tools_cache_by_tenant = {}
    _mcp_tools_loaded_by_tenant = set()
    logger.info("mcp_tools_cache_cleared")


# 默认工具错误处理函数
def default_tool_error_handler(error: Exception) -> str:
    """默认工具错误处理函数

    根据环境返回适当的错误消息：
    - 生产环境：不暴露敏感信息
    - 开发环境：显示详细错误

    Args:
        error: 工具执行抛出的异常

    Returns:
        错误消息字符串
    """
    return handle_tool_error(error)


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

    管理所有已注册的 LangChain 工具，支持自定义错误处理。
    线程安全的实现，支持并发访问。
    """

    def __init__(self, error_handler: Callable[[Exception], str] | None = None) -> None:
        """初始化工具注册表

        Args:
            error_handler: 自定义工具错误处理函数
        """
        self._registry: dict[str, BaseTool] = {}
        self._error_handler = error_handler or default_tool_error_handler
        self._lock = RLock()  # 实例锁，保护当前实例

    def register(self, tool_obj: BaseTool, enable_monitoring: bool = True) -> None:
        """注册工具到注册表（线程安全）

        Args:
            tool_obj: LangChain 工具实例
            enable_monitoring: 是否启用监控（默认 True）
        """
        # 自动应用监控包装器
        if enable_monitoring:
            try:
                from app.agent.tools.observability import MonitoredTool

                # 只对非 MonitoredTool 进行包装
                if not isinstance(tool_obj, MonitoredTool):
                    tool_obj = MonitoredTool.wrap(tool_obj)
                    logger.debug("tool_monitoring_enabled", tool_name=tool_obj.name)
            except ImportError:
                # 监控模块不可用，跳过
                pass

        with self._lock:
            self._registry[tool_obj.name] = tool_obj
            logger.info(
                "tool_registered",
                tool_name=tool_obj.name,
                tool_type=type(tool_obj).__name__,
            )

    def get(self, name: str) -> BaseTool | None:
        """获取工具（线程安全）

        Args:
            name: 工具名称

        Returns:
            工具实例或 None
        """
        with self._lock:
            return self._registry.get(name)

    def list_all(self) -> list[BaseTool]:
        """列出所有已注册的工具（线程安全）

        Returns:
            工具列表（副本）
        """
        with self._lock:
            return list(self._registry.values())

    def create_tool_node(self) -> ToolNode:
        """获取包含所有已注册工具的 ToolNode（线程安全）

        使用自定义错误处理函数处理工具执行错误。

        Returns:
            ToolNode 实例
        """
        tools = self.list_all()
        logger.debug("creating_tool_node", tool_count=len(tools))
        return ToolNode(tools, handle_tool_errors=self._error_handler)

    def set_error_handler(self, error_handler: Callable[[Exception], str]) -> None:
        """设置工具错误处理函数（线程安全）

        Args:
            error_handler: 错误处理函数
        """
        with self._lock:
            self._error_handler = error_handler
            logger.info("error_handler_updated")

    def unregister(self, name: str) -> bool:
        """注销工具（线程安全）

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        with self._lock:
            if name in self._registry:
                del self._registry[name]
                logger.info("tool_unregistered", tool_name=name)
                return True
            return False

    def clear(self) -> None:
        """清空注册表（线程安全）"""
        with self._lock:
            self._registry.clear()
            logger.info("tool_registry_cleared")


# 全局工具注册表实例
_global_registry = ToolRegistry()


def register_tool(tool_obj: BaseTool) -> None:
    """注册工具到全局注册表（线程安全）

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
    with _thread_lock:
        _global_registry.register(tool_obj)


def get_tool(name: str) -> BaseTool | None:
    """获取工具（线程安全）

    Args:
        name: 工具名称

    Returns:
        工具实例或 None
    """
    with _thread_lock:
        return _global_registry.get(name)


def list_tools(include_mcp: bool = False, tenant_id: int | None = None) -> list[BaseTool]:
    """列出所有已注册的工具（线程安全）

    Args:
        include_mcp: 是否包含 MCP 工具（注意：同步模式下只能使用已缓存的 MCP 工具）
        tenant_id: 租户 ID

    Returns:
        工具列表
    """
    with _thread_lock:
        tools = list(_global_registry.list_all())

    if include_mcp and tenant_id in _mcp_tools_cache_by_tenant:
        tools.extend(_mcp_tools_cache_by_tenant[tenant_id])

    return tools


async def alist_tools(include_mcp: bool = True, tenant_id: int | None = None) -> list[BaseTool]:
    """异步列出所有已注册的工具（异步安全）

    这是推荐的方式，尤其是在使用 MCP 工具时。

    Args:
        include_mcp: 是否包含 MCP 工具（默认 True）
        tenant_id: 租户 ID

    Returns:
        工具列表

    Examples:
        ```python
        # 推荐方式：异步获取所有工具（包含 MCP 工具）
        tools = await alist_tools()
        ```
    """
    async with _async_lock:
        tools = list(_global_registry.list_all())

    if include_mcp:
        mcp_tools = await _ensure_mcp_tools_loaded(tenant_id)
        tools.extend(mcp_tools)

    return tools


def get_tool_node(include_mcp: bool = False, tenant_id: int | None = None) -> ToolNode:
    """获取包含所有已注册工具的 ToolNode（线程安全）

    Args:
        include_mcp: 是否包含 MCP 工具（注意：同步模式下只能使用已缓存的 MCP 工具）
        tenant_id: 租户 ID

    Returns:
        ToolNode 实例
    """
    with _thread_lock:
        tools = list(_global_registry.list_all())

    # 添加已缓存的 MCP 工具（如果有）
    if include_mcp and tenant_id in _mcp_tools_cache_by_tenant:
        cached = _mcp_tools_cache_by_tenant[tenant_id]
        tools.extend(cached)
        logger.debug("mcp_tools_included", count=len(cached))

    return ToolNode(tools, handle_tool_errors=_global_registry._error_handler)


async def aget_tool_node(
    include_mcp: bool = True,
    tenant_id: int | None = None,
) -> ToolNode:
    """异步获取包含所有已注册工具的 ToolNode（异步安全）

    这是推荐的方式，尤其是在使用 MCP 工具时。

    Args:
        include_mcp: 是否包含 MCP 工具（默认 True）
        tenant_id: 租户 ID

    Returns:
        ToolNode 实例

    Examples:
        ```python
        # 推荐方式：异步获取 ToolNode（包含 MCP 工具）
        tool_node = await aget_tool_node()
        ```
    """
    async with _async_lock:
        tools = list(_global_registry.list_all())

    # 异步加载 MCP 工具
    if include_mcp:
        mcp_tools = await _ensure_mcp_tools_loaded(tenant_id)
        tools.extend(mcp_tools)
        logger.debug("mcp_tools_included_async", count=len(mcp_tools))

    return ToolNode(tools, handle_tool_errors=_global_registry._error_handler)


def set_tool_error_handler(error_handler: Callable[[Exception], str]) -> None:
    """设置全局工具错误处理函数（线程安全）

    Args:
        error_handler: 错误处理函数，接收异常并返回错误消息

    Examples:
        ```python
        def custom_handler(error: Exception) -> str:
            return f"执行出错: {error}"

        set_tool_error_handler(custom_handler)
        ```
    """
    with _thread_lock:
        _global_registry.set_error_handler(error_handler)


def unregister_tool(name: str) -> bool:
    """注销工具（线程安全）

    Args:
        name: 工具名称

    Returns:
        是否成功注销
    """
    with _thread_lock:
        return _global_registry.unregister(name)


def clear_tools() -> None:
    """清空全局工具注册表（线程安全）"""
    with _thread_lock:
        _global_registry.clear()


async def aregister_tool(tool_obj: BaseTool) -> None:
    """异步注册工具到全局注册表（异步安全）

    Args:
        tool_obj: LangChain 工具实例
    """
    async with _async_lock:
        _global_registry.register(tool_obj)


async def aget_tool(name: str) -> BaseTool | None:
    """异步获取工具（异步安全）

    Args:
        name: 工具名称

    Returns:
        工具实例或 None
    """
    async with _async_lock:
        return _global_registry.get(name)


def tool(*args, **kwargs):
    """工具装饰器

    便捷的装饰器，自动注册工具。

    Examples:
        ```python
        from app.agent.tools import tool

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
