"""服务层模块

提供业务逻辑封装，将业务逻辑从 API 路由中分离。

使用延迟导入避免循环依赖。
"""

__all__ = [
    # 认证服务
    "AuthService",
    "get_auth_service",
    # 租户服务
    "TenantService",
    # API Key 管理
    "ApiKeyManagementService",
    "get_api_key_management_service",
    # MCP 服务
    "McpServiceService",
    "get_mcp_service",
    # Web 搜索
    "WebSearchService",
    "get_web_search_service",
    # 流式响应服务
    "StreamContinuationService",
    "get_stream_continuation_service",
    # 系统服务
    "SystemService",
    # 会话服务
    "SessionService",
    # 消息服务
    "MessageService",
    # 会话状态管理
    "SessionStateManager",
    # 占位符服务
    "PlaceholderService",
    # Agent 克隆服务
    "AgentCloner",
]

def __getattr__(name: str):
    """延迟导入服务模块，避免循环依赖

    Args:
        name: 要导入的名称

    Returns:
        导入的对象
    """
    # AuthService
    if name == "AuthService" or name == "get_auth_service":
        from app.services.core.auth import AuthService, get_auth_service

        if name == "AuthService":
            return AuthService
        return get_auth_service

    # TenantService
    if name == "TenantService":
        from app.services.core.tenant import TenantService

        return TenantService

    # ApiKeyManagementService
    if name == "ApiKeyManagementService" or name == "get_api_key_management_service":
        from app.services.agent.api_key_management_service import (
            ApiKeyManagementService,
            get_api_key_management_service,
        )
        if name == "ApiKeyManagementService":
            return ApiKeyManagementService
        return get_api_key_management_service

    # McpServiceService
    if name == "McpServiceService" or name == "get_mcp_service":
        from app.services.agent.mcp_service import (
            McpServiceService,
        )
        from app.services.agent.mcp_service import (
            get_mcp_service as get_mcp_service_func,
        )
        if name == "McpServiceService":
            return McpServiceService
        return get_mcp_service_func

    # WebSearchService
    if name == "WebSearchService" or name == "get_web_search_service":
        from app.services.web.web_search import (
            WebSearchService,
            get_web_search_service,
        )
        if name == "WebSearchService":
            return WebSearchService
        return get_web_search_service

    # StreamContinuationService
    if name == "StreamContinuationService" or name == "get_stream_continuation_service":
        from app.agent.streaming.service import (
            StreamContinuationService,
            get_stream_continuation_service,
        )
        if name == "StreamContinuationService":
            return StreamContinuationService
        return get_stream_continuation_service

    # SystemService
    if name == "SystemService":
        from app.services.core.system_service import SystemService

        return SystemService

    # SessionService
    if name == "SessionService":
        from app.services.core.session_service import SessionService

        return SessionService

    # MessageService（已废弃，功能已迁移到 repositories）
    if name == "MessageService":
        raise NotImplementedError("MessageService 已移除，请使用 repositories 层直接访问")

    # SessionStateManager
    if name == "SessionStateManager":
        from app.services.core.session_state import SessionStateManager

        return SessionStateManager

    # PlaceholderService
    if name == "PlaceholderService":
        from app.services.placeholder_service import PlaceholderService

        return PlaceholderService

    # AgentCloner
    if name == "AgentCloner":
        from app.services.agent_clone import AgentCloner

        return AgentCloner

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
