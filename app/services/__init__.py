"""服务层模块

提供业务逻辑封装，将业务逻辑从 API 路由中分离。

使用延迟导入避免循环依赖。
"""

__all__ = [
    "AuthService",
    "get_auth_service",
    "TenantService",
    "ApiKeyManagementService",
    "get_api_key_management_service",
    "McpServiceService",
    "get_mcp_service_service",
    "WebSearchService",
    "get_web_search_service",
    "ElasticsearchService",
    "KnowledgeCloner",
    "CopyProgress",
    "CopyTaskStatus",
    "create_copy_task",
    "get_copy_progress",
    "execute_copy_task",
]


def __getattr__(name: str):
    """延迟导入服务模块，避免循环依赖

    Args:
        name: 要导入的名称

    Returns:
        导入的对象
    """
    if name == "AuthService" or name == "get_auth_service":
        from app.services.auth import AuthService, get_auth_service
        if name == "AuthService":
            return AuthService
        return get_auth_service

    if name == "TenantService":
        from app.services.tenant import TenantService
        return TenantService

    if name == "ApiKeyManagementService" or name == "get_api_key_management_service":
        from app.services.api_key_management_service import (
            ApiKeyManagementService,
            get_api_key_management_service,
        )
        if name == "ApiKeyManagementService":
            return ApiKeyManagementService
        return get_api_key_management_service

    if name == "McpServiceService" or name == "get_mcp_service_service":
        from app.services.mcp_service_service import McpServiceService, get_mcp_service_service
        if name == "McpServiceService":
            return McpServiceService
        return get_mcp_service_service

    if name == "WebSearchService" or name == "get_web_search_service":
        from app.services.web_search import WebSearchService, get_web_search_service
        if name == "WebSearchService":
            return WebSearchService
        return get_web_search_service

    if name == "ElasticsearchService":
        from app.services.elasticsearch_service import ElasticsearchService
        return ElasticsearchService

    if name in (
        "KnowledgeCloner",
        "CopyProgress",
        "CopyTaskStatus",
        "create_copy_task",
        "get_copy_progress",
        "execute_copy_task",
    ):
        from app.services.knowledge_clone import (
            CopyProgress,
            CopyTaskStatus,
            KnowledgeCloner,
            create_copy_task,
            execute_copy_task,
            get_copy_progress,
        )
        return {
            "KnowledgeCloner": KnowledgeCloner,
            "CopyProgress": CopyProgress,
            "CopyTaskStatus": CopyTaskStatus,
            "create_copy_task": create_copy_task,
            "get_copy_progress": get_copy_progress,
            "execute_copy_task": execute_copy_task,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
