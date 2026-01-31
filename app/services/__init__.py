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
    # Elasticsearch
    "ElasticsearchService",
    # 知识克隆
    "KnowledgeCloner",
    "CopyProgress",
    "CopyTaskStatus",
    "create_copy_task",
    "get_copy_progress",
    "execute_copy_task",
    # FAQ 服务
    "FAQService",
    "get_faq_service",
    # 流式响应服务
    "StreamContinuationService",
    "get_stream_continuation_service",
    # 文档服务
    "DocumentService",
    "get_document_service",
    # 系统服务
    "SystemService",
    # 会话服务
    "SessionService",
    # 占位符服务
    "PlaceholderService",
    # 模型测试服务
    "ModelTestService",
    # 消息服务
    "MessageService",
    # 初始化服务
    "InitializationService",
    # 知识库服务
    "KnowledgeBaseService",
    "KnowledgeService",
    # 向量服务
    "VectorService",
    # 搜索服务
    "SearchService",
    # 知识搜索服务
    "KnowledgeSearchService",
    # 会话状态管理
    "SessionStateManager",
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

    # ElasticsearchService
    if name == "ElasticsearchService":
        from app.services.search.elasticsearch import ElasticsearchClient

        return ElasticsearchClient

    # KnowledgeCloner & 知识克隆相关
    if name in (
        "KnowledgeCloner",
        "CopyProgress",
        "CopyTaskStatus",
        "create_copy_task",
        "get_copy_progress",
        "execute_copy_task",
    ):
        from app.services.knowledge.knowledge_clone import (
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

    # FAQService
    if name == "FAQService" or name == "get_faq_service":
        from app.services.shared.faq import FAQService, get_faq_service

        if name == "FAQService":
            return FAQService
        return get_faq_service

    # StreamContinuationService
    if name == "StreamContinuationService" or name == "get_stream_continuation_service":
        from app.agent.streaming.service import (
            StreamContinuationService,
            get_stream_continuation_service,
        )
        if name == "StreamContinuationService":
            return StreamContinuationService
        return get_stream_continuation_service

    # DocumentService
    if name == "DocumentService" or name == "get_document_service":
        from app.services.knowledge.document.service import (
            DocumentService,
            get_document_service,
        )
        if name == "DocumentService":
            return DocumentService
        return get_document_service

    # SystemService
    if name == "SystemService":
        from app.services.core.system_service import SystemService

        return SystemService

    # SessionService
    if name == "SessionService":
        from app.services.core.session_service import SessionService

        return SessionService

    # PlaceholderService
    if name == "PlaceholderService":
        from app.services.shared.placeholder_service import PlaceholderService

        return PlaceholderService

    # ModelTestService
    if name == "ModelTestService":
        from app.services.llm.model_test import ModelTestService

        return ModelTestService

    # MessageService
    if name == "MessageService":
        from app.services.shared.message_service import MessageService

        return MessageService

    # InitializationService
    if name == "InitializationService":
        from app.services.knowledge.initialization_service import InitializationService

        return InitializationService

    # KnowledgeBaseService & KnowledgeService
    if name in ("KnowledgeBaseService", "KnowledgeService"):
        from app.services.knowledge.base import (
            KnowledgeBaseService,
            KnowledgeService,
        )
        return {"KnowledgeBaseService": KnowledgeBaseService, "KnowledgeService": KnowledgeService}[name]

    # VectorService
    if name == "VectorService":
        from app.services.shared.vector_service import VectorService

        return VectorService

    # SearchService
    if name == "SearchService":
        from app.services.search.search_service import SearchService

        return SearchService

    # KnowledgeSearchService
    if name == "KnowledgeSearchService":
        from app.services.knowledge.knowledge_search import KnowledgeSearchService

        return KnowledgeSearchService

    # SessionStateManager
    if name == "SessionStateManager":
        from app.services.core.session_state import SessionStateManager

        return SessionStateManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
