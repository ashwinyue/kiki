"""Pydantic 模式定义"""

# 统一响应
from app.schemas.response import (
    ApiResponse,
    DataResponse,
    PaginatedResponse,
    PaginationMeta,
)

# 聊天
from app.schemas.chat import (
    ChatRequest,
    StreamChatRequest,
    ChatResponse,
    Message,
    ChatHistoryResponse,
    SSEEvent,
)

# Agent
from app.schemas.agent import (
    AgentConfig,
    AgentRequest,
    AgentPublic,
    AgentDetailResponse,
    AgentListResponse,
    AgentCopyRequest,
    AgentCopyResponse,
    BatchAgentCopyRequest,
    BatchAgentCopyResponse,
)

# 租户
from app.schemas.tenant import (
    TenantListResponse,
    ApiKeyResponse,
    TenantItem,
    TenantSearchRequest,
    TenantSearchResponse,
)

# MCP 服务
from app.schemas.mcp_service import (
    MCPServiceRequest,
    MCPServiceResponse,
    MCPServiceListResponse,
)

# 工具
from app.schemas.tool import (
    ToolInfo,
    ToolsListResponse,
)

# 会话
from app.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionDetailResponse,
    SessionListResponse,
    GenerateTitleRequest,
)

# 消息
from app.schemas.message import (
    MessageResponse,
    MessageUpdate,
    MessageRegenerateRequest,
    MessageListResponse,
    MessageSearchResponse,
)

# 评估
from app.schemas.evaluation import (
    RunEvaluationRequest,
    RunEvaluationStreamRequest,
    DatasetListItem,
    EvaluationRunResponse,
    EvaluationStatusResponse,
)

# 知识库
from app.schemas.knowledge import (
    ChunkingConfig,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    HybridSearchRequest,
    HybridSearchResult,
    KnowledgeResponse,
)

# 模型
from app.schemas.model import (
    ModelType,
    ModelSource,
    ModelParameters,
    ModelCreate,
    ModelUpdate,
    ModelResponse,
)

# 网络搜索
from app.schemas.web_search import (
    WebSearchConfig,
    WebSearchResult,
    WebSearchProviderInfo,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchProvidersResponse,
    WebSearchCompressRequest,
    WebSearchCompressResponse,
)

# Elasticsearch
from app.schemas.elasticsearch import (
    IndexCreateRequest,
    IndexStatsResponse,
    IndexListResponse,
    DocumentIndexRequest,
    DocumentIndexBatchRequest,
    DocumentUpdateRequest,
    DocumentResponse,
    BulkOperationResponse,
    ElasticsearchSearchRequest,
    HybridSearchRequest,
    RawSearchRequest,
    ElasticsearchSearchResult,
    ElasticsearchSearchResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeToken,
    ElasticsearchConfigResponse,
    ElasticsearchHealthResponse,
)

__all__ = [
    # 统一响应
    "ApiResponse",
    "DataResponse",
    "PaginatedResponse",
    "PaginationMeta",
    # 聊天
    "ChatRequest",
    "StreamChatRequest",
    "ChatResponse",
    "Message",
    "ChatHistoryResponse",
    "SSEEvent",
    # Agent - CRUD
    "AgentConfig",
    "AgentRequest",
    "AgentPublic",
    "AgentDetailResponse",
    "AgentListResponse",
    "AgentCopyRequest",
    "AgentCopyResponse",
    "BatchAgentCopyRequest",
    "BatchAgentCopyResponse",
    # 租户
    "TenantListResponse",
    "ApiKeyResponse",
    "TenantItem",
    "TenantSearchRequest",
    "TenantSearchResponse",
    # MCP 服务
    "MCPServiceRequest",
    "MCPServiceResponse",
    "MCPServiceListResponse",
    # 工具
    "ToolInfo",
    "ToolsListResponse",
    # 会话
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionDetailResponse",
    "SessionListResponse",
    "GenerateTitleRequest",
    # 消息
    "MessageResponse",
    "MessageUpdate",
    "MessageRegenerateRequest",
    "MessageListResponse",
    "MessageSearchResponse",
    # 评估
    "RunEvaluationRequest",
    "RunEvaluationStreamRequest",
    "DatasetListItem",
    "EvaluationRunResponse",
    "EvaluationStatusResponse",
    # 知识库
    "ChunkingConfig",
    "KnowledgeBaseCreate",
    "KnowledgeBaseUpdate",
    "KnowledgeBaseResponse",
    "HybridSearchRequest",
    "HybridSearchResult",
    "KnowledgeResponse",
    # 模型
    "ModelType",
    "ModelSource",
    "ModelParameters",
    "ModelCreate",
    "ModelUpdate",
    "ModelResponse",
    # 网络搜索
    "WebSearchConfig",
    "WebSearchResult",
    "WebSearchProviderInfo",
    "WebSearchRequest",
    "WebSearchResponse",
    "WebSearchProvidersResponse",
    "WebSearchCompressRequest",
    "WebSearchCompressResponse",
    # Elasticsearch
    "IndexCreateRequest",
    "IndexStatsResponse",
    "IndexListResponse",
    "DocumentIndexRequest",
    "DocumentIndexBatchRequest",
    "DocumentUpdateRequest",
    "DocumentResponse",
    "BulkOperationResponse",
    "ElasticsearchSearchRequest",
    "HybridSearchRequest",
    "RawSearchRequest",
    "ElasticsearchSearchResult",
    "ElasticsearchSearchResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "AnalyzeToken",
    "ElasticsearchConfigResponse",
    "ElasticsearchHealthResponse",
]
