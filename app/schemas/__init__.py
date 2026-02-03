"""Pydantic 模式定义"""

from app.schemas.agent import (
    AgentConfig,
    AgentCopyRequest,
    AgentCopyResponse,
    AgentDetailResponse,
    AgentListResponse,
    AgentPublic,
    AgentRequest,
    BatchAgentCopyRequest,
    BatchAgentCopyResponse,
)

from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Message,
    SSEEvent,
    StreamChatRequest,
)

from app.schemas.mcp_service import (
    MCPServiceListResponse,
    MCPServiceRequest,
    MCPServiceResponse,
)

from app.schemas.message import (
    MessageListResponse,
    MessageRegenerateRequest,
    MessageResponse,
    MessageSearchResponse,
    MessageUpdate,
)

from app.schemas.model import (
    ModelCreate,
    ModelParameters,
    ModelResponse,
    ModelSource,
    ModelType,
    ModelUpdate,
)

from app.schemas.response import (
    ApiResponse,
    DataResponse,
    PaginatedResponse,
    PaginationMeta,
)

from app.schemas.session import (
    GenerateTitleRequest,
    SessionCreate,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
)

from app.schemas.tenant import (
    ApiKeyResponse,
    TenantItem,
    TenantListResponse,
    TenantSearchRequest,
    TenantSearchResponse,
)

from app.schemas.tool import (
    ToolInfo,
    ToolsListResponse,
)

from app.schemas.web_search import (
    WebSearchCompressRequest,
    WebSearchCompressResponse,
    WebSearchConfig,
    WebSearchProviderInfo,
    WebSearchProvidersResponse,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)

__all__ = [
    "ApiResponse",
    "DataResponse",
    "PaginatedResponse",
    "PaginationMeta",
    "ChatRequest",
    "StreamChatRequest",
    "ChatResponse",
    "Message",
    "ChatHistoryResponse",
    "SSEEvent",
    "AgentConfig",
    "AgentRequest",
    "AgentPublic",
    "AgentDetailResponse",
    "AgentListResponse",
    "AgentCopyRequest",
    "AgentCopyResponse",
    "BatchAgentCopyRequest",
    "BatchAgentCopyResponse",
    "TenantListResponse",
    "ApiKeyResponse",
    "TenantItem",
    "TenantSearchRequest",
    "TenantSearchResponse",
    "MCPServiceRequest",
    "MCPServiceResponse",
    "MCPServiceListResponse",
    "ToolInfo",
    "ToolsListResponse",
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionDetailResponse",
    "SessionListResponse",
    "GenerateTitleRequest",
    "MessageResponse",
    "MessageUpdate",
    "MessageRegenerateRequest",
    "MessageListResponse",
    "MessageSearchResponse",
    "ModelType",
    "ModelSource",
    "ModelParameters",
    "ModelCreate",
    "ModelUpdate",
    "ModelResponse",
    "WebSearchConfig",
    "WebSearchResult",
    "WebSearchProviderInfo",
    "WebSearchRequest",
    "WebSearchResponse",
    "WebSearchProvidersResponse",
    "WebSearchCompressRequest",
    "WebSearchCompressResponse",
]
