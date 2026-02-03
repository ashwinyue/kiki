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

# 工具
from app.schemas.tool import (
    ToolInfo,
    ToolsListResponse,
)

# 网络搜索
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
]
