"""Pydantic 模式定义"""

# 统一响应
# Agent
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

# 聊天
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Message,
    SSEEvent,
    StreamChatRequest,
)

# MCP 服务
from app.schemas.mcp_service import (
    MCPServiceListResponse,
    MCPServiceRequest,
    MCPServiceResponse,
)

# 消息
from app.schemas.message import (
    MessageListResponse,
    MessageRegenerateRequest,
    MessageResponse,
    MessageSearchResponse,
    MessageUpdate,
)

# 模型
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

# 会话
from app.schemas.session import (
    GenerateTitleRequest,
    SessionCreate,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
)

# 租户
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
