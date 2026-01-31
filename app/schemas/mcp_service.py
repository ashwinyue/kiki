"""MCP 服务相关模式"""

from pydantic import BaseModel, Field


class MCPServiceRequest(BaseModel):
    """MCP 服务创建/更新请求"""

    name: str = Field(..., min_length=1, max_length=255, description="服务名称")
    description: str | None = Field(None, description="服务描述")
    enabled: bool = Field(True, description="是否启用")
    transport_type: str = Field("stdio", description="传输方式：stdio/http/sse")
    url: str | None = Field(None, description="HTTP/SSE URL")
    headers: dict | None = Field(None, description="请求头配置")
    auth_config: dict | None = Field(None, description="鉴权配置")
    advanced_config: dict | None = Field(None, description="高级配置")
    stdio_config: dict | None = Field(None, description="STDIO 配置")
    env_vars: dict | None = Field(None, description="环境变量")


class MCPServiceResponse(BaseModel):
    """MCP 服务响应"""

    id: int
    tenant_id: int | None
    name: str
    description: str | None
    enabled: bool
    transport_type: str
    url: str | None
    headers: dict | None
    auth_config: dict | None
    advanced_config: dict | None
    stdio_config: dict | None
    env_vars: dict | None
    created_at: str


class MCPServiceListResponse(BaseModel):
    """MCP 服务列表响应"""

    items: list[MCPServiceResponse]
    total: int
