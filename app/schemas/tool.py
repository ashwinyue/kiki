"""工具相关模式"""

from pydantic import BaseModel, Field


class ToolInfo(BaseModel):
    """工具信息"""

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    args_schema: str | None = Field(None, description="参数架构")


class ToolsListResponse(BaseModel):
    """工具列表响应"""

    tools: list[ToolInfo] = Field(default_factory=list, description="工具列表")
    count: int = Field(..., description="工具数量")
