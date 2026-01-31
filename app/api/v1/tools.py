"""工具管理 API

提供工具查询、列表等接口。
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.requests import Request as StarletteRequest

from app.agent.tools import get_tool, list_tools
from app.core.limiter import RateLimit, limiter
from app.observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


class ToolInfo(BaseModel):
    """工具信息"""

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    args_schema: str | None = Field(None, description="参数架构")


class ToolsListResponse(BaseModel):
    """工具列表响应"""

    tools: list[ToolInfo] = Field(default_factory=list, description="工具列表")
    count: int = Field(..., description="工具数量")


@router.get("", response_model=ToolsListResponse)
@limiter.limit(RateLimit.API)
async def list_tools_api(request: StarletteRequest) -> ToolsListResponse:
    """列出所有已注册的工具

    Returns:
        ToolsListResponse: 工具列表
    """
    tools = list_tools()

    tool_infos = []
    for tool in tools:
        tool_infos.append(
            ToolInfo(
                name=tool.name,
                description=tool.description or "",
                args_schema=tool.args_schema.__name__ if tool.args_schema else None,
            )
        )

    logger.info("tools_listed", count=len(tool_infos))

    return ToolsListResponse(tools=tool_infos, count=len(tool_infos))


@router.get("/{tool_name}", response_model=ToolInfo)
@limiter.limit(RateLimit.API)
async def get_tool_info(request: StarletteRequest, tool_name: str) -> ToolInfo:
    """获取指定工具的详细信息

    Args:
        tool_name: 工具名称

    Returns:
        ToolInfo: 工具信息
    """
    tool = get_tool(tool_name)

    if tool is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"工具 '{tool_name}' 不存在")

    return ToolInfo(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.args_schema.__name__ if tool.args_schema else None,
    )
