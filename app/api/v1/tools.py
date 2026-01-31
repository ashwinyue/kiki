"""工具管理 API

提供工具查询、列表等接口。
"""

from fastapi import APIRouter
from starlette.requests import Request as StarletteRequest

from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.tool import ToolInfo, ToolsListResponse
from app.services.tool_service import ToolService

logger = get_logger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("", response_model=ToolsListResponse)
@limiter.limit(RateLimit.API)
async def list_tools_api(request: StarletteRequest) -> ToolsListResponse:
    """列出所有已注册的工具

    Returns:
        ToolsListResponse: 工具列表
    """
    result = ToolService.list_tools()
    logger.info("tools_listed", count=result.count)
    return result


@router.get("/{tool_name}", response_model=ToolInfo)
@limiter.limit(RateLimit.API)
async def get_tool_info(request: StarletteRequest, tool_name: str) -> ToolInfo:
    """获取指定工具的详细信息

    Args:
        tool_name: 工具名称

    Returns:
        ToolInfo: 工具信息
    """
    return ToolService.get_tool_info(tool_name)
