"""工具服务

提供工具查询和转换功能。
"""

from fastapi import HTTPException

from app.agent.tools import get_tool, list_tools
from app.schemas.tool import ToolInfo, ToolsListResponse


class ToolService:
    """工具服务"""

    @staticmethod
    def list_tools() -> ToolsListResponse:
        """列出所有已注册的工具

        Returns:
            ToolsListResponse: 工具列表
        """
        tools = list_tools()

        tool_infos = [
            ToolInfo(
                name=tool.name,
                description=tool.description or "",
                args_schema=tool.args_schema.__name__ if tool.args_schema else None,
            )
            for tool in tools
        ]

        return ToolsListResponse(tools=tool_infos, count=len(tool_infos))

    @staticmethod
    def get_tool_info(tool_name: str) -> ToolInfo:
        """获取指定工具的详细信息

        Args:
            tool_name: 工具名称

        Returns:
            ToolInfo: 工具信息

        Raises:
            HTTPException: 如果工具不存在
        """
        tool = get_tool(tool_name)

        if tool is None:
            raise HTTPException(status_code=404, detail=f"工具 '{tool_name}' 不存在")

        return ToolInfo(
            name=tool.name,
            description=tool.description or "",
            args_schema=tool.args_schema.__name__ if tool.args_schema else None,
        )
