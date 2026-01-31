"""天气查询工具

示例工具，展示如何创建外部 API 调用工具。
"""

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)


@tool
async def get_weather(location: str) -> str:
    """获取指定位置的天气信息

    查询指定位置的当前天气情况。

    Args:
        location: 位置名称（如 "Beijing", "Shanghai", "New York"）

    Returns:
        天气信息描述字符串

    Examples:
        ```python
        weather = await get_weather("Beijing")
        # 返回: "Beijing 今天天气晴朗，温度 25°C"
        ```
    """
    logger.info("get_weather_called", location=location)

    # 实际实现应该调用天气 API
    # 示例：
    # import httpx
    # async with httpx.AsyncClient() as client:
    #     response = await client.get(f"https://api.weather.com/{location}")
    #     data = response.json()
    #     return format_weather(data)

    return f"{location} 今天天气晴朗，温度 25°C"
