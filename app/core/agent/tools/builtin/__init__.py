"""内置示例工具

提供常用工具示例，供使用者参考和扩展。

包含:
- search_database: 数据库搜索工具示例
- get_weather: 天气查询工具示例
- calculate: 数学计算工具示例

这些工具会自动注册到全局工具注册表。
"""

from app.core.agent.tools.builtin.calculation import calculate
from app.core.agent.tools.builtin.database import search_database
from app.core.agent.tools.builtin.weather import get_weather

__all__ = [
    "search_database",
    "get_weather",
    "calculate",
]
