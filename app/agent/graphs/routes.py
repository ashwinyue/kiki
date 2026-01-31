"""路由函数

定义图工作流中的路由决策逻辑。
"""

# 导入带有迭代计数检查的路由函数
from app.agent.graphs.chat import route_by_tools

__all__ = ["route_by_tools"]
