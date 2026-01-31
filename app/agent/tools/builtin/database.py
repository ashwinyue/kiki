"""数据库搜索工具

示例工具，展示如何创建数据库相关的工具。
"""

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)


@tool
async def search_database(query: str) -> str:
    """搜索数据库

    根据查询语句搜索数据库并返回结果。

    Args:
        query: 搜索查询语句或关键词

    Returns:
        搜索结果字符串

    Examples:
        ```python
        result = await search_database("用户信息")
        # 返回: "数据库搜索结果: 用户信息"
        ```
    """
    logger.info("search_database_called", query=query)

    # 实际实现应该查询数据库
    # 示例：
    # from app.repositories.user import user_repo
    # results = await user_repo.search(query)
    # return format_results(results)

    return f"数据库搜索结果: {query}"
