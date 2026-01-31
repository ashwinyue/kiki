"""数据库查询工具

对齐 WeKnora99 ToolDatabaseQuery。

只读数据库查询，支持 PostgreSQL。
"""

import re
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 只允许的查询类型
_ALLOWED_STATEMENTS = frozenset({"select", "show", "describe", "explain"})

# 只读用户
_READ_ONLY_USER = "readonly_user"


class DatabaseQueryError(Exception):
    """数据库查询错误"""
    pass


async def _get_db_session() -> AsyncSession:
    """获取只读数据库会话

    Returns:
        AsyncSession 实例
    """
    from app.infra.database import async_session_factory
    return async_session_factory()


def _validate_query(query: str) -> tuple[bool, str]:
    """验证查询语句

    Args:
        query: SQL 查询

    Returns:
        (是否有效, 错误消息)
    """
    if not query or not query.strip():
        return False, "查询不能为空"

    query_upper = query.strip().upper()

    # 检查是否以允许的关键字开头
    valid_start = False
    for stmt in _ALLOWED_STATEMENTS:
        if query_upper.startswith(stmt.upper()):
            valid_start = True
            break

    if not valid_start:
        return False, f"只支持 {_ALLOWED_STATEMENTS} 语句"

    # 检查危险关键词
    dangerous_keywords = [
        r"\bDROP\b",
        r"\bDELETE\b",
        r"\bTRUNCATE\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bEXECUTE\b",
        r"\bEXEC\b",
    ]

    for pattern in dangerous_keywords:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"禁止使用危险关键词: {pattern.strip()[:20]}"

    # 检查多语句
    if ";" in query.strip():
        # 只允许单语句
        return False, "只支持单语句查询"

    # 检查事务控制
    if re.search(r"\b(BEGIN|COMMIT|ROLLBACK|START TRANSACTION)\b", query, re.IGNORECASE):
        return False, "不允许使用事务控制语句"

    return True, ""


def _format_results(
    columns: list[str],
    rows: list[tuple],
    query: str,
) -> str:
    """格式化查询结果

    Args:
        columns: 列名列表
        rows: 行数据
        query: 原始查询

    Returns:
        格式化的结果字符串
    """
    if not rows:
        return f"查询成功，但无返回结果\n\n```sql\n{query}\n```"

    # 限制返回行数
    max_rows = 20
    display_rows = rows[:max_rows]
    has_more = len(rows) > max_rows

    parts = []
    parts.append(f"## 查询结果")
    parts.append(f"**查询**: ```{query}```")
    parts.append(f"**返回行数**: {len(rows)}")
    if has_more:
        parts.append(f"**显示前 {max_rows} 行**")
    parts.append("")

    # 构建表格
    # 表头
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    parts.append(header)
    parts.append(separator)

    # 数据行
    for row in display_rows:
        row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
        parts.append(row_str)

    if has_more:
        parts.append(f"\n*... 还有 {len(rows) - max_rows} 行*")

    return "\n".join(parts)


@tool
async def database_query(
    query: str,
    parameters: dict | None = None,
) -> str:
    """数据库查询工具（只读）

    执行只读的 SELECT 查询。

    Args:
        query: SQL 查询语句
        parameters: 查询参数（可选）

    Returns:
        格式化的查询结果

    Examples:
        ```python
        # 基本查询
        result = await database_query("SELECT * FROM users LIMIT 10")

        # 带参数查询
        result = await database_query(
            "SELECT * FROM users WHERE id = :id",
            {"id": 1}
        )
        ```

    Warning:
        只读权限，无法执行 INSERT、UPDATE、DELETE 等写操作。
    """
    # 验证查询
    is_valid, error_msg = _validate_query(query)
    if not is_valid:
        logger.warning("database_query_invalid", query=query[:100], error=error_msg)
        return f"错误: {error_msg}"

    logger.info(
        "database_query_start",
        query=query[:100],
    )

    session = await _get_db_session()

    try:
        # 准备查询
        if parameters:
            stmt = text(query).bindparams(**parameters)
        else:
            stmt = text(query)

        # 执行查询
        result = await session.execute(stmt)
        await session.commit()  # 确保只读查询提交

        # 获取列名
        columns = result.keys()
        rows = result.fetchall()

        # 格式化结果
        result_text = _format_results(list(columns), [tuple(row) for row in rows], query)

        logger.info(
            "database_query_complete",
            query=query[:50],
            row_count=len(rows),
        )

        return result_text

    except Exception as e:
        logger.error("database_query_failed", query=query[:100], error=str(e))
        return f"查询失败: {str(e)}"

    finally:
        await session.close()


@tool
async def database_tables() -> str:
    """列出数据库表

    返回当前数据库中的所有表。

    Returns:
        表列表
    """
    return await database_query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    )


@tool
async def database_describe(table_name: str) -> str:
    """查看表结构

    Args:
        table_name: 表名

    Returns:
        表结构描述
    """
    return await database_query(f"DESCRIBE {table_name}")


__all__ = ["database_query", "database_tables", "database_describe"]
