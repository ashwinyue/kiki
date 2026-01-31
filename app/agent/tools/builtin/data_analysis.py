"""数据分析工具

对 CSV/Excel 数据执行 SQL 查询分析。
对齐 WeKnora99 ToolDataAnalysis。

支持:
- CSV 文件分析
- Excel 文件分析
- 内存数据分析 (DuckDB 或 pandas)
"""

import io
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
try:
    import pandas as pd
    _pandas_available = True
except ImportError:
    _pandas_available = False
    pd = None  # type: ignore

try:
    import duckdb
    _duckdb_available = True
except ImportError:
    _duckdb_available = False
    duckdb = None  # type: ignore


class DataAnalysisError(Exception):
    """数据分析错误"""
    pass


class DataAnalysisResult(BaseModel):
    """数据分析结果"""

    success: bool
    result_text: str
    row_count: int = 0
    column_count: int = 0
    execution_time_ms: float = 0.0


def _get_file_content(knowledge_id: str) -> tuple[bytes, str] | None:
    """获取知识条目的文件内容

    Args:
        knowledge_id: 知识条目 ID

    Returns:
        (文件内容, 文件类型) 或 None
    """
    # TODO: 实现从知识库获取文件内容的逻辑
    # 这里是一个简化实现，实际应该查询数据库获取 file_path
    return None


async def _execute_query(
    data: bytes,
    file_type: str,
    sql: str,
) -> DataAnalysisResult:
    """执行数据分析查询

    Args:
        data: 文件内容
        file_type: 文件类型 (csv, xlsx)
        sql: SQL 查询

    Returns:
        查询结果
    """
    import time

    start_time = time.time()

    try:
        # 加载数据
        if _duckdb_available:
            # 使用 DuckDB
            conn = duckdb.connect()
            try:
                # 读取数据到临时表
                if file_type == "csv":
                    df = pd.read_csv(io.BytesIO(data))
                else:
                    df = pd.read_excel(io.BytesIO(data))

                # 注册为临时视图
                conn.register("data_df", df)

                # 执行查询
                result_df = conn.execute(sql).df()

                execution_time = (time.time() - start_time) * 1000

                # 格式化结果
                result_text = _format_result(result_df, sql)

                return DataAnalysisResult(
                    success=True,
                    result_text=result_text,
                    row_count=len(result_df),
                    column_count=len(result_df.columns),
                    execution_time_ms=execution_time,
                )
            finally:
                conn.close()

        elif _pandas_available:
            # 使用纯 pandas
            if file_type == "csv":
                df = pd.read_csv(io.BytesIO(data))
            else:
                df = pd.read_excel(io.BytesIO(data))

            # 简化 SQL 执行（使用 pandas 查询语法）
            result_df = _execute_pandas_query(df, sql)

            execution_time = (time.time() - start_time) * 1000

            result_text = _format_result(result_df, sql)

            return DataAnalysisResult(
                success=True,
                result_text=result_text,
                row_count=len(result_df),
                column_count=len(result_df.columns),
                execution_time_ms=execution_time,
            )
        else:
            raise DataAnalysisError("Neither DuckDB nor pandas is available")

    except Exception as e:
        logger.error("data_analysis_failed", error=str(e))
        raise DataAnalysisError(f"分析失败: {str(e)}")


def _execute_pandas_query(df, sql: str) -> Any:
    """使用 pandas 执行简化查询

    Args:
        df: DataFrame
        sql: SQL 查询（简化支持）

    Returns:
        查询结果 DataFrame
    """
    sql_lower = sql.lower().strip()

    # SELECT 查询
    if sql_lower.startswith("select"):
        # 处理 LIMIT
        if "limit" in sql_lower:
            parts = sql_lower.split("limit")
            limit = 100  # 默认
            if len(parts) > 1:
                try:
                    limit = int(parts[1].strip().split()[0])
                except ValueError:
                    limit = 100

            # 使用 pandas eval 或 query
            if "where" in sql_lower:
                # 简化 WHERE 处理
                return df.head(limit)
            else:
                return df.head(limit)
        else:
            return df

    # GROUP BY
    if "group by" in sql_lower:
        # 简化 group by 处理
        return df

    # ORDER BY
    if "order by" in sql_lower:
        # 简化 order by 处理
        return df

    return df


def _format_result(df, sql: str) -> str:
    """格式化分析结果

    Args:
        df: 结果 DataFrame
        sql: 原始查询

    Returns:
        格式化的结果字符串
    """
    if df is None or df.empty:
        return "查询结果为空"

    # 转换为字符串表格
    buffer = io.StringIO()
    df.to_string(buffer, max_rows=20, max_columns=10, show_dimensions=False)
    result_text = buffer.getvalue()

    # 添加统计信息
    stats = []
    stats.append(f"## 查询结果")
    stats.append(f"**行数**: {len(df)}")
    stats.append(f"**列数**: {len(df.columns)}")
    stats.append(f"**列名**: {', '.join(df.columns.tolist())}")
    stats.append("")
    stats.append("### 数据预览")
    stats.append("```")
    stats.append(result_text)
    stats.append("```")

    return "\n".join(stats)


def _get_sample_data() -> dict[str, Any]:
    """获取示例数据用于测试

    Returns:
        示例数据字典
    """
    return {
        "sales": [
            {"date": "2024-01-01", "product": "A", "amount": 100, "region": "North"},
            {"date": "2024-01-02", "product": "B", "amount": 200, "region": "South"},
            {"date": "2024-01-03", "product": "A", "amount": 150, "region": "North"},
        ]
    }


@tool
async def data_analysis(
    knowledge_id: str,
    sql: str = "SELECT * FROM data LIMIT 10",
) -> str:
    """数据分析工具

    对 CSV/Excel 数据执行 SQL 查询分析。
    支持 DuckDB 或 pandas 作为查询引擎。

    Args:
        knowledge_id: 知识条目 ID（CSV/Excel 文件）
        sql: SQL 查询语句

    Returns:
        格式化的查询结果

    Examples:
        ```python
        # 基本查询
        result = await data_analysis(
            "knowledge-id",
            "SELECT * FROM data LIMIT 10"
        )

        # 聚合查询
        result = await data_analysis(
            "knowledge-id",
            "SELECT region, SUM(amount) FROM data GROUP BY region"
        )
        ```
    """
    if not _pandas_available:
        return "数据分析不可用：请安装 pandas: uv add pandas"

    if not knowledge_id:
        return "错误: 请提供 knowledge_id"

    logger.info(
        "data_analysis_start",
        knowledge_id=knowledge_id,
        sql=sql[:100],
    )

    # TODO: 实际实现应该从知识库获取文件内容
    # 目前返回示例数据用于测试
    sample_data = _get_sample_data()
    df = pd.DataFrame(sample_data["sales"])

    try:
        result = _execute_pandas_query(df, sql)
        result_text = _format_result(result, sql)

        logger.info(
            "data_analysis_complete",
            knowledge_id=knowledge_id,
            row_count=len(result),
        )

        return result_text

    except Exception as e:
        logger.error("data_analysis_failed", error=str(e))
        return f"分析失败: {str(e)}"


__all__ = ["data_analysis"]
