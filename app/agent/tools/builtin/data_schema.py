"""数据模式工具

对齐 WeKnora99 ToolDataSchema。

获取数据结构信息（列名、类型、统计信息等）。
"""

import io
from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
try:
    import pandas as pd
    _pandas_available = True
except ImportError:
    _pandas_available = False
    pd = None  # type: ignore


def _format_schema(schema_info: dict[str, Any]) -> str:
    """格式化模式信息

    Args:
        schema_info: 模式信息字典

    Returns:
        格式化的模式描述
    """
    parts = []
    parts.append(f"## 数据模式: {schema_info.get('name', 'Unknown')}")
    parts.append("")

    # 基本信息
    parts.append("### 基本信息")
    parts.append(f"- **行数**: {schema_info.get('row_count', 'N/A')}")
    parts.append(f"- **列数**: {schema_info.get('column_count', 'N/A')}")
    parts.append("")

    # 列信息
    parts.append("### 列信息")
    parts.append("| 列名 | 类型 | 非空数 | 唯一值 | 示例 |")
    parts.append("|------|------|--------|--------|------|")

    for col in schema_info.get("columns", []):
        name = col.get("name", "Unknown")
        dtype = col.get("dtype", "Unknown")
        non_null = col.get("non_null", "N/A")
        unique = col.get("unique", "N/A")
        sample = str(col.get("sample", ""))[:30]

        parts.append(f"| {name} | {dtype} | {non_null} | {unique} | {sample} |")

    parts.append("")

    # 数据类型分布
    dtype_counts = schema_info.get("dtype_counts", {})
    if dtype_counts:
        parts.append("### 数据类型分布")
        for dtype, count in dtype_counts.items():
            parts.append(f"- {dtype}: {count} 列")
        parts.append("")

    return "\n".join(parts)


def _get_sample_schema() -> dict[str, Any]:
    """获取示例模式信息用于测试

    Returns:
        模式信息字典
    """
    return {
        "name": "sales_data",
        "row_count": 1000,
        "column_count": 5,
        "columns": [
            {
                "name": "date",
                "dtype": "object",
                "non_null": 1000,
                "unique": 365,
                "sample": "2024-01-01",
            },
            {
                "name": "product",
                "dtype": "object",
                "non_null": 1000,
                "unique": 50,
                "sample": "Product A",
            },
            {
                "name": "amount",
                "dtype": "float64",
                "non_null": 1000,
                "unique": 200,
                "sample": "123.45",
            },
            {
                "name": "region",
                "dtype": "object",
                "non_null": 980,
                "unique": 5,
                "sample": "North",
            },
            {
                "name": "status",
                "dtype": "category",
                "non_null": 950,
                "unique": 3,
                "sample": "completed",
            },
        ],
        "dtype_counts": {
            "object": 3,
            "float64": 1,
            "category": 1,
        },
    }


def _get_schema_from_df(df) -> dict[str, Any]:
    """从 DataFrame 获取模式信息

    Args:
        df: pandas DataFrame

    Returns:
        模式信息字典
    """
    columns = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": df[col].notna().sum(),
            "unique": df[col].nunique(),
        }

        # 获取非空样本
        non_null_samples = df[col].dropna().astype(str).unique()
        if len(non_null_samples) > 0:
            col_info["sample"] = non_null_samples[0]
        else:
            col_info["sample"] = ""

        columns.append(col_info)

    # 数据类型分布
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_counts = {str(k): v for k, v in dtype_counts.items()}

    return {
        "name": "data",
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "dtype_counts": dtype_counts,
    }


@tool
async def data_schema(knowledge_id: str) -> str:
    """获取数据结构信息

    返回数据的列信息、类型、统计信息等。

    Args:
        knowledge_id: 知识条目 ID（CSV/Excel 文件）

    Returns:
        格式化的模式描述

    Examples:
        ```python
        result = await data_schema("knowledge-id")
        ```
    """
    logger.info(
        "data_schema_start",
        knowledge_id=knowledge_id,
    )

    if not _pandas_available:
        return "数据模式分析不可用：请安装 pandas: uv add pandas"

    # TODO: 实际实现应该从知识库获取文件内容
    # 目前返回示例数据用于测试
    schema_info = _get_sample_schema()

    try:
        result_text = _format_schema(schema_info)

        logger.info(
            "data_schema_complete",
            knowledge_id=knowledge_id,
            column_count=schema_info["column_count"],
        )

        return result_text

    except Exception as e:
        logger.error("data_schema_failed", knowledge_id=knowledge_id, error=str(e))
        return f"获取模式信息失败: {str(e)}"


@tool
async def data_summary(knowledge_id: str) -> str:
    """获取数据摘要统计

    Args:
        knowledge_id: 知识条目 ID

    Returns:
        数据统计摘要
    """
    if not _pandas_available:
        return "数据统计不可用：请安装 pandas: uv add pandas"

    # TODO: 从知识库获取实际数据
    return "```\n统计信息（示例）\n\n" \
           "          amount\n" \
           "count  1000.000000\n" \
           "mean     123.450000\n" \
           "std       45.678900\n" \
           "min       10.000000\n" \
           "25%       50.000000\n" \
           "50%      100.000000\n" \
           "75%      150.000000\n" \
           "max     999.990000\n" \
           "```"


__all__ = ["data_schema", "data_summary"]
