"""表格摘要任务处理器

对齐 WeKnora99 的表格摘要功能 (TypeDataTableSummary)
"""

import re
from datetime import UTC, datetime

from app.infra.database import async_session_factory
from app.observability.logging import get_logger
from app.repositories.knowledge import ChunkRepository
from app.tasks.handlers.base import TaskHandler, task_context

logger = get_logger(__name__)


async def process_datatable_summary(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理表格摘要任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_id = payload.get("knowledge_id")
    chunk_id = payload.get("chunk_id")
    model_id = payload.get("model_id")
    summary_type = payload.get("summary_type", "full")  # full, concise, structured

    task_id = f"table_{chunk_id[:8]}"

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="datatable:summary",
                payload=payload,
                title="表格数据摘要",
                business_id=chunk_id,
                business_type="chunk",
            )

            await handler.mark_started()

            try:
                # 初始化仓储
                chunk_repo = ChunkRepository(session)

                # 获取分块内容
                await handler.update_progress(25, "解析表格结构")

                chunk = await chunk_repo.get_by_tenant(chunk_id, tenant_id)
                if not chunk:
                    raise ValueError(f"Chunk {chunk_id} not found")

                content = chunk.content

                # 解析表格内容
                table_data = _parse_table_content(content)
                if not table_data:
                    raise ValueError("无法解析表格内容")

                # 生成摘要
                await handler.update_progress(50, "生成摘要")

                summary = await _generate_table_summary(
                    session,
                    table_data,
                    summary_type,
                    model_id,
                    tenant_id,
                )

                # 保存结果
                await handler.update_progress(75, "保存结果")

                metadata = chunk.meta_data or {}
                table_summary = metadata.get("table_summary", {})
                table_summary.update({
                    "summary": summary,
                    "summary_type": summary_type,
                    "generated_at": str(datetime.now(UTC)),
                    "rows": table_data.get("rows", 0),
                    "columns": table_data.get("columns", []),
                })
                metadata["table_summary"] = table_summary

                await chunk_repo.update_fields(
                    chunk_id=chunk_id,
                    tenant_id=tenant_id,
                    meta_data=metadata,
                )

                result = {
                    "status": "completed",
                    "chunk_id": chunk_id,
                    "summary": summary,
                    "table_data": table_data,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


def _parse_table_content(content: str) -> dict | None:
    """解析表格内容

    Args:
        content: 原始内容

    Returns:
        表格数据结构
    """
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return None

    # 尝试解析 CSV 格式
    if "," in lines[0] and "\t" not in lines[0]:
        return _parse_csv_table(lines)

    # 尝试解析制表符分隔格式
    if "\t" in lines[0]:
        return _parse_tsv_table(lines)

    # 尝试解析 Markdown 表格
    if "|" in lines[0] or "|---" in lines[1]:
        return _parse_markdown_table(lines)

    # 默认返回原始内容
    return {
        "raw": content,
        "rows": len(lines),
        "columns": [],
        "data": [],
    }


def _parse_csv_table(lines: list[str]) -> dict:
    """解析 CSV 格式表格"""
    import csv
    from io import StringIO

    reader = csv.reader(StringIO("\n".join(lines)))
    rows = list(reader)

    if not rows:
        return {"rows": 0, "columns": [], "data": []}

    headers = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []

    return {
        "format": "csv",
        "headers": headers,
        "columns": headers,
        "rows": len(data),
        "data": data,
    }


def _parse_tsv_table(lines: list[str]) -> dict:
    """解析制表符分隔格式表格"""
    headers = lines[0].split("\t")
    data = [line.split("\t") for line in lines[1:] if line.strip()]

    return {
        "format": "tsv",
        "headers": headers,
        "columns": headers,
        "rows": len(data),
        "data": data,
    }


def _parse_markdown_table(lines: list[str]) -> dict:
    """解析 Markdown 格式表格"""
    # 提取表头和分隔行
    header_line = None
    separator_line = None

    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            if header_line is None:
                header_line = i
            elif separator_line is None and "---" in line:
                separator_line = i
                break

    if header_line is None:
        return {"rows": 0, "columns": [], "data": []}

    # 解析表头
    headers = [cell.strip() for cell in lines[header_line].strip("|").split("|")]

    # 解析数据行
    data = []
    for line in lines[separator_line + 1:]:
        if line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == len(headers):
                data.append(cells)

    return {
        "format": "markdown",
        "headers": headers,
        "columns": headers,
        "rows": len(data),
        "data": data,
    }


async def _generate_table_summary(
    session,
    table_data: dict,
    summary_type: str,
    model_id: str | None,
    tenant_id: int,
) -> dict:
    """生成表格摘要

    Args:
        session: 数据库会话
        table_data: 表格数据
        summary_type: 摘要类型
        model_id: 模型 ID
        tenant_id: 租户 ID

    Returns:
        摘要结果
    """
    from app.services.model_service import ModelService

    model_service = ModelService(session)

    # 获取 LLM 模型
    try:
        model = await model_service.get_default_model(tenant_id, "llm")
    except Exception:
        logger.warning(
            "no_llm_model",
            tenant_id=tenant_id,
        )
        # 返回基于规则的摘要
        return _generate_rule_based_summary(table_data, summary_type)

    # 构建表格内容描述
    rows = table_data.get("rows", 0)
    columns = table_data.get("columns", [])
    data = table_data.get("data", [])

    # 构建 prompt
    if summary_type == "structured":
        prompt = _build_structured_summary_prompt(rows, columns, data)
    elif summary_type == "concise":
        prompt = _build_concise_summary_prompt(rows, columns, data)
    else:
        prompt = _build_full_summary_prompt(rows, columns, data)

    try:
        # 调用 LLM（如果需要）
        # response = await model_service.call_llm(
        #     tenant_id=tenant_id,
        #     model_id=model.id,
        #     messages=[{"role": "user", "content": prompt}],
        # )

        # 生成摘要（模拟）
        return _generate_rule_based_summary(table_data, summary_type)

    except Exception as e:
        logger.warning(
            "table_summary_failed",
            error=str(e),
        )
        return _generate_rule_based_summary(table_data, summary_type)


def _generate_rule_based_summary(table_data: dict, summary_type: str) -> dict:
    """基于规则生成表格摘要

    Args:
        table_data: 表格数据
        summary_type: 摘要类型

    Returns:
        摘要结果
    """
    rows = table_data.get("rows", 0)
    columns = table_data.get("columns", [])
    data = table_data.get("data", [])

    if summary_type == "structured":
        # 结构化摘要
        column_info = [
            {
                "name": col,
                "type": _infer_column_type([row[i] for row in data if i < len(row)]),
                "sample": data[0][i] if data and i < len(data[0]) else "",
            }
            for i, col in enumerate(columns)
        ]

        return {
            "type": "structured",
            "row_count": rows,
            "column_count": len(columns),
            "columns": column_info,
            "description": f"表格包含 {rows} 行数据，{len(columns)} 列：{', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}",
        }

    elif summary_type == "concise":
        # 简洁摘要
        return {
            "type": "concise",
            "row_count": rows,
            "column_count": len(columns),
            "columns": columns[:5],
            "description": f"表格：{rows} 行 × {len(columns)} 列",
        }

    else:
        # 完整摘要
        return {
            "type": "full",
            "row_count": rows,
            "column_count": len(columns),
            "columns": columns,
            "column_types": {
                col: _infer_column_type([row[i] for row in data if i < len(row)])
                for i, col in enumerate(columns)
            },
            "description": f"表格包含 {rows} 行数据，{len(columns)} 列：{', '.join(columns)}",
            "data_preview": data[:5] if data else [],
        }


def _infer_column_type(values: list[str]) -> str:
    """推断列的数据类型

    Args:
        values: 列值列表

    Returns:
        数据类型描述
    """
    if not values:
        return "unknown"

    # 检查是否是数字
    try:
        float(values[0])
        return "number"
    except ValueError:
        pass

    # 检查是否是日期
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{4}/\d{2}/\d{2}",
        r"\d{2}/\d{2}/\d{4}",
    ]
    for pattern in date_patterns:
        if re.match(pattern, values[0]):
            return "date"

    # 检查是否是百分比
    if "%" in values[0]:
        return "percentage"

    # 默认为文本
    return "text"


def _build_full_summary_prompt(rows: int, columns: list[str], data: list[list[str]]) -> str:
    """构建完整摘要的 prompt"""
    preview_data = data[:10] if data else []
    preview_lines = "\n".join([", ".join(row) for row in preview_data])

    return f"""请对以下表格数据生成详细摘要。

表格信息：
- 行数: {rows}
- 列数: {len(columns)}
- 列名: {', '.join(columns)}

数据预览：
{preview_lines}

请返回以下格式的 JSON：
{{
    "description": "表格的详细描述",
    "column_types": {{"列名": "数据类型", ...}},
    "key_insights": ["洞察1", "洞察2", "洞察3"],
    "data_quality": "数据质量描述"
}}

请只返回 JSON。"""


def _build_concise_summary_prompt(rows: int, columns: list[str], data: list[list[str]]) -> str:
    """构建简洁摘要的 prompt"""
    return f"""请用一句话描述以下表格。

表格：{rows} 行 × {len(columns)} 列，列名：{', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}

请直接返回描述文字。"""


def _build_structured_summary_prompt(rows: int, columns: list[str], data: list[list[str]]) -> str:
    """构建结构化摘要的 prompt"""
    preview_data = data[:5] if data else []

    return f"""请提取表格的结构化信息。

表格信息：
- 行数: {rows}
- 列数: {len(columns)}
- 列名: {', '.join(columns)}

数据示例：
{preview_data}

请返回以下格式的 JSON：
{{
    "columns": [
        {{"name": "列名", "type": "数据类型", "sample": "示例值"}}
    ],
    "total_rows": {rows}
}}

请只返回 JSON。"""
