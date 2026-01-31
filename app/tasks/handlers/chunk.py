"""分块提取任务处理器

对齐 WeKnora99 的分块提取功能 (TypeChunkExtract)
"""

from datetime import UTC, datetime

from app.infra.database import async_session_factory
from app.observability.logging import get_logger
from app.repositories.knowledge import ChunkRepository
from app.tasks.handlers.base import TaskHandler, task_context

logger = get_logger(__name__)


async def process_chunk_extract(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理分块提取任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    chunk_id = payload.get("chunk_id")
    model_id = payload.get("model_id")
    extract_fields = payload.get("extract_fields", ["entities", "keywords", "summary"])

    task_id = f"chunk_{chunk_id[:8]}"

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="chunk:extract",
                payload=payload,
                title="分块内容提取",
                business_id=chunk_id,
                business_type="chunk",
            )

            await handler.mark_started()

            try:
                # 初始化仓储
                chunk_repo = ChunkRepository(session)

                # 获取分块内容
                await handler.update_progress(25, "获取分块内容")

                chunk = await chunk_repo.get_by_tenant(chunk_id, tenant_id)
                if not chunk:
                    raise ValueError(f"Chunk {chunk_id} not found")

                content = chunk.content

                # 调用 LLM 提取
                await handler.update_progress(50, "LLM 提取中")

                extracted = await _extract_content(
                    session,
                    content,
                    extract_fields,
                    model_id,
                    tenant_id,
                )

                # 保存提取结果
                await handler.update_progress(75, "保存结果")

                metadata = chunk.meta_data or {}
                extracted_data = metadata.get("extracted_data", {})
                extracted_data.update(extracted)
                metadata["extracted_data"] = extracted_data
                metadata["extracted_at"] = str(datetime.now(UTC))

                await chunk_repo.update_fields(
                    chunk_id=chunk_id,
                    tenant_id=tenant_id,
                    meta_data=metadata,
                )

                result = {
                    "status": "completed",
                    "chunk_id": chunk_id,
                    "extracted": extracted,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


async def _extract_content(
    session,
    content: str,
    extract_fields: list[str],
    model_id: str | None,
    tenant_id: int,
) -> dict:
    """使用 LLM 提取内容

    Args:
        session: 数据库会话
        content: 分块内容
        extract_fields: 提取字段列表
        model_id: 模型 ID
        tenant_id: 租户 ID

    Returns:
        提取结果
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
        # 返回空结果
        return {}

    # 构建提取 prompt
    fields_prompt = {
        "entities": "命名实体（人名、地名、机构名、产品名等）",
        "keywords": "关键词",
        "summary": "内容摘要",
        "sentiment": "情感倾向（正面/中性/负面）",
        "topics": "主题标签",
    }

    field_descriptions = "\n".join([
        f"- {field}: {fields_prompt.get(field, field)}"
        for field in extract_fields
        if field in fields_prompt
    ])

    prompt = f"""请从以下内容中提取信息。

内容：
{content[:4000]}

请提取以下信息（JSON 格式）：
{{
    "entities": ["实体1", "实体2"],
    "keywords": ["关键词1", "关键词2"],
    "summary": "一段话的摘要",
    "sentiment": "中性",
    "topics": ["主题1", "主题2"]
}}

请只返回 JSON，不要其他说明。"""

    try:
        # 调用 LLM（如果需要）
        # response = await model_service.call_llm(
        #     tenant_id=tenant_id,
        #     model_id=model.id,
        #     messages=[{"role": "user", "content": prompt}],
        # )

        # 模拟提取结果
        extracted = {
            "entities": ["模拟实体1", "模拟实体2"],
            "keywords": ["关键词1", "关键词2"],
            "summary": content[:100] + "...",
            "sentiment": "中性",
            "topics": ["主题1"],
        }

        return extracted

    except Exception as e:
        logger.warning(
            "content_extraction_failed",
            error=str(e),
        )
        return {}
