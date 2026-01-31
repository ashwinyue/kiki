"""分块提取任务处理器

对齐 WeKnora99 的分块提取功能 (TypeChunkExtract)
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.tasks.handlers.base import task_context

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

    task_id = f"chunk_{chunk_id[:8]}"

    from app.infra.database import get_session_factory

    session_factory = get_session_factory()

    async with session_factory() as session:
        async with task_context(celery_task, session, task_id, tenant_id) as handler:
            await handler.create_task(
                task_type="chunk:extract",
                payload=payload,
                title="分块内容提取",
                business_id=chunk_id,
                business_type="chunk",
            )

            await handler.mark_started()

            try:
                # TODO: 实现分块提取逻辑
                # 1. 获取分块内容
                await handler.update_progress(25, "获取分块内容")
                # 2. 调用 LLM 提取
                await handler.update_progress(50, "LLM 提取中")
                # 3. 保存提取结果
                await handler.update_progress(75, "保存结果")

                result = {
                    "status": "completed",
                    "chunk_id": chunk_id,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise
