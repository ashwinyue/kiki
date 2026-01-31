"""生成任务处理器

对齐 WeKnora99 的问题生成和摘要生成功能
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.tasks.handlers.base import task_context

logger = get_logger(__name__)


async def process_question_generation(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理问题生成任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_base_id = payload.get("knowledge_base_id")
    knowledge_id = payload.get("knowledge_id")
    question_count = payload.get("question_count", 5)

    task_id = f"qgen_{knowledge_id[:8]}"

    from app.infra.database import get_session_factory

    session_factory = get_session_factory()

    async with session_factory() as session:
        async with task_context(celery_task, session, task_id, tenant_id) as handler:
            await handler.create_task(
                task_type="question:generation",
                payload=payload,
                title=f"生成 {question_count} 个问题",
                business_id=knowledge_id,
                business_type="knowledge",
            )

            await handler.mark_started()

            try:
                # TODO: 实现问题生成逻辑
                await handler.update_progress(50, "生成问题中")

                result = {
                    "status": "completed",
                    "knowledge_id": knowledge_id,
                    "question_count": question_count,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


async def process_summary_generation(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理摘要生成任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_base_id = payload.get("knowledge_base_id")
    knowledge_id = payload.get("knowledge_id")

    task_id = f"sum_{knowledge_id[:8]}"

    from app.infra.database import get_session_factory

    session_factory = get_session_factory()

    async with session_factory() as session:
        async with task_context(celery_task, session, task_id, tenant_id) as handler:
            await handler.create_task(
                task_type="summary:generation",
                payload=payload,
                title="生成知识摘要",
                business_id=knowledge_id,
                business_type="knowledge",
            )

            await handler.mark_started()

            try:
                # TODO: 实现摘要生成逻辑
                await handler.update_progress(50, "生成摘要中")

                result = {
                    "status": "completed",
                    "knowledge_id": knowledge_id,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise
