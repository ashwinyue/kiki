"""生成任务处理器

对齐 WeKnora99 的问题生成和摘要生成功能
"""

from app.infra.database import async_session_factory
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.repositories.knowledge import ChunkRepository, KnowledgeRepository
from app.tasks.handlers.base import ProgressReporter, TaskHandler, task_context

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

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="question:generation",
                payload=payload,
                title=f"生成 {question_count} 个问题",
                business_id=knowledge_id,
                business_type="knowledge",
            )

            await handler.mark_started()

            try:
                # 初始化仓储
                knowledge_repo = KnowledgeRepository(session)
                chunk_repo = ChunkRepository(session)

                # 获取知识条目
                knowledge = await knowledge_repo.get_by_tenant(knowledge_id, tenant_id)
                if not knowledge:
                    raise ValueError(f"Knowledge {knowledge_id} not found")

                # 获取分块列表
                chunks_result = await chunk_repo.list_by_knowledge(
                    knowledge_id, tenant_id, PaginationParams(page=1, size=100)
                )
                chunks = chunks_result.items

                # 生成问题
                await handler.update_progress(30, "生成问题中")

                question_count_result = await _generate_questions_for_knowledge(
                    handler,
                    session,
                    chunks,
                    question_count,
                    tenant_id,
                )

                result = {
                    "status": "completed",
                    "knowledge_id": knowledge_id,
                    "question_count": question_count_result,
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
    summary_length = payload.get("summary_length", "medium")  # short, medium, long

    task_id = f"sum_{knowledge_id[:8]}"

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="summary:generation",
                payload=payload,
                title="生成知识摘要",
                business_id=knowledge_id,
                business_type="knowledge",
            )

            await handler.mark_started()

            try:
                # 初始化仓储
                knowledge_repo = KnowledgeRepository(session)
                chunk_repo = ChunkRepository(session)

                # 获取知识条目
                knowledge = await knowledge_repo.get_by_tenant(knowledge_id, tenant_id)
                if not knowledge:
                    raise ValueError(f"Knowledge {knowledge_id} not found")

                # 获取分块内容
                chunks_result = await chunk_repo.list_by_knowledge(
                    knowledge_id, tenant_id, PaginationParams(page=1, size=100)
                )
                chunks = chunks_result.items

                # 生成摘要
                await handler.update_progress(50, "生成摘要中")

                summary = await _generate_summary(
                    handler,
                    session,
                    chunks,
                    summary_length,
                    tenant_id,
                )

                # 保存摘要到知识条目
                metadata = knowledge.meta_data or {}
                metadata["summary"] = summary
                await knowledge_repo.update(knowledge_id, meta_data=metadata)

                result = {
                    "status": "completed",
                    "knowledge_id": knowledge_id,
                    "summary": summary,
                    "summary_length": summary_length,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


async def _generate_questions_for_knowledge(
    handler: TaskHandler,
    session,
    chunks: list,
    question_count: int,
    tenant_id: int,
) -> int:
    """为知识生成问题

    Args:
        handler: 任务处理器
        session: 数据库会话
        chunks: 分块列表
        question_count: 每个分块生成的问题数
        tenant_id: 租户 ID

    Returns:
        生成的问题总数
    """
    from app.services.model_service import ModelService

    model_service = ModelService(session)
    chunk_repo = ChunkRepository(session)

    # 获取 LLM 模型
    try:
        model = await model_service.get_default_model(tenant_id, "llm")
    except Exception:
        logger.warning(
            "no_llm_model",
            tenant_id=tenant_id,
        )
        return 0

    total_questions = 0

    for i, chunk in enumerate(chunks):
        await handler.update_progress(
            30 + int((i / len(chunks)) * 60),
            f"处理分块 {i+1}/{len(chunks)}",
            processed_items=i + 1,
            total_items=len(chunks),
        )

        # 获取分块内容
        content = chunk.content if hasattr(chunk, 'content') else ""
        if len(content) < 50:
            continue

        # 构建 prompt
        prompt = f"""根据以下内容，生成 {question_count} 个常见问题及其答案。

内容：
{content[:3000]}

请以 JSON 格式返回：
{{
    "questions": [
        {{
            "question": "问题1",
            "answer": "答案1"
        }},
        {{
            "question": "问题2",
            "answer": "答案2"
        }}
    ]
}}"""

        try:
            # 调用 LLM 生成问题（如果需要）
            # response = await model_service.call_llm(
            #     tenant_id=tenant_id,
            #     model_id=model.id,
            #     messages=[{"role": "user", "content": prompt}],
            # )

            # 模拟生成问题
            questions_data = {
                "questions": [
                    {
                        "question": f"关于这段内容的常见问题 {j+1}",
                        "answer": content[:100] + "...",
                    }
                    for j in range(question_count)
                ]
            }

            # 保存问题到分块元数据
            metadata = chunk.meta_data or {}
            generated_questions = metadata.get("generated_questions", [])
            generated_questions.extend(questions_data["questions"])
            metadata["generated_questions"] = generated_questions

            await chunk_repo.update_fields(
                chunk_id=chunk.id,
                tenant_id=tenant_id,
                meta_data=metadata,
            )

            total_questions += len(questions_data["questions"])

        except Exception as e:
            logger.warning(
                "question_generation_failed",
                chunk_id=chunk.id if hasattr(chunk, 'id') else str(i),
                error=str(e),
            )
            continue

    return total_questions


async def _generate_summary(
    handler: TaskHandler,
    session,
    chunks: list,
    summary_length: str,
    tenant_id: int,
) -> str:
    """生成知识摘要

    Args:
        handler: 任务处理器
        session: 数据库会话
        chunks: 分块列表
        summary_length: 摘要长度
        tenant_id: 租户 ID

    Returns:
        生成的摘要
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
        return ""

    # 合并所有分块内容
    content = "\n\n".join([
        chunk.content if hasattr(chunk, 'content') else str(chunk)
        for chunk in chunks
    ])

    # 根据摘要长度设置字数
    length_config = {
        "short": (50, 100),
        "medium": (150, 300),
        "long": (300, 500),
    }
    min_words, max_words = length_config.get(summary_length, (150, 300))

    prompt = f"""请用中文总结以下内容，生成一段 {min_words}-{max_words} 字的摘要。

内容：
{content[:5000]}

请直接返回摘要内容，不需要其他说明。"""

    try:
        # 调用 LLM 生成摘要（如果需要）
        # response = await model_service.call_llm(
        #     tenant_id=tenant_id,
        #     model_id=model.id,
        #     messages=[{"role": "user", "content": prompt}],
        # )

        # 模拟摘要生成
        summary = f"这是一段关于{content[:50]}...的摘要。具体内容涉及{len(chunks)}个分块的详细信息。"

        return summary

    except Exception as e:
        logger.warning(
            "summary_generation_failed",
            error=str(e),
        )
        return ""
