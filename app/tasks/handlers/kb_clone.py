"""知识库复制任务处理器

对齐 WeKnora99 的知识库复制功能 (TypeKBClone)
"""

from app.infra.database import async_session_factory
from app.models.knowledge import Chunk, Knowledge, KnowledgeBase
from app.observability.logging import get_logger
from app.repositories.knowledge import (
    ChunkRepository,
    KnowledgeBaseRepository,
    KnowledgeRepository,
)
from app.tasks.handlers.base import ProgressReporter, TaskHandler, task_context

logger = get_logger(__name__)


async def process_kb_clone(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理知识库复制任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    task_id = payload.get("task_id")
    source_id = payload.get("source_id")
    target_id = payload.get("target_id")

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            # 创建任务记录
            await handler.create_task(
                task_type="kb:clone",
                payload=payload,
                title=f"复制知识库 {source_id} -> {target_id}",
                business_id=target_id,
                business_type="knowledge_base",
            )

            # 执行复制
            result = await _execute_kb_clone(
                handler,
                session,
                source_id,
                target_id,
                tenant_id,
            )

            return result


async def _execute_kb_clone(
    handler: TaskHandler,
    session,
    source_id: str,
    target_id: str,
    tenant_id: int,
) -> dict:
    """执行知识库复制

    Args:
        handler: 任务处理器
        session: 数据库会话
        source_id: 源知识库 ID
        target_id: 目标知识库 ID
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    kb_repo = KnowledgeBaseRepository(session)
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)

    await handler.mark_started()

    try:
        # 步骤 1: 验证源知识库存在
        await handler.update_progress(5, "验证源知识库")

        source_kb = await kb_repo.get_by_tenant(source_id, tenant_id)
        if not source_kb:
            raise ValueError(f"Source knowledge base {source_id} not found")

        # 步骤 2: 验证目标知识库存在
        await handler.update_progress(10, "验证目标知识库")

        target_kb = await kb_repo.get_by_tenant(target_id, tenant_id)
        if not target_kb:
            raise ValueError(f"Target knowledge base {target_id} not found")

        logger.info(
            "kb_clone_start",
            source_id=source_id,
            target_id=target_id,
            tenant_id=tenant_id,
        )

        # 步骤 3: 获取源知识库的所有知识条目
        await handler.update_progress(15, "获取源知识库数据")

        from app.repositories.base import PaginationParams

        knowledges_result = await knowledge_repo.list_by_kb(source_id, tenant_id, PaginationParams(page=1, size=1000))
        knowledges = knowledges_result.items
        total_knowledges = knowledges_result.total

        await handler.update_progress(
            20,
            f"准备复制 {total_knowledges} 个知识",
            total_items=total_knowledges,
        )

        logger.info(
            "kb_clone_found_knowledges",
            source_id=source_id,
            count=total_knowledges,
        )

        # 步骤 4: 逐个复制知识
        copied_knowledges = 0
        copied_chunks = 0
        failed_knowledges = 0

        reporter = ProgressReporter(handler, total=total_knowledges or 1, current_step="复制知识")
        reporter.processed = 0

        for i, knowledge in enumerate(knowledges):
            try:
                await handler.update_progress(
                    20 + int((i / max(total_knowledges, 1)) * 70),
                    f"复制知识: {knowledge.title[:30]}",
                    processed_items=i + 1,
                    total_items=total_knowledges,
                )

                # 获取源知识的分块
                chunks_result = await chunk_repo.list_by_knowledge(
                    knowledge.id,
                    tenant_id,
                    PaginationParams(page=1, size=10000),
                )
                source_chunks = chunks_result.items

                # 复制知识条目
                new_knowledge = await _clone_knowledge(
                    session,
                    knowledge,
                    target_id,
                    tenant_id,
                )

                # 复制分块
                if source_chunks:
                    new_chunks = await _clone_chunks(
                        session,
                        source_chunks,
                        new_knowledge.id,
                        target_id,
                        tenant_id,
                    )
                    copied_chunks += len(new_chunks)

                copied_knowledges += 1
                reporter.processed += 1

            except Exception as e:
                logger.error(
                    "kb_clone_knowledge_failed",
                    knowledge_id=knowledge.id,
                    error=str(e),
                )
                failed_knowledges += 1
                reporter.failed += 1
                continue

        # 完成
        result = {
            "status": "completed",
            "source_id": source_id,
            "target_id": target_id,
            "total_knowledges": total_knowledges,
            "copied_knowledges": copied_knowledges,
            "failed_knowledges": failed_knowledges,
            "copied_chunks": copied_chunks,
        }

        await handler.mark_completed(result)

        logger.info(
            "kb_clone_completed",
            source_id=source_id,
            target_id=target_id,
            copied_knowledges=copied_knowledges,
            copied_chunks=copied_chunks,
        )

        return result

    except Exception as e:
        import traceback

        await handler.mark_failed(str(e), traceback.format_exc())
        raise


async def _clone_knowledge(
    session,
    source_knowledge: Knowledge,
    target_kb_id: str,
    tenant_id: int,
) -> Knowledge:
    """复制知识条目

    Args:
        session: 数据库会话
        source_knowledge: 源知识条目
        target_kb_id: 目标知识库 ID
        tenant_id: 租户 ID

    Returns:
        新创建的知识条目
    """
    import uuid
    from datetime import UTC
    from copy import deepcopy

    # 创建新的知识条目
    new_knowledge = Knowledge(
        id=str(uuid.uuid4()),
        knowledge_base_id=target_kb_id,
        tenant_id=tenant_id,
        title=source_knowledge.title,
        type=source_knowledge.type,
        source=source_knowledge.source,
        file_path=source_knowledge.file_path,
        file_name=source_knowledge.file_name,
        file_size=source_knowledge.file_size,
        file_type=source_knowledge.file_type,
        content=source_knowledge.content,
        meta_data=deepcopy(source_knowledge.meta_data or {}),
        enable_status=source_knowledge.enable_status,
        parse_status=source_knowledge.parse_status,
        parse_error=source_knowledge.parse_error,
        tag_ids=source_knowledge.tag_ids,
        created_at=source_knowledge.created_at,  # 保持原创建时间
        updated_at=source_knowledge.updated_at,
        deleted_at=None,
    )

    session.add(new_knowledge)
    await session.commit()
    await session.refresh(new_knowledge)

    logger.debug(
        "knowledge_cloned",
        source_id=source_knowledge.id,
        target_id=new_knowledge.id,
    )

    return new_knowledge


async def _clone_chunks(
    session,
    source_chunks: list[Chunk],
    target_knowledge_id: str,
    target_kb_id: str,
    tenant_id: int,
) -> list[Chunk]:
    """复制分块

    Args:
        session: 数据库会话
        source_chunks: 源分块列表
        target_knowledge_id: 目标知识 ID
        target_kb_id: 目标知识库 ID
        tenant_id: 租户 ID

    Returns:
        新创建的分块列表
    """
    import uuid
    from copy import deepcopy

    new_chunks = []
    for i, chunk in enumerate(source_chunks):
        new_chunk = Chunk(
            id=str(uuid.uuid4()),
            knowledge_id=target_knowledge_id,
            knowledge_base_id=target_kb_id,
            tenant_id=tenant_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            chunk_type=chunk.chunk_type,
            start_at=chunk.start_at,
            end_at=chunk.end_at,
            is_enabled=chunk.is_enabled,
            embedding=chunk.embedding,  # 复制向量
            meta_data=deepcopy(chunk.meta_data or {}),
        )
        session.add(new_chunk)
        new_chunks.append(new_chunk)

    await session.commit()

    # 刷新获取 ID
    for chunk in new_chunks:
        await session.refresh(chunk)

    logger.debug(
        "chunks_cloned",
        knowledge_id=target_knowledge_id,
        count=len(new_chunks),
    )

    return new_chunks
