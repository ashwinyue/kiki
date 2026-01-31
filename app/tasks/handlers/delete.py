"""删除任务处理器

对齐 WeKnora99 的删除功能 (IndexDelete, KBDelete, KnowledgeListDelete)
"""

from app.infra.database import async_session_factory
from app.observability.logging import get_logger
from app.repositories.knowledge import (
    ChunkRepository,
    KnowledgeBaseRepository,
    KnowledgeRepository,
)
from app.services.search.hybrid_search import HybridSearchService
from app.tasks.handlers.base import ProgressReporter, TaskHandler, task_context

logger = get_logger(__name__)


async def process_index_delete(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理索引删除任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_base_id = payload.get("knowledge_base_id")
    chunk_ids = payload.get("chunk_ids", [])
    embedding_model_id = payload.get("embedding_model_id")

    task_id = f"idx_del_{knowledge_base_id[:8]}"

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="index:delete",
                payload=payload,
                title=f"删除 {len(chunk_ids)} 个向量索引",
                business_id=knowledge_base_id,
                business_type="knowledge_base",
                total_items=len(chunk_ids),
            )

            await handler.mark_started()

            try:
                # 初始化服务
                chunk_repo = ChunkRepository(session)
                search_service = HybridSearchService(
                    session=session,
                    embedding_model_id=embedding_model_id,
                )

                # 步骤 1: 删除向量索引
                await handler.update_progress(
                    30,
                    "删除向量索引",
                    processed_items=0,
                    total_items=len(chunk_ids),
                )

                deleted_count = 0
                for i, chunk_id in enumerate(chunk_ids):
                    await handler.update_progress(
                        30 + int((i / len(chunk_ids)) * 60),
                        f"删除索引 {i+1}/{len(chunk_ids)}",
                        processed_items=i + 1,
                        total_items=len(chunk_ids),
                    )

                    try:
                        # 删除向量存储中的索引
                        await search_service.delete_vectors(
                            knowledge_base_id=knowledge_base_id,
                            chunk_ids=[chunk_id],
                        )

                        # 更新分块状态
                        await chunk_repo.update_fields(
                            chunk_id=chunk_id,
                            tenant_id=tenant_id,
                            is_enabled=False,
                        )

                        deleted_count += 1

                    except Exception as e:
                        logger.warning(
                            "index_delete_failed",
                            chunk_id=chunk_id,
                            error=str(e),
                        )
                        continue

                # 完成
                result = {
                    "status": "completed",
                    "knowledge_base_id": knowledge_base_id,
                    "chunk_count": len(chunk_ids),
                    "deleted_count": deleted_count,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


async def process_kb_delete(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理知识库删除任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_base_id = payload.get("knowledge_base_id")
    delete_files = payload.get("delete_files", False)

    task_id = f"kb_del_{knowledge_base_id[:8]}"

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="kb:delete",
                payload=payload,
                title=f"删除知识库 {knowledge_base_id}",
                business_id=knowledge_base_id,
                business_type="knowledge_base",
            )

            await handler.mark_started()

            try:
                # 初始化仓储
                kb_repo = KnowledgeBaseRepository(session)
                knowledge_repo = KnowledgeRepository(session)
                chunk_repo = ChunkRepository(session)

                # 验证知识库存在
                kb = await kb_repo.get_by_tenant(knowledge_base_id, tenant_id)
                if not kb:
                    raise ValueError(f"Knowledge base {knowledge_base_id} not found")

                # 获取知识库下的所有知识
                from app.repositories.base import PaginationParams

                knowledges_result = await knowledge_repo.list_by_kb(
                    knowledge_base_id, tenant_id, PaginationParams(page=1, size=10000)
                )
                total_knowledge = knowledges_result.total

                # 步骤 1: 删除向量索引
                await handler.update_progress(20, "删除向量索引")

                try:
                    search_service = HybridSearchService(
                        session=session,
                        embedding_model_id=kb.embedding_model_id,
                    )
                    await search_service.delete_kb_index(knowledge_base_id)
                except Exception as e:
                    logger.warning(
                        "vector_index_delete_failed",
                        kb_id=knowledge_base_id,
                        error=str(e),
                    )

                # 步骤 2: 删除分块
                await handler.update_progress(40, "删除分块")

                deleted_chunks = await chunk_repo.delete_by_knowledge_bulk(
                    knowledge_base_id, tenant_id
                )
                logger.info(
                    "chunks_deleted",
                    kb_id=knowledge_base_id,
                    count=deleted_chunks,
                )

                # 步骤 3: 删除知识
                await handler.update_progress(60, "删除知识")

                deleted_knowledge = await knowledge_repo.delete_by_knowledge_base(
                    knowledge_base_id, tenant_id
                )
                logger.info(
                    "knowledges_deleted",
                    kb_id=knowledge_base_id,
                    count=deleted_knowledge,
                )

                # 步骤 4: 删除知识库（软删除）
                await handler.update_progress(80, "删除知识库")

                await kb_repo.soft_delete(knowledge_base_id, tenant_id)

                # 步骤 5: 删除文件（如果需要）
                if delete_files:
                    await handler.update_progress(90, "删除文件")

                    # TODO: 实现文件删除逻辑
                    # from app.infra.storage import storage
                    # for knowledge in knowledges_result.items:
                    #     if knowledge.file_path:
                    #         await storage.delete(knowledge.file_path)

                # 完成
                result = {
                    "status": "completed",
                    "knowledge_base_id": knowledge_base_id,
                    "deleted_chunks": deleted_chunks,
                    "deleted_knowledge": deleted_knowledge,
                    "deleted_files": delete_files,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


async def process_knowledge_list_delete(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理批量删除知识任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_ids = payload.get("knowledge_ids", [])

    task_id = f"know_del_{len(knowledge_ids)}"

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            await handler.create_task(
                task_type="knowledge:list_delete",
                payload=payload,
                title=f"批量删除 {len(knowledge_ids)} 个知识",
                total_items=len(knowledge_ids),
            )

            await handler.mark_started()

            try:
                # 初始化仓储
                knowledge_repo = KnowledgeRepository(session)
                chunk_repo = ChunkRepository(session)

                # 获取知识库信息（用于向量删除）
                knowledge = await knowledge_repo.get_by_tenant(knowledge_ids[0], tenant_id)
                if knowledge:
                    kb_id = knowledge.knowledge_base_id

                    try:
                        search_service = HybridSearchService(
                            session=session,
                            embedding_model_id=None,
                        )
                        vector_service_available = True
                    except Exception:
                        vector_service_available = False
                else:
                    vector_service_available = False

                # 逐个删除知识
                deleted_count = 0
                failed_count = 0
                reporter = ProgressReporter(handler, total=len(knowledge_ids), current_step="删除知识")

                for i, knowledge_id in enumerate(knowledge_ids):
                    try:
                        await handler.update_progress(
                            int((i / len(knowledge_ids)) * 100),
                            f"删除知识: {knowledge_id[:8]}",
                            processed_items=i + 1,
                            total_items=len(knowledge_ids),
                            failed_items=failed_count,
                        )

                        # 获取知识条目以获取知识库 ID 和分块信息
                        knowledge = await knowledge_repo.get_by_tenant(knowledge_id, tenant_id)
                        if not knowledge:
                            logger.warning(
                                "knowledge_not_found",
                                knowledge_id=knowledge_id,
                            )
                            failed_count += 1
                            reporter.failed += 1
                            continue

                        # 删除向量索引
                        if vector_service_available:
                            try:
                                chunks = await chunk_repo.list_by_knowledge(
                                    knowledge_id,
                                    tenant_id,
                                    app.repositories.base.PaginationParams(page=1, size=10000),
                                )
                                chunk_ids = [c.id for c in chunks.items]
                                if chunk_ids:
                                    await search_service.delete_vectors(
                                        knowledge_base_id=kb_id,
                                        chunk_ids=chunk_ids,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "vector_delete_failed",
                                    knowledge_id=knowledge_id,
                                    error=str(e),
                                )

                        # 删除分块
                        await chunk_repo.delete_by_knowledge(knowledge_id, tenant_id)

                        # 删除知识
                        await knowledge_repo.soft_delete(knowledge_id, tenant_id)

                        deleted_count += 1
                        reporter.processed += 1

                    except Exception as e:
                        logger.error(
                            "knowledge_delete_failed",
                            knowledge_id=knowledge_id,
                            error=str(e),
                        )
                        failed_count += 1
                        reporter.failed += 1
                        continue

                # 完成
                result = {
                    "status": "completed",
                    "total_count": len(knowledge_ids),
                    "deleted_count": deleted_count,
                    "failed_count": failed_count,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise
