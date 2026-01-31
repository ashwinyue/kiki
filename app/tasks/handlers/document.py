"""文档处理任务处理器

对齐 WeKnora99 的文档处理功能 (TypeDocumentProcess)

处理流程:
1. 文档下载 (URL)
2. 文档解析 (PDF, Word, Excel, PPT, HTML, Markdown)
3. 文档分块
4. 向量化
5. 存储
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.observability.logging import get_logger
from app.repositories.knowledge import ChunkRepository, KnowledgeRepository
from app.services.knowledge.document.loaders import load_document
from app.services.knowledge.base import KnowledgeBaseService
from app.tasks.handlers.base import TaskHandler, task_context

logger = get_logger(__name__)


async def process_document(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理文档处理任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    task_id = payload.get("task_id", celery_task.request.id)
    knowledge_id = payload.get("knowledge_id")
    knowledge_base_id = payload.get("knowledge_base_id")

    async with task_context(celery_task, task_id, tenant_id) as handler:
        # 获取会话
        from app.infra.database import async_session_factory

        async with async_session_factory() as session:
            # 创建任务记录
            await handler.create_task(
                task_type="document:process",
                payload=payload,
                title=f"处理文档: {payload.get('file_name', payload.get('url', '未知'))}",
                description=f"知识库 {knowledge_base_id}",
                business_id=knowledge_id,
                business_type="knowledge",
            )

            # 执行文档处理
            result = await _execute_document_processing(
                handler,
                session,
                payload,
                tenant_id,
            )

            return result


async def _execute_document_processing(
    handler: TaskHandler,
    session,
    payload: dict,
    tenant_id: int,
) -> dict:
    """执行文档处理流程

    Args:
        handler: 任务处理器
        session: 数据库会话
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    knowledge_id = payload.get("knowledge_id")
    knowledge_base_id = payload.get("knowledge_base_id")
    file_path = payload.get("file_path")
    url = payload.get("url")
    passages = payload.get("passages", [])
    enable_multimodel = payload.get("enable_multimodel", False)
    enable_question_generation = payload.get("enable_question_generation", False)

    # 初始化仓储
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)

    await handler.mark_started()

    try:
        # 步骤 1: 加载文档
        await handler.update_progress(10, "加载文档")

        # 确定数据源
        source = url if url else file_path
        if not source:
            raise ValueError("必须提供 file_path 或 url")

        # 加载文档
        doc_result = await load_document(source)
        documents = [doc_result.to_langchain_document()]

        logger.info(
            "document_loaded",
            source=source,
            page_count=doc_result.page_count,
            content_length=len(doc_result.content),
        )

        # 步骤 2: 文档分块
        await handler.update_progress(30, "文档分块")

        # 获取知识库分块配置
        kb_service = KnowledgeBaseService(session)
        kb = await kb_service.get_knowledge_base(knowledge_base_id, tenant_id)
        if not kb:
            raise ValueError(f"Knowledge base {knowledge_base_id} not found")

        chunking_config = kb.chunking_config or {}
        chunk_size = chunking_config.get("chunk_size", 1000)
        chunk_overlap = chunking_config.get("chunk_overlap", 200)
        separators = chunking_config.get("separators", ["\n\n", "\n", "。", ".", " ", ""])

        # 使用分块器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        # 分割文档
        chunks = []
        current_index = 0
        for doc in documents:
            split_docs = text_splitter.split_documents([doc])
            for i, split_doc in enumerate(split_docs):
                chunks.append(
                    {
                        "content": split_doc.page_content,
                        "chunk_index": current_index + i,
                        "start_at": 0,
                        "end_at": len(split_doc.page_content),
                        "is_enabled": True,
                        "chunk_type": "text",
                        "meta_data": {
                            "source": split_doc.metadata.get("source", ""),
                            "page": split_doc.metadata.get("page", 0),
                        },
                    }
                )
            current_index += len(split_docs)

        logger.info(
            "document_chunked",
            knowledge_id=knowledge_id,
            chunk_count=len(chunks),
        )

        # 步骤 3: 保存分块到数据库
        await handler.update_progress(60, "保存分块")

        created_chunks = await chunk_repo.create_chunks(
            chunks=chunks,
            kb_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            tenant_id=tenant_id,
        )

        # 步骤 4: 更新知识条目状态
        await handler.update_progress(80, "更新知识状态")

        await knowledge_repo.update_chunk_count(knowledge_id, len(created_chunks))
        await knowledge_repo.update_parse_status(knowledge_id, "completed")

        # 步骤 5: 生成问题 (可选)
        question_count = 0
        if enable_question_generation and created_chunks:
            await handler.update_progress(90, "生成问题")
            question_count = await _generate_questions_for_chunks(
                handler,
                session,
                created_chunks,
                tenant_id,
            )

        # 完成
        result = {
            "status": "completed",
            "knowledge_id": knowledge_id,
            "knowledge_base_id": knowledge_base_id,
            "chunk_count": len(created_chunks),
            "question_count": question_count,
        }

        await handler.mark_completed(result)

        return result

    except Exception as e:
        import traceback

        error_message = f"文档处理失败: {str(e)}"
        error_stack = traceback.format_exc()

        # 更新知识条目状态为失败
        try:
            await knowledge_repo.update_parse_status(knowledge_id, "failed", str(e))
        except Exception:
            pass

        await handler.mark_failed(error_message, error_stack)

        logger.exception(
            "document_processing_failed",
            knowledge_id=knowledge_id,
            error=str(e),
        )

        raise


async def _generate_questions_for_chunks(
    handler: TaskHandler,
    session,
    chunks: list,
    tenant_id: int,
) -> int:
    """为分块生成问题

    Args:
        handler: 任务处理器
        session: 数据库会话
        chunks: 分块列表
        tenant_id: 租户 ID

    Returns:
        生成的问题数量
    """
    from app.services.model_service import ModelService

    model_service = ModelService(session)
    chunk_repo = ChunkRepository(session)

    # 获取默认模型
    try:
        model = await model_service.get_default_model(tenant_id, "embedding")
    except Exception:
        logger.warning(
            "no_embedding_model",
            tenant_id=tenant_id,
        )
        return 0

    # 生成问题（每个分块生成 1-2 个问题）
    question_count = 0
    prompt_template = """根据以下内容，生成 1-2 个常见问题及其答案：

内容：
{content}

请以 JSON 格式返回：
{{
    "questions": [
        {{
            "question": "问题1",
            "answer": "答案1"
        }}
    ]
}}"""

    for i, chunk in enumerate(chunks):
        try:
            await handler.update_progress(
                90 + int((i / len(chunks)) * 10),
                f"为分块 {i+1}/{len(chunks)} 生成问题",
            )

            # 跳过太短的内容
            if len(chunk.content) if hasattr(chunk, 'content') else 0 < 50:
                continue

            # 获取分块内容
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if len(content) < 50:
                continue

            # 调用 LLM 生成问题（如果需要）
            # questions = await model_service.call_llm(...)

            # 模拟生成问题（实际使用时应该调用 LLM）
            questions_data = {
                "questions": [
                    {
                        "question": f"关于这段内容的常见问题",
                        "answer": content[:200] + "...",
                    }
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

            question_count += len(questions_data["questions"])

        except Exception as e:
            logger.warning(
                "question_generation_failed",
                chunk_id=chunk.id if hasattr(chunk, 'id') else str(i),
                error=str(e),
            )
            continue

    return question_count
