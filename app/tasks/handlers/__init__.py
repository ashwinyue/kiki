"""异步任务处理器

对齐 WeKnora99 的任务处理器 (internal/router/task.go)

任务处理器列表:
    - chunk_extract: 分块提取
    - document_process: 文档处理
    - faq_import: FAQ 导入
    - question_generation: 问题生成
    - summary_generation: 摘要生成
    - kb_clone: 知识库复制
    - index_delete: 索引删除
    - kb_delete: 知识库删除
    - knowledge_list_delete: 批量删除知识
    - datatable_summary: 表格摘要

每个处理器:
    1. 在 Redis 创建/更新任务状态
    2. 执行业务逻辑
    3. 更新任务进度
    4. 处理错误和重试
"""

from app.tasks.celery_app import DatabaseTask, task
from app.tasks.types import TaskType
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 文档处理任务 ==============


@task(
    name=TaskType.DOCUMENT_PROCESS,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def document_process(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """文档处理任务

    对应 WeKnora99 的 TypeDocumentProcess

    Payload:
        - request_id: 请求 ID
        - knowledge_id: 知识 ID
        - knowledge_base_id: 知识库 ID
        - file_path: 文件路径
        - url: URL
        - passages: 文本段落
        - enable_multimodel: 是否启用多模态
        - enable_question_generation: 是否启用问题生成
    """
    from app.tasks.handlers.document import process_document

    return process_document(self, payload, tenant_id)


# ============== FAQ 导入任务 ==============


@task(
    name=TaskType.FAQ_IMPORT,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def faq_import(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """FAQ 导入任务

    对应 WeKnora99 的 TypeFAQImport

    Payload:
        - task_id: 任务 ID
        - kb_id: 知识库 ID
        - knowledge_id: 知识 ID
        - entries: FAQ 条目
        - entries_url: FAQ 条目 URL
        - dry_run: 是否为 dry run 模式
    """
    from app.tasks.handlers.faq import process_faq_import

    return process_faq_import(self, payload, tenant_id)


# ============== 知识库复制任务 ==============


@task(
    name=TaskType.KB_CLONE,
    base=DatabaseTask,
    bind=True,
    max_retries=2,
    default_retry_delay=120,
)
def kb_clone(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """知识库复制任务

    对应 WeKnora99 的 TypeKBClone

    Payload:
        - task_id: 任务 ID
        - source_id: 源知识库 ID
        - target_id: 目标知识库 ID
    """
    from app.tasks.handlers.kb_clone import process_kb_clone

    return process_kb_clone(self, payload, tenant_id)


# ============== 问题生成任务 ==============


@task(
    name=TaskType.QUESTION_GENERATION,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def question_generation(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """问题生成任务

    对应 WeKnora99 的 TypeQuestionGeneration

    Payload:
        - knowledge_base_id: 知识库 ID
        - knowledge_id: 知识 ID
        - question_count: 问题数量
    """
    from app.tasks.handlers.generation import process_question_generation

    return process_question_generation(self, payload, tenant_id)


# ============== 摘要生成任务 ==============


@task(
    name=TaskType.SUMMARY_GENERATION,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def summary_generation(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """摘要生成任务

    对应 WeKnora99 的 TypeSummaryGeneration

    Payload:
        - knowledge_base_id: 知识库 ID
        - knowledge_id: 知识 ID
    """
    from app.tasks.handlers.generation import process_summary_generation

    return process_summary_generation(self, payload, tenant_id)


# ============== 分块提取任务 ==============


@task(
    name=TaskType.CHUNK_EXTRACT,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def chunk_extract(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """分块提取任务

    对应 WeKnora99 的 TypeChunkExtract

    Payload:
        - chunk_id: 分块 ID
        - model_id: 模型 ID
    """
    from app.tasks.handlers.chunk import process_chunk_extract

    return process_chunk_extract(self, payload, tenant_id)


# ============== 删除任务 ==============


@task(
    name=TaskType.INDEX_DELETE,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
)
def index_delete(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """索引删除任务

    对应 WeKnora99 的 TypeIndexDelete

    Payload:
        - knowledge_base_id: 知识库 ID
        - chunk_ids: 分块 ID 列表
        - embedding_model_id: 嵌入模型 ID
    """
    from app.tasks.handlers.delete import process_index_delete

    return process_index_delete(self, payload, tenant_id)


@task(
    name=TaskType.KB_DELETE,
    base=DatabaseTask,
    bind=True,
    max_retries=2,
)
def kb_delete(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """知识库删除任务

    对应 WeKnora99 的 TypeKBDelete

    Payload:
        - knowledge_base_id: 知识库 ID
    """
    from app.tasks.handlers.delete import process_kb_delete

    return process_kb_delete(self, payload, tenant_id)


@task(
    name=TaskType.KNOWLEDGE_LIST_DELETE,
    base=DatabaseTask,
    bind=True,
    max_retries=2,
)
def knowledge_list_delete(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """批量删除知识任务

    对应 WeKnora99 的 TypeKnowledgeListDelete

    Payload:
        - knowledge_ids: 知识 ID 列表
    """
    from app.tasks.handlers.delete import process_knowledge_list_delete

    return process_knowledge_list_delete(self, payload, tenant_id)


# ============== 表格摘要任务 ==============


@task(
    name=TaskType.DATATABLE_SUMMARY,
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def datatable_summary(
    self,
    payload: dict,
    tenant_id: int,
) -> dict:
    """表格摘要任务

    对应 WeKnora99 的 TypeDataTableSummary

    Payload:
        - knowledge_id: 知识 ID
        - chunk_id: 分块 ID
        - model_id: 模型 ID
    """
    from app.tasks.handlers.datatable import process_datatable_summary

    return process_datatable_summary(self, payload, tenant_id)


# ============== 清理任务 ==============


@task(
    name="tasks.cleanup",
    base=DatabaseTask,
)
def cleanup_tasks(days: int = 7) -> dict:
    """清理已完成的旧任务

    Args:
        days: 保留天数
    """
    from app.tasks.handlers.cleanup import process_cleanup

    return process_cleanup(days)


# ============== 导出 ==============

__all__ = [
    # 任务处理器
    "document_process",
    "faq_import",
    "kb_clone",
    "question_generation",
    "summary_generation",
    "chunk_extract",
    "index_delete",
    "kb_delete",
    "knowledge_list_delete",
    "datatable_summary",
    "cleanup_tasks",
]
