"""异步任务模块

对齐 WeKnora99 的异步任务系统，使用 Celery + Redis 实现。

任务类型:
    - chunk:extract: 分块提取
    - document:process: 文档处理
    - faq:import: FAQ 导入
    - question:generation: 问题生成
    - summary:generation: 摘要生成
    - kb:clone: 知识库复制
    - index:delete: 索引删除
    - kb:delete: 知识库删除
    - knowledge:list_delete: 批量删除知识
    - datatable:summary: 表格摘要

使用方式:
    from app.tasks import send_task, generate_task_id
    from app.tasks.types import TaskType, TaskPriority

    # 生成任务 ID
    task_id = generate_task_id(TaskType.DOCUMENT_PROCESS, tenant_id=123, business_id="kb789")

    # 发送任务
    celery_task_id = send_task(
        TaskType.DOCUMENT_PROCESS,
        payload={"knowledge_id": "know-123"},
        tenant_id=123,
        priority=TaskPriority.DEFAULT,
    )
"""

# ============== Celery 任务系统 (对齐 WeKnora99) ==============

# 任务类型
from app.tasks.types import (
    ChunkExtractPayload,
    DataTableSummaryPayload,
    DocumentProcessPayload,
    FAQImportPayload,
    IndexDeletePayload,
    KBDeletePayload,
    KBClonePayload,
    KnowledgeListDeletePayload,
    QuestionGenerationPayload,
    SummaryGenerationPayload,
    TaskPriority,
    TaskPriorityLiteral,
    TaskStatus,
    TaskStatusLiteral,
    TaskType,
    TaskTypeLiteral,
    QUEUE_PRIORITY_WEIGHTS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_RETRY_BACKOFF_MAX,
)

# 任务 ID 工具
from app.tasks.task_id import (
    ParsedTaskID,
    build_task_id_pattern,
    extract_task_type,
    extract_tenant_id,
    generate_task_id,
    generate_task_id_with_prefix,
    matches_task_id,
    parse_task_id,
    sanitize_business_id,
    sanitize_task_type,
    validate_task_id,
)

# Celery 应用
from app.tasks.celery_app import (
    CeleryConfig,
    DatabaseTask,
    celery_app,
    create_celery_app,
    get_celery_app,
    get_task_status as get_celery_task_status,
    revoke_task,
    send_task,
    task,
)

# ============== 向后兼容 - 旧的 asyncio 任务 ==============

from app.tasks.copy_tasks import (
    cancel_task as cancel_copy_task,
    get_running_tasks as get_copy_running_tasks,
    is_task_running as is_copy_task_running,
    start_copy_task,
)
from app.tasks.initialization import (
    cancel_task as cancel_init_task,
    get_running_tasks as get_init_running_tasks,
    get_task_status,
    is_task_running as is_init_task_running,
    start_initialization_task,
)
from app.tasks.ollama_tasks import (
    cancel_download_task,
    get_download_task,
    list_download_tasks,
    start_model_download,
)

# 延迟导入任务模型，避免循环导入
# from app.models.task import Task, TaskCreate, TaskUpdate, TaskPublic, TaskListModel, TaskLog, TaskLogCreate, TaskLogPublic

__all__ = [
    # ========== Celery 任务系统 ==========
    # 任务类型
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    "TaskTypeLiteral",
    "TaskPriorityLiteral",
    "TaskStatusLiteral",
    # Payload 类
    "ChunkExtractPayload",
    "DocumentProcessPayload",
    "FAQImportPayload",
    "QuestionGenerationPayload",
    "SummaryGenerationPayload",
    "KBClonePayload",
    "IndexDeletePayload",
    "KBDeletePayload",
    "KnowledgeListDeletePayload",
    "DataTableSummaryPayload",
    # 配置常量
    "QUEUE_PRIORITY_WEIGHTS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF",
    "DEFAULT_RETRY_BACKOFF_MAX",
    # 任务 ID 工具
    "generate_task_id",
    "generate_task_id_with_prefix",
    "parse_task_id",
    "validate_task_id",
    "extract_task_type",
    "extract_tenant_id",
    "build_task_id_pattern",
    "matches_task_id",
    "ParsedTaskID",
    # Celery
    "celery_app",
    "get_celery_app",
    "create_celery_app",
    "task",
    "send_task",
    "revoke_task",
    "get_celery_task_status",
    "CeleryConfig",
    "DatabaseTask",
    # ========== 向后兼容 (asyncio 任务) ==========
    # 复制任务
    "start_copy_task",
    "is_copy_task_running",
    "cancel_copy_task",
    "get_copy_running_tasks",
    # 初始化任务
    "start_initialization_task",
    "is_init_task_running",
    "cancel_init_task",
    "get_init_running_tasks",
    "get_task_status",
    # Ollama 任务
    "start_model_download",
    "get_download_task",
    "cancel_download_task",
    "list_download_tasks",
]
