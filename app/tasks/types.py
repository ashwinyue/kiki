"""异步任务类型定义

对齐 WeKnora99 的异步任务类型 (internal/types/extract_graph.go)
使用 Celery 替代 Go 的 asynq 实现任务队列。

任务类型映射:
    WeKnora (Go)          ->  Kiki (Python)
    ----------------------------------------
    chunk:extract         ->  chunk_extract
    document:process      ->  document_process
    faq:import            ->  faq_import
    question:generation   ->  question_generation
    summary:generation    ->  summary_generation
    kb:clone              ->  kb_clone
    index:delete          ->  index_delete
    kb:delete             ->  kb_delete
    knowledge:list_delete ->  knowledge_list_delete
    datatable:summary     ->  datatable_summary
"""

from enum import Enum
from typing import Literal

# ============== 任务类型常量 ==============


class TaskType(str, Enum):
    """异步任务类型

    对齐 WeKnora99 的任务类型定义。
    使用枚举确保类型安全和 IDE 提示。
    """

    # 分块提取任务
    CHUNK_EXTRACT = "chunk:extract"

    # 文档处理任务
    DOCUMENT_PROCESS = "document:process"

    # FAQ 导入任务（包含 dry run 模式）
    FAQ_IMPORT = "faq:import"

    # 问题生成任务
    QUESTION_GENERATION = "question:generation"

    # 摘要生成任务
    SUMMARY_GENERATION = "summary:generation"

    # 知识库复制任务
    KB_CLONE = "kb:clone"

    # 索引删除任务
    INDEX_DELETE = "index:delete"

    # 知识库删除任务
    KB_DELETE = "kb:delete"

    # 批量删除知识任务
    KNOWLEDGE_LIST_DELETE = "knowledge:list_delete"

    # 表格摘要任务
    DATATABLE_SUMMARY = "datatable:summary"


# ============== 任务优先级 ==============


class TaskPriority(str, Enum):
    """任务优先级

    对齐 WeKnora99 的 asynq 队列优先级:
    - critical: 最高优先级 (权重 6)
    - default: 默认优先级 (权重 3)
    - low: 低优先级 (权重 1)
    """

    CRITICAL = "critical"
    DEFAULT = "default"
    LOW = "low"


# ============== 任务状态 ==============


class TaskStatus(str, Enum):
    """任务执行状态

    对齐 WeKnora99 的 KBCloneTaskStatus
    """

    PENDING = "pending"  # 等待执行
    PROCESSING = "processing"  # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 已取消
    RETRYING = "retrying"  # 重试中


# ============== 类型别名 ==============


# 用于类型提示的字面量类型
TaskTypeLiteral = Literal[
    "chunk:extract",
    "document:process",
    "faq:import",
    "question:generation",
    "summary:generation",
    "kb:clone",
    "index:delete",
    "kb:delete",
    "knowledge:list_delete",
    "datatable:summary",
]

TaskPriorityLiteral = Literal["critical", "default", "low"]
TaskStatusLiteral = Literal["pending", "processing", "completed", "failed", "cancelled", "retrying"]


# ============== 任务队列配置 ==============

# 队列优先级权重 (对齐 WeKnora99)
QUEUE_PRIORITY_WEIGHTS = {
    TaskPriority.CRITICAL: 6,
    TaskPriority.DEFAULT: 3,
    TaskPriority.LOW: 1,
}

# 默认重试配置
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 60  # 秒
DEFAULT_RETRY_BACKOFF_MAX = 600  # 10 分钟


# ============== 任务 Payload 定义 ==============


class ChunkExtractPayload:
    """分块提取任务 Payload

    对应 WeKnora99 ExtractChunkPayload
    """

    task_type: TaskTypeLiteral = "chunk:extract"
    tenant_id: int
    chunk_id: str
    model_id: str


class DocumentProcessPayload:
    """文档处理任务 Payload

    对应 WeKnora99 DocumentProcessPayload
    """

    task_type: TaskTypeLiteral = "document:process"
    request_id: str
    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    file_path: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    url: str | None = None
    passages: list[str] | None = None
    enable_multimodel: bool = False
    enable_question_generation: bool = False
    question_count: int | None = None


class FAQImportPayload:
    """FAQ 导入任务 Payload

    对应 WeKnora99 FAQImportPayload
    """

    task_type: TaskTypeLiteral = "faq:import"
    tenant_id: int
    task_id: str
    kb_id: str
    knowledge_id: str | None = None
    entries: list[dict] | None = None
    entries_url: str | None = None
    entry_count: int | None = None
    mode: str = "create"
    dry_run: bool = False
    enqueued_at: int | None = None


class QuestionGenerationPayload:
    """问题生成任务 Payload

    对应 WeKnora99 QuestionGenerationPayload
    """

    task_type: TaskTypeLiteral = "question:generation"
    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    question_count: int = 5


class SummaryGenerationPayload:
    """摘要生成任务 Payload

    对应 WeKnora99 SummaryGenerationPayload
    """

    task_type: TaskTypeLiteral = "summary:generation"
    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str


class KBClonePayload:
    """知识库复制任务 Payload

    对应 WeKnora99 KBClonePayload
    """

    task_type: TaskTypeLiteral = "kb:clone"
    tenant_id: int
    task_id: str
    source_id: str
    target_id: str


class IndexDeletePayload:
    """索引删除任务 Payload

    对应 WeKnora99 IndexDeletePayload
    """

    task_type: TaskTypeLiteral = "index:delete"
    tenant_id: int
    knowledge_base_id: str
    embedding_model_id: str
    kb_type: str
    chunk_ids: list[str]
    # effective_engines: list[dict]  # TODO: 定义 RetrieverEngineParams


class KBDeletePayload:
    """知识库删除任务 Payload

    对应 WeKnora99 KBDeletePayload
    """

    task_type: TaskTypeLiteral = "kb:delete"
    tenant_id: int
    knowledge_base_id: str
    # effective_engines: list[dict]


class KnowledgeListDeletePayload:
    """批量删除知识任务 Payload

    对应 WeKnora99 KnowledgeListDeletePayload
    """

    task_type: TaskTypeLiteral = "knowledge:list_delete"
    tenant_id: int
    knowledge_ids: list[str]


class DataTableSummaryPayload:
    """表格摘要任务 Payload

    对应 WeKnora99 DataTableSummary
    """

    task_type: TaskTypeLiteral = "datatable:summary"
    tenant_id: int
    knowledge_id: str
    chunk_id: str
    model_id: str


# ============== 导出 ==============

__all__ = [
    # 枚举类
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    # 字面量类型
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
]
