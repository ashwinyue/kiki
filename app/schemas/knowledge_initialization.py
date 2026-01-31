"""知识库初始化配置 Schema

对齐 WeKnora99 internal/handler/initialization.go 的请求/响应结构
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class InitializationStatus(str, Enum):
    """初始化状态

    对齐 WeKnora99 的任务状态
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============== LLM 配置 ==============


class LLMConfig(BaseModel):
    """LLM 模型配置"""

    model_id: str = Field(..., description="模型 ID")
    source: str = Field(default="remote", description="模型源: remote, local, aliyun, zhipu")
    model_name: str = Field(default="", description="模型名称")
    base_url: str = Field(default="", description="API Base URL")
    api_key: str = Field(default="", description="API Key")


# ============== Embedding 配置 ==============


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""

    model_id: str = Field(..., description="模型 ID")
    source: str = Field(default="remote", description="模型源")
    model_name: str = Field(default="", description="模型名称")
    base_url: str = Field(default="", description="API Base URL")
    api_key: str = Field(default="", description="API Key")
    dimension: int = Field(default=0, description="向量维度")


# ============== Rerank 配置 ==============


class RerankConfig(BaseModel):
    """Rerank 模型配置"""

    enabled: bool = Field(default=False, description="是否启用重排序")
    model_id: str = Field(default="", description="模型 ID")
    source: str = Field(default="remote", description="模型源")
    model_name: str = Field(default="", description="模型名称")
    base_url: str = Field(default="", description="API Base URL")
    api_key: str = Field(default="", description="API Key")


# ============== 存储配置 ==============


class StorageConfig(BaseModel):
    """腾讯云 COS 配置"""

    secret_id: str = Field(default="", description="腾讯云 Secret ID")
    secret_key: str = Field(default="", description="腾讯云 Secret Key")
    region: str = Field(default="", description="地域")
    bucket_name: str = Field(default="", description="存储桶名称")
    app_id: str = Field(default="", description="应用 ID")
    path_prefix: str = Field(default="", description="路径前缀")


class MinioConfig(BaseModel):
    """MinIO 配置"""

    bucket_name: str = Field(default="", description="存储桶名称")
    path_prefix: str = Field(default="", description="路径前缀")
    use_ssl: bool = Field(default=False, description="是否使用 SSL")


# ============== 多模态配置 ==============


class MultimodalConfig(BaseModel):
    """多模态配置"""

    enabled: bool = Field(default=False, description="是否启用多模态")
    vlm_model_id: str = Field(default="", description="VLM 模型 ID")
    storage_type: str = Field(default="cos", description="存储类型: cos, minio")
    cos: StorageConfig | None = Field(None, description="COS 配置")
    minio: MinioConfig | None = Field(None, description="MinIO 配置")


# ============== 知识图谱提取配置 ==============


class GraphNode(BaseModel):
    """图节点"""

    name: str = Field(..., description="节点名称")
    attributes: list[str] = Field(default_factory=list, description="节点属性")


class GraphRelation(BaseModel):
    """图关系"""

    node1: str = Field(..., description="节点1名称")
    node2: str = Field(..., description="节点2名称")
    type: str = Field(..., description="关系类型")


class ExtractConfig(BaseModel):
    """知识图谱提取配置"""

    enabled: bool = Field(default=False, description="是否启用")
    text: str = Field(default="", description="示例文本")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    nodes: list[GraphNode] = Field(default_factory=list, description="提取的节点")
    relations: list[GraphRelation] = Field(default_factory=list, description="提取的关系")


# ============== 问题生成配置 ==============


class QuestionGenerationConfig(BaseModel):
    """问题生成配置"""

    enabled: bool = Field(default=False, description="是否启用问题生成")
    question_count: int = Field(default=3, ge=1, le=10, description="生成问题数量")


# ============== 向量存储配置 ==============


class VectorStoreConfig(BaseModel):
    """向量存储配置"""

    provider: str = Field(default="pgvector", description="向量存储提供商")
    index_type: str = Field(default="hnsw", description="索引类型")
    dimension: int = Field(default=1536, description="向量维度")


# ============== 初始化配置请求/响应 ==============


class KBInitConfigRequest(BaseModel):
    """知识库初始化配置请求

    对齐 WeKnora99 InitializationRequest
    """

    # LLM 配置
    llm: LLMConfig | None = Field(None, description="LLM 配置")

    # Embedding 配置
    embedding: EmbeddingConfig | None = Field(None, description="Embedding 配置")

    # Rerank 配置
    rerank: RerankConfig | None = Field(None, description="Rerank 配置")

    # 多模态配置
    multimodal: MultimodalConfig | None = Field(None, description="多模态配置")

    # 知识图谱提取配置
    extract: ExtractConfig | None = Field(None, description="知识图谱提取配置")

    # 问题生成配置
    question_generation: QuestionGenerationConfig | None = Field(
        None, description="问题生成配置"
    )

    # 向量存储配置
    vector_store: VectorStoreConfig | None = Field(None, description="向量存储配置")

    # 文档分块配置
    chunking: dict | None = Field(
        None,
        description="文档分块配置",
        examples=[
            {"chunk_size": 1000, "chunk_overlap": 200, "separators": ["\n\n", "\n", "。", "."]}
        ],
    )


class InitializationConfig(BaseModel):
    """知识库初始化配置

    对齐 WeKnora99 GET /initialization/kb/{kbId}/config 响应
    """

    kb_id: str = Field(..., description="知识库 ID")
    kb_name: str = Field(..., description="知识库名称")
    has_files: bool = Field(default=False, description="是否有文件")

    # 模型配置
    llm: LLMConfig | None = Field(None, description="LLM 配置")
    embedding: EmbeddingConfig | None = Field(None, description="Embedding 配置")
    rerank: RerankConfig = Field(
        default_factory=RerankConfig, description="Rerank 配置"
    )

    # 多模态配置
    multimodal: MultimodalConfig = Field(
        default_factory=MultimodalConfig, description="多模态配置"
    )

    # 知识图谱配置
    extract: ExtractConfig = Field(
        default_factory=ExtractConfig, description="知识图谱提取配置"
    )

    # 问题生成配置
    question_generation: QuestionGenerationConfig = Field(
        default_factory=QuestionGenerationConfig, description="问题生成配置"
    )

    # 向量存储配置
    vector_store: VectorStoreConfig | None = Field(None, description="向量存储配置")

    # 分块配置
    chunking: dict = Field(
        default_factory=lambda: {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", "。", "."],
        },
        description="文档分块配置",
    )


class KBInitConfigResponse(BaseModel):
    """知识库初始化配置响应

    对齐 WeKnora99 GET /initialization/kb/{kbId}/config 响应
    """

    success: bool = Field(..., description="是否成功")
    data: InitializationConfig = Field(..., description="配置数据")


class ValidationResult(BaseModel):
    """配置验证结果"""

    valid: bool = Field(..., description="是否有效")
    errors: list[str] = Field(default_factory=list, description="错误列表")
    warnings: list[str] = Field(default_factory=list, description="警告列表")


class KBValidationResponse(BaseModel):
    """知识库配置验证响应"""

    success: bool = Field(..., description="是否成功")
    data: ValidationResult = Field(..., description="验证结果")


class KBInitStatus(BaseModel):
    """知识库初始化状态

    对齐 WeKnora99 的任务状态响应
    """

    task_id: str = Field(..., description="任务 ID")
    kb_id: str = Field(..., description="知识库 ID")
    tenant_id: int = Field(..., description="租户 ID")
    status: InitializationStatus = Field(..., description="状态")
    message: str = Field(..., description="状态消息")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="进度百分比")
    error: str | None = Field(None, description="错误信息")
    started_at: datetime | None = Field(None, description="开始时间")
    completed_at: datetime | None = Field(None, description="完成时间")


class KBInitStatusResponse(BaseModel):
    """知识库初始化状态响应"""

    success: bool = Field(..., description="是否成功")
    data: KBInitStatus = Field(..., description="状态数据")


# ============== 模型检查请求/响应 ==============


class RemoteModelCheckRequest(BaseModel):
    """远程模型检查请求

    对齐 WeKnora99 RemoteModelCheckRequest
    """

    model_name: str = Field(..., description="模型名称")
    base_url: str = Field(..., description="Base URL")
    api_key: str = Field(default="", description="API Key")
    model_type: str = Field(default="llm", description="模型类型: llm, embedding, rerank")


class RemoteModelCheckResponse(BaseModel):
    """远程模型检查响应"""

    success: bool = Field(..., description="是否成功")
    data: dict = Field(
        default_factory=lambda: {"available": False, "message": ""},
        description="检查结果",
    )


# ============== Embedding 测试请求/响应 ==============


class EmbeddingTestRequest(BaseModel):
    """Embedding 测试请求

    对齐 WeKnora99 TestEmbeddingModel
    """

    source: str = Field(..., description="模型源")
    model_name: str = Field(..., description="模型名称")
    base_url: str = Field(default="", description="Base URL")
    api_key: str = Field(default="", description="API Key")
    dimension: int = Field(default=0, description="预期维度")
    provider: str = Field(default="", description="提供商")


class EmbeddingTestResponse(BaseModel):
    """Embedding 测试响应"""

    success: bool = Field(..., description="是否成功")
    data: dict = Field(
        default_factory=lambda: {"available": False, "dimension": 0, "message": ""},
        description="测试结果",
    )


# ============== Rerank 测试请求/响应 ==============


class RerankTestRequest(BaseModel):
    """Rerank 测试请求"""

    model_name: str = Field(..., description="模型名称")
    base_url: str = Field(..., description="Base URL")
    api_key: str = Field(default="", description="API Key")


class RerankTestResponse(BaseModel):
    """Rerank 测试响应"""

    success: bool = Field(..., description="是否成功")
    data: dict = Field(
        default_factory=lambda: {"available": False, "message": ""},
        description="测试结果",
    )


# ============== 文本关系提取请求/响应 ==============


class TextRelationExtractionRequest(BaseModel):
    """文本关系提取请求

    对齐 WeKnora99 TextRelationExtractionRequest
    """

    text: str = Field(..., min_length=1, max_length=5000, description="待提取文本")
    tags: list[str] = Field(..., min_items=1, description="关系标签")
    llm_config: LLMConfig = Field(..., description="LLM 配置")


class TextRelationExtractionResponse(BaseModel):
    """文本关系提取响应"""

    success: bool = Field(..., description="是否成功")
    data: dict = Field(
        default_factory=lambda: {"nodes": [], "relations": []},
        description="提取结果",
    )


# ============== 示例文本生成请求/响应 ==============


class FabriTextRequest(BaseModel):
    """生成示例文本请求

    对齐 WeKnora99 FabriTextRequest
    """

    tags: list[str] = Field(default_factory=list, description="标签列表")
    llm_config: LLMConfig = Field(..., description="LLM 配置")


class FabriTextResponse(BaseModel):
    """生成示例文本响应"""

    success: bool = Field(..., description="是否成功")
    data: dict = Field(
        default_factory=lambda: {"text": ""},
        description="生成的文本",
    )


__all__ = [
    # 枚举
    "InitializationStatus",
    # 模型配置
    "LLMConfig",
    "EmbeddingConfig",
    "RerankConfig",
    # 存储配置
    "StorageConfig",
    "MinioConfig",
    # 多模态配置
    "MultimodalConfig",
    # 知识图谱配置
    "GraphNode",
    "GraphRelation",
    "ExtractConfig",
    # 问题生成配置
    "QuestionGenerationConfig",
    # 向量存储配置
    "VectorStoreConfig",
    # 请求/响应
    "KBInitConfigRequest",
    "InitializationConfig",
    "KBInitConfigResponse",
    "ValidationResult",
    "KBValidationResponse",
    "KBInitStatus",
    "KBInitStatusResponse",
    # 模型检查
    "RemoteModelCheckRequest",
    "RemoteModelCheckResponse",
    # Embedding 测试
    "EmbeddingTestRequest",
    "EmbeddingTestResponse",
    # Rerank 测试
    "RerankTestRequest",
    "RerankTestResponse",
    # 文本关系提取
    "TextRelationExtractionRequest",
    "TextRelationExtractionResponse",
    # 示例文本生成
    "FabriTextRequest",
    "FabriTextResponse",
]
