"""系统初始化 Schema

对齐 WeKnora99 系统初始化 API 规范
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============== 模型配置 Schema ==============


class LLMConfig(BaseModel):
    """LLM 模型配置"""

    source: str = Field(..., description="模型来源")
    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field("", alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""

    source: str = Field(..., description="模型来源")
    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field("", alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")
    dimension: int = Field(0, description="向量维度")


class RerankConfig(BaseModel):
    """Rerank 模型配置"""

    enabled: bool = Field(False, description="是否启用")
    model_name: str = Field("", alias="modelName", description="模型名称")
    base_url: str = Field("", alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")


class VLMConfig(BaseModel):
    """VLM 模型配置"""

    model_name: str = Field("", alias="modelName", description="模型名称")
    base_url: str = Field("", alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")
    interface_type: str = Field("openai", alias="interfaceType", description="接口类型")


class MultimodalConfig(BaseModel):
    """多模态配置"""

    enabled: bool = Field(False, description="是否启用")
    vlm: VLMConfig | None = Field(None, description="VLM 配置")
    storage_type: str = Field("", alias="storageType", description="存储类型")


class DocumentSplittingConfig(BaseModel):
    """文档分割配置"""

    chunk_size: int = Field(1000, alias="chunkSize", description="块大小")
    chunk_overlap: int = Field(200, alias="chunkOverlap", description="块重叠")
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。"],
        description="分隔符",
    )


class NodeExtractConfig(BaseModel):
    """节点提取配置（知识图谱）"""

    enabled: bool = Field(False, description="是否启用")
    text: str = Field("", description="提取文本")
    tags: list[str] = Field(default_factory=list, description="标签")


class QuestionGenerationConfig(BaseModel):
    """问题生成配置"""

    enabled: bool = Field(False, description="是否启用")
    question_count: int = Field(3, alias="questionCount", ge=1, le=10)


class InitializationRequest(BaseModel):
    """初始化请求

    对齐 WeKnora99 POST /initialization/kb/{kbId}
    """

    llm: LLMConfig = Field(..., description="LLM 配置")
    embedding: EmbeddingConfig = Field(..., description="Embedding 配置")
    rerank: RerankConfig = Field(
        default_factory=lambda: RerankConfig(), description="Rerank 配置"
    )
    multimodal: MultimodalConfig = Field(
        default_factory=lambda: MultimodalConfig(), description="多模态配置"
    )
    document_splitting: DocumentSplittingConfig = Field(
        default_factory=DocumentSplittingConfig,
        alias="documentSplitting",
        description="文档分割配置",
    )
    node_extract: NodeExtractConfig = Field(
        default_factory=lambda: NodeExtractConfig(),
        alias="nodeExtract",
        description="节点提取配置",
    )
    question_generation: QuestionGenerationConfig = Field(
        default_factory=lambda: QuestionGenerationConfig(),
        alias="questionGeneration",
        description="问题生成配置",
    )


class KBModelConfigRequest(BaseModel):
    """知识库模型配置请求

    对齐 WeKnora99 PUT /initialization/kb/{kbId}/config
    """

    llm_model_id: str = Field(..., alias="llmModelId", description="LLM 模型 ID")
    embedding_model_id: str = Field(..., alias="embeddingModelId", description="Embedding 模型 ID")
    vlm_config: VLMConfig | None = Field(None, alias="vlm_config", description="VLM 配置")

    document_splitting: DocumentSplittingConfig = Field(
        default_factory=DocumentSplittingConfig,
        alias="documentSplitting",
        description="文档分割配置",
    )

    multimodal: MultimodalConfig = Field(
        default_factory=lambda: MultimodalConfig(), description="多模态配置"
    )

    node_extract: NodeExtractConfig = Field(
        default_factory=lambda: NodeExtractConfig(), alias="nodeExtract", description="节点提取配置"
    )

    question_generation: QuestionGenerationConfig = Field(
        default_factory=lambda: QuestionGenerationConfig(),
        alias="questionGeneration",
        description="问题生成配置",
    )


# ============== 测试请求 Schema ==============


class EmbeddingTestRequest(BaseModel):
    """Embedding 模型测试请求"""

    source: str = Field(..., description="模型来源")
    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field("", alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")
    dimension: int = Field(0, description="向量维度")
    provider: str = Field("", description="服务商")


class RerankTestRequest(BaseModel):
    """Rerank 模型测试请求"""

    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field(..., alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")


class RemoteModelCheckRequest(BaseModel):
    """远程模型检查请求"""

    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field(..., alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")


# ============== 响应 Schema ==============


class ModelTestResponse(BaseModel):
    """模型测试响应"""

    available: bool = Field(..., description="是否可用")
    message: str = Field(..., description="响应消息")
    dimension: int = Field(0, description="向量维度（仅 Embedding）")


class OllamaStatusResponse(BaseModel):
    """Ollama 状态响应"""

    available: bool = Field(..., description="是否可用")
    version: str = Field("", description="Ollama 版本")
    base_url: str = Field("", alias="baseUrl", description="Ollama 地址")


class KBConfigResponse(BaseModel):
    """知识库配置响应"""

    has_files: bool = Field(..., alias="hasFiles", description="是否有文件")
    llm: dict[str, Any] | None = Field(None, description="LLM 配置")
    embedding: dict[str, Any] | None = Field(None, description="Embedding 配置")
    rerank: dict[str, Any] | None = Field(None, description="Rerank 配置")
    multimodal: dict[str, Any] | None = Field(None, description="多模态配置")
    document_splitting: DocumentSplittingConfig | None = Field(
        None, alias="documentSplitting", description="文档分割配置"
    )
    node_extract: dict[str, Any] | None = Field(None, alias="nodeExtract", description="节点提取配置")


# ============== Ollama 模型管理 Schema ==============


class OllamaModelInfo(BaseModel):
    """Ollama 模型信息"""

    name: str = Field(..., description="模型名称")
    size: int = Field(0, description="模型大小（字节）")
    digest: str = Field("", description="模型摘要")
    modified_at: str = Field("", alias="modified_at", description="修改时间")

    class Config:
        populate_by_name = True


class OllamaPullRequest(BaseModel):
    """Ollama 模型拉取请求"""

    model_name: str = Field(..., alias="modelName", description="模型名称")
    insecure: bool = Field(False, description="是否跳过 TLS 验证")


class OllamaProgressResponse(BaseModel):
    """Ollama 下载进度响应"""

    id: str = Field(..., description="任务 ID")
    model_name: str = Field(..., alias="modelName", description="模型名称")
    status: str = Field(..., description="状态: pending/downloading/completed/failed")
    progress: float = Field(..., description="进度百分比 (0-100)")
    message: str = Field(..., description="状态消息")
    start_time: str = Field(..., alias="startTime", description="开始时间")
    end_time: str | None = Field(None, alias="endTime", description="结束时间")

    class Config:
        populate_by_name = True


class OllamaDeleteRequest(BaseModel):
    """Ollama 模型删除请求"""

    model_name: str = Field(..., alias="modelName", description="模型名称")


# ============== 模型测试 Schema ==============


class TestStatus(str, Enum):
    """测试状态"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class ModelTestRequest(BaseModel):
    """模型测试请求（通用）"""

    model_id: str = Field(..., description="模型 ID")


class LLMTestRequest(BaseModel):
    """LLM 模型测试请求"""

    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field(..., alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")
    provider: str = Field("", description="服务商")
    test_message: str = Field("Hi", alias="testMessage", description="测试消息")


class MultimodalTestRequest(BaseModel):
    """多模态模型测试请求"""

    model_name: str = Field(..., alias="modelName", description="模型名称")
    base_url: str = Field(..., alias="baseUrl", description="API 地址")
    api_key: str = Field("", alias="apiKey", description="API 密钥")
    image_base64: str | None = Field(None, alias="imageBase64", description="测试图片的 base64 编码")


class TestResultResponse(BaseModel):
    """测试结果响应"""

    status: TestStatus = Field(..., description="测试状态")
    message: str = Field(..., description="响应消息")
    latency_ms: int = Field(0, alias="latencyMs", description="延迟（毫秒）")
    details: dict[str, Any] | None = Field(None, description="详细信息")


__all__ = [
    "LLMConfig",
    "EmbeddingConfig",
    "RerankConfig",
    "VLMConfig",
    "MultimodalConfig",
    "DocumentSplittingConfig",
    "NodeExtractConfig",
    "QuestionGenerationConfig",
    "InitializationRequest",
    "KBModelConfigRequest",
    "EmbeddingTestRequest",
    "RerankTestRequest",
    "RemoteModelCheckRequest",
    "ModelTestResponse",
    "OllamaStatusResponse",
    "KBConfigResponse",
    # Ollama
    "OllamaModelInfo",
    "OllamaPullRequest",
    "OllamaProgressResponse",
    "OllamaDeleteRequest",
    # Model Test
    "TestStatus",
    "ModelTestRequest",
    "LLMTestRequest",
    "MultimodalTestRequest",
    "TestResultResponse",
]
