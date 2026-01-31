"""知识库相关 Schema

对齐 WeKnora99 API 接口规范
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """文档分块配置"""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(default_factory=lambda: ["."])
    enable_multimodal: bool = True


class VLMConfig(BaseModel):
    """视觉语言模型配置

    用于处理知识库中的图像内容
    """

    enabled: bool = Field(default=False, description="是否启用 VLM")
    model_id: str = Field(default="", description="VLM 模型 ID")


class ImageProcessingConfig(BaseModel):
    """图像处理配置"""

    model_id: str = Field(default="", description="图像处理模型 ID")


class CosConfig(BaseModel):
    """腾讯云对象存储配置

    用于存储知识库文件
    """

    secret_id: str = Field(default="", description="腾讯云 Secret ID")
    secret_key: str = Field(default="", description="腾讯云 Secret Key")
    region: str = Field(default="", description="地域")
    bucket_name: str = Field(default="", description="存储桶名称")
    app_id: str = Field(default="", description="应用 ID")
    path_prefix: str = Field(default="", description="路径前缀")


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求

    对齐 WeKnora99 POST /knowledge-bases 请求结构
    """

    name: str = Field(..., min_length=1, max_length=255, description="知识库名称")
    description: str | None = Field(None, max_length=1000, description="知识库描述")
    kb_type: str = Field(
        default="document",
        pattern="^(document|faq)$",
        description="知识库类型",
    )
    chunking_config: ChunkingConfig | None = Field(None, description="分块配置")
    embedding_model_id: str = Field(..., description="嵌入模型 ID")
    summary_model_id: str = Field(..., description="摘要模型 ID")
    rerank_model_id: str | None = Field(None, description="重排序模型 ID")
    vlm_config: VLMConfig | None = Field(None, description="VLM 配置")
    image_processing_config: ImageProcessingConfig | None = Field(
        None, description="图像处理配置"
    )
    cos_config: CosConfig | None = Field(None, description="腾讯云存储配置")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""

    name: str | None = Field(None, min_length=1, max_length=255, description="知识库名称")
    description: str | None = Field(None, max_length=1000, description="知识库描述")
    chunking_config: ChunkingConfig | None = Field(None, description="分块配置")
    embedding_model_id: str | None = Field(None, description="嵌入模型 ID")
    summary_model_id: str | None = Field(None, description="摘要模型 ID")
    rerank_model_id: str | None = Field(None, description="重排序模型 ID")
    vlm_config: VLMConfig | None = Field(None, description="VLM 配置")
    image_processing_config: ImageProcessingConfig | None = Field(
        None, description="图像处理配置"
    )
    cos_config: CosConfig | None = Field(None, description="腾讯云存储配置")


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""

    id: str
    name: str
    description: str | None
    kb_type: str
    chunking_config: dict
    embedding_model_id: str
    summary_model_id: str
    rerank_model_id: str | None = None
    vlm_config: dict | None = None
    image_processing_config: dict | None = None
    cos_config: dict | None = None
    knowledge_count: int = 0
    created_at: datetime
    updated_at: datetime


class HybridSearchRequest(BaseModel):
    """混合搜索请求

    对齐 WeKnora99 SearchParams
    """

    query_text: str = Field(..., min_length=1, description="查询文本")
    vector_threshold: float = Field(default=0.5, ge=0, le=1, description="向量相似度阈值")
    keyword_threshold: float = Field(default=0.3, ge=0, le=1, description="关键词匹配阈值")
    match_count: int = Field(default=5, ge=1, le=100, description="返回结果数量")
    disable_keywords_match: bool = Field(default=False, description="禁用关键词匹配")
    disable_vector_match: bool = Field(default=False, description="禁用向量匹配")
    knowledge_ids: list[str] | None = Field(None, description="知识 ID 过滤")
    tag_ids: list[str] | None = Field(None, description="标签 ID 过滤")
    only_recommended: bool = Field(default=False, description="仅推荐结果")
    enable_rerank: bool = Field(default=False, description="启用重排序")
    rerank_model_id: str | None = Field(None, description="重排序模型 ID")
    top_k: int = Field(default=20, ge=1, le=100, description="检索候选数量")


class HybridSearchResult(BaseModel):
    """混合搜索结果

    对齐 WeKnora99 SearchResult
    """

    content: str
    score: float
    chunk_id: str
    knowledge_id: str
    knowledge_title: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    start_at: int = 0
    end_at: int = 0
    match_type: str = "hybrid"
    chunk_type: str = "text"
    parent_chunk_id: str | None = None
    image_info: str | None = None
    knowledge_filename: str | None = None
    knowledge_source: str | None = None
    chunk_metadata: dict[str, Any] | None = None
    matched_content: str | None = None


class HybridSearchResponse(BaseModel):
    """混合搜索响应"""

    results: list[HybridSearchResult]
    total: int
    query: str
    vector_count: int = 0
    keyword_count: int = 0


class KnowledgeResponse(BaseModel):
    """知识条目响应"""

    id: str
    knowledge_base_id: str
    type: str
    title: str
    source: str
    parse_status: str
    enable_status: str
    file_name: str | None
    file_size: int | None
    chunk_count: int = 0
    created_at: datetime
    processed_at: datetime | None


class KnowledgeUpdate(BaseModel):
    """更新知识条目请求"""

    title: str | None = Field(None, min_length=1, max_length=500, description="知识条目标题")
    enable_status: str | None = Field(
        None, pattern="^(enabled|disabled)$", description="启用状态"
    )


class ManualKnowledgeCreate(BaseModel):
    """手工创建知识请求

    对齐 WeKnora99 POST /knowledge-bases/{id}/knowledge/manual
    """

    title: str = Field(..., min_length=1, max_length=500, description="知识条目标题")
    content: str = Field(..., min_length=1, description="Markdown 内容")


class CopyKnowledgeBaseRequest(BaseModel):
    """拷贝知识库请求

    对齐 WeKnora99 CopyKnowledgeBaseRequest
    """

    task_id: str = Field(default="", description="任务 ID（可选，为空时自动生成）")
    source_id: str = Field(..., description="源知识库 ID")
    target_id: str = Field(default="", description="目标知识库 ID（为空时创建新知识库）")


class CopyKnowledgeBaseResponse(BaseModel):
    """拷贝知识库响应"""

    task_id: str = Field(..., description="任务 ID")
    source_id: str = Field(..., description="源知识库 ID")
    target_id: str = Field(..., description="目标知识库 ID")
    message: str = Field(..., description="响应消息")


# ============== 知识标签 Schema ==============


class TagCreate(BaseModel):
    """创建标签请求

    对齐 WeKnora99 POST /knowledge-bases/:id/tags
    """

    name: str = Field(..., min_length=1, max_length=50, description="标签名称")
    color: str = Field(default="#1890ff", description="标签颜色（十六进制）")
    sort_order: int = Field(default=0, description="排序顺序")


class TagUpdate(BaseModel):
    """更新标签请求"""

    name: str | None = Field(None, min_length=1, max_length=50, description="标签名称")
    color: str | None = Field(None, description="标签颜色（十六进制）")
    sort_order: int | None = Field(None, description="排序顺序")


class TagResponse(BaseModel):
    """标签响应"""

    id: str
    knowledge_base_id: str
    name: str
    color: str
    sort_order: int
    knowledge_count: int = Field(default=0, description="关联的知识数量")
    chunk_count: int = Field(default=0, description="关联的分块数量")
    created_at: datetime
    updated_at: datetime


class TagListResponse(BaseModel):
    """标签列表响应（分页）"""

    total: int
    page: int
    page_size: int
    data: list[TagResponse]


# ============== 知识库复制进度 Schema ==============


class CopyProgressResponse(BaseModel):
    """知识库复制进度响应

    对齐 WeKnora99 GET /knowledge-bases/copy/progress/{task_id}
    """

    task_id: str = Field(..., description="任务 ID")
    source_id: str = Field(..., description="源知识库 ID")
    target_id: str = Field(..., description="目标知识库 ID")
    tenant_id: int = Field(..., description="租户 ID")
    status: str = Field(..., description="任务状态：pending/running/completed/failed/cancelled")
    message: str = Field(..., description="状态消息")
    total_knowledges: int = Field(default=0, description="总知识条目数")
    copied_knowledges: int = Field(default=0, description="已复制知识条目数")
    total_chunks: int = Field(default=0, description="总分块数")
    copied_chunks: int = Field(default=0, description="已复制分块数")
    total_tags: int = Field(default=0, description="总标签数")
    copied_tags: int = Field(default=0, description="已复制标签数")
    error: str | None = Field(None, description="错误信息")
    started_at: datetime | None = Field(None, description="开始时间")
    completed_at: datetime | None = Field(None, description="完成时间")
    progress_percent: float = Field(..., description="进度百分比")


# ============== 独立知识搜索 Schema ==============


class SourceReference(BaseModel):
    """来源引用

    用于知识搜索结果中的文档引用
    """

    chunk_id: str = Field(..., description="分块 ID")
    knowledge_id: str = Field(..., description="知识 ID")
    knowledge_title: str = Field(..., description="知识标题")
    content: str = Field(..., description="引用内容")
    score: float = Field(..., description="相似度分数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class KnowledgeSearchRequest(BaseModel):
    """知识搜索请求

    独立知识搜索，不需要 session_id
    """

    query: str = Field(..., min_length=1, description="查询问题")
    knowledge_base_id: str = Field(..., description="知识库 ID")
    retriever_type: str = Field(
        default="vector",
        pattern="^(vector|bm25|ensemble|conversational)$",
        description="检索器类型",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    score_threshold: float | None = Field(None, ge=0, le=1, description="相似度阈值")
    chat_history: list[dict[str, str]] | None = Field(
        None, description="聊天历史（用于 conversational 模式）"
    )
    enable_rerank: bool = Field(default=False, description="是否启用重排序")
    filter_dict: dict[str, Any] | None = Field(None, description="过滤条件")


class KnowledgeSearchResponse(BaseModel):
    """知识搜索响应"""

    answer: str = Field(..., description="生成的答案")
    sources: list[SourceReference] = Field(
        default_factory=list, description="来源引用"
    )
    question: str | None = Field(None, description="重述后的问题（conversational 模式）")
    retrieved_count: int = Field(default=0, description="检索到的文档数量")


__all__ = [
    "ChunkingConfig",
    "VLMConfig",
    "ImageProcessingConfig",
    "CosConfig",
    "KnowledgeBaseCreate",
    "KnowledgeBaseUpdate",
    "KnowledgeBaseResponse",
    "HybridSearchRequest",
    "HybridSearchResult",
    "HybridSearchResponse",
    "KnowledgeResponse",
    "KnowledgeUpdate",
    "ManualKnowledgeCreate",
    "CopyKnowledgeBaseRequest",
    "CopyKnowledgeBaseResponse",
    "TagCreate",
    "TagUpdate",
    "TagResponse",
    "TagListResponse",
    "CopyProgressResponse",
    "SourceReference",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
]
