"""Elasticsearch API 模式

定义 Elasticsearch 相关的 API 请求/响应模式。
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============== 索引管理 ==============


class IndexCreateRequest(BaseModel):
    """创建索引请求"""

    name: str = Field(..., description="索引名称", min_length=1, max_length=100)
    dimension: int = Field(default=1024, description="向量维度", ge=1, le=4096)
    similarity: Literal["cosine", "dot_product", "l2_norm", "max_inner_product"] = Field(
        default="cosine", description="相似度度量"
    )
    force: bool = Field(default=False, description="是否强制重建（删除已存在的索引）")
    text_field: str = Field(default="text", description="文本字段名")
    vector_field: str = Field(default="vector", description="向量字段名")
    metadata_field: str = Field(default="metadata", description="元数据字段名")


class IndexStatsResponse(BaseModel):
    """索引统计响应"""

    index_name: str = Field(..., description="索引名称")
    doc_count: int = Field(..., description="文档数量")
    store_size: int = Field(..., description="存储大小（字节）")
    dimension: int = Field(..., description="向量维度")
    health: str = Field(..., description="健康状态")
    status: str = Field(..., description="状态")


class IndexListResponse(BaseModel):
    """索引列表响应"""

    indices: list[str] = Field(default_factory=list, description="索引名称列表")
    total: int = Field(..., description="索引总数")


# ============== 文档操作 ==============


class DocumentIndexRequest(BaseModel):
    """文档索引请求"""

    index_name: str = Field(..., description="索引名称")
    doc_id: str | None = Field(default=None, description="文档 ID（可选）")
    text: str = Field(..., description="文档内容", min_length=1)
    metadata: dict[str, Any] | None = Field(default=None, description="元数据")


class DocumentIndexBatchRequest(BaseModel):
    """批量文档索引请求"""

    index_name: str = Field(..., description="索引名称")
    documents: list[tuple[str, dict[str, Any] | None]] = Field(
        ...,
        description="文档列表，每项为 (text, metadata) 元组",
    )
    ids: list[str] | None = Field(default=None, description="文档 ID 列表（可选）")


class DocumentUpdateRequest(BaseModel):
    """文档更新请求"""

    text: str | None = Field(default=None, description="新文本内容")
    metadata: dict[str, Any] | None = Field(default=None, description="新元数据")


class DocumentResponse(BaseModel):
    """文档响应"""

    id: str = Field(..., description="文档 ID")
    text: str = Field(..., description="文档内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class BulkOperationResponse(BaseModel):
    """批量操作响应"""

    total: int = Field(..., description="总数")
    successful: int = Field(..., description="成功数量")
    failed: int = Field(..., description="失败数量")
    errors: list[dict[str, Any]] = Field(default_factory=list, description="错误列表")


# ============== 搜索请求 ==============


@dataclass
class HighlightConfig:
    """高亮配置"""

    enabled: bool = True
    pre_tag: str = "<em>"
    post_tag: str = "</em>"
    fragment_size: int = 150
    number_of_fragments: int = 3


class ElasticsearchSearchRequest(BaseModel):
    """Elasticsearch 搜索请求"""

    index_name: str = Field(..., description="索引名称")
    query: str = Field(..., description="搜索查询", min_length=1)
    k: int = Field(default=5, ge=1, le=100, description="返回结果数量")
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="相似度阈值"
    )
    filter: dict[str, Any] | None = Field(default=None, description="过滤条件")
    enable_highlight: bool = Field(default=False, description="是否启用高亮")
    search_type: Literal["keyword", "vector", "hybrid"] = Field(
        default="hybrid", description="搜索类型"
    )


class HybridSearchRequest(BaseModel):
    """混合搜索请求"""

    index_name: str = Field(..., description="索引名称")
    query: str = Field(..., description="搜索查询", min_length=1)
    k: int = Field(default=5, ge=1, le=100, description="返回结果数量")
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="文本搜索权重")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="向量搜索权重")
    filter: dict[str, Any] | None = Field(default=None, description="过滤条件")


class RawSearchRequest(BaseModel):
    """原始查询请求"""

    index_name: str = Field(..., description="索引名称")
    query: dict[str, Any] = Field(..., description="Elasticsearch 查询 DSL")


# ============== 搜索结果 ==============


class HighlightItem(BaseModel):
    """高亮项"""

    field: str = Field(..., description="字段名")
    fragments: list[str] = Field(default_factory=list, description="高亮片段")


class ElasticsearchSearchResult(BaseModel):
    """Elasticsearch 搜索结果"""

    id: str | None = Field(default=None, description="文档 ID")
    content: str = Field(..., description="文档内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    score: float = Field(..., description="相似度得分", ge=0.0)
    highlights: list[str] | None = Field(default=None, description="高亮片段")


class ElasticsearchSearchResponse(BaseModel):
    """Elasticsearch 搜索响应"""

    results: list[ElasticsearchSearchResult] = Field(
        default_factory=list, description="搜索结果"
    )
    total: int = Field(..., description="结果总数")
    query: str = Field(..., description="原始查询")
    index_name: str = Field(..., description="索引名称")
    response_time_ms: int | None = Field(default=None, description="响应时间（毫秒）")


# ============== 分析器 ==============


class AnalyzeRequest(BaseModel):
    """文本分析请求"""

    index_name: str = Field(..., description="索引名称")
    text: str = Field(..., description="待分析文本", min_length=1)
    analyzer: str | None = Field(default=None, description="分析器名称")


class AnalyzeToken(BaseModel):
    """分词结果"""

    token: str = Field(..., description="分词")
    start_offset: int | None = Field(default=None, description="起始偏移")
    end_offset: int | None = Field(default=None, description="结束偏移")
    position: int | None = Field(default=None, description="位置")
    type: str | None = Field(default=None, description="类型")


class AnalyzeResponse(BaseModel):
    """文本分析响应"""

    tokens: list[AnalyzeToken] = Field(default_factory=list, description="分词结果")


# ============== 配置 ==============


class ElasticsearchConfigResponse(BaseModel):
    """Elasticsearch 配置响应"""

    url: str | None = Field(default=None, description="服务器 URL")
    cloud_id: str | None = Field(default=None, description="Cloud ID")
    index_name: str | None = Field(default=None, description="默认索引名称")
    strategy: str = Field(default="dense", description="向量策略")
    dimension: int = Field(default=1024, description="向量维度")
    similarity: str = Field(default="cosine", description="相似度度量")


# ============== 健康检查 ==============


class ElasticsearchHealthResponse(BaseModel):
    """Elasticsearch 健康检查响应"""

    status: str = Field(..., description="状态 (healthy/unhealthy)")
    cluster_name: str | None = Field(default=None, description="集群名称")
    version: str | None = Field(default=None, description="版本")
    timestamp: float = Field(..., description="时间戳")


__all__ = [
    "IndexCreateRequest",
    "IndexStatsResponse",
    "IndexListResponse",
    "DocumentIndexRequest",
    "DocumentIndexBatchRequest",
    "DocumentUpdateRequest",
    "DocumentResponse",
    "BulkOperationResponse",
    "ElasticsearchSearchRequest",
    "HybridSearchRequest",
    "RawSearchRequest",
    "ElasticsearchSearchResult",
    "ElasticsearchSearchResponse",
    "AnalyzeRequest",
    "AnalyzeToken",
    "AnalyzeResponse",
    "ElasticsearchConfigResponse",
    "ElasticsearchHealthResponse",
    "HighlightConfig",
]
