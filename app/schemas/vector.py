"""向量 API 模式

定义向量存储相关的 API 请求/响应模式。
"""

from typing import Any

from pydantic import BaseModel, Field

# ============== 索引请求 ==============


class VectorIndexRequest(BaseModel):
    """向量索引请求

    用于将文档添加到向量存储。
    """

    documents: list[str] = Field(..., description="文档文本列表")
    metadatas: list[dict[str, Any]] | None = Field(
        default=None, description="文档元数据列表"
    )
    ids: list[str] | None = Field(default=None, description="可选的文档 ID 列表")
    collection_name: str = Field(default="default", description="集合名称")


class VectorIndexBatchRequest(BaseModel):
    """批量向量索引请求

    支持大规模文档索引。
    """

    documents: list[str] = Field(..., description="文档文本列表")
    metadatas: list[dict[str, Any]] | None = Field(
        default=None, description="文档元数据列表"
    )
    collection_name: str = Field(default="default", description="集合名称")
    batch_size: int = Field(default=100, description="批处理大小")
    chunk_size: int = Field(default=1000, description="文本分块大小")
    chunk_overlap: int = Field(default=200, description="分块重叠大小")


# ============== 搜索请求 ==============


class VectorSearchRequest(BaseModel):
    """向量搜索请求

    用于语义搜索。
    """

    query: str = Field(..., description="搜索查询", min_length=1)
    k: int = Field(default=5, ge=1, le=100, description="返回结果数量")
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="相似度阈值"
    )
    filter_dict: dict[str, Any] | None = Field(default=None, description="过滤条件")
    collection_name: str = Field(default="default", description="集合名称")


class HybridSearchRequest(VectorSearchRequest):
    """混合搜索请求

    结合向量搜索和关键词搜索。
    """

    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="关键词搜索权重")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="向量搜索权重")


# ============== 搜索结果 ==============


class SearchResultItem(BaseModel):
    """搜索结果项"""

    content: str = Field(..., description="文档内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    score: float = Field(..., description="相似度得分", ge=0.0, le=1.0)
    id: str | None = Field(default=None, description="文档 ID")


class VectorSearchResponse(BaseModel):
    """向量搜索响应"""

    results: list[SearchResultItem] = Field(default_factory=list, description="搜索结果")
    total: int = Field(..., description="结果总数")
    query: str = Field(..., description="原始查询")
    collection_name: str = Field(..., description="集合名称")
    response_time_ms: int | None = Field(default=None, description="响应时间（毫秒）")


# ============== 索引响应 ==============


class VectorIndexResponse(BaseModel):
    """向量索引响应"""

    ids: list[str] = Field(default_factory=list, description="文档 ID 列表")
    count: int = Field(..., description="成功索引的文档数量")
    failed: int = Field(default=0, description="失败的文档数量")
    collection_name: str = Field(..., description="集合名称")


# ============== 统计信息 ==============


class VectorStatsResponse(BaseModel):
    """向量存储统计响应"""

    total_vectors: int = Field(..., description="向量总数")
    collections: int = Field(..., description="集合数量")
    dimension: int = Field(..., description="向量维度")
    metric: str = Field(..., description="相似度度量方式")
    store_type: str = Field(..., description="存储类型")


# ============== 删除请求 ==============


class VectorDeleteRequest(BaseModel):
    """向量删除请求"""

    ids: list[str] = Field(..., description="要删除的文档 ID 列表")
    collection_name: str = Field(default="default", description="集合名称")


# ============== 集合管理 ==============


class CollectionCreateRequest(BaseModel):
    """创建集合请求"""

    name: str = Field(..., description="集合名称", min_length=1, max_length=100)
    dimension: int = Field(default=1024, description="向量维度")
    metric: str = Field(default="cosine", description="相似度度量")


class CollectionResponse(BaseModel):
    """集合响应"""

    name: str = Field(..., description="集合名称")
    dimension: int = Field(..., description="向量维度")
    metric: str = Field(..., description="相似度度量")
    vector_count: int = Field(default=0, description="向量数量")


class CollectionsListResponse(BaseModel):
    """集合列表响应"""

    collections: list[CollectionResponse] = Field(default_factory=list)
    total: int = Field(..., description="集合总数")


__all__ = [
    "VectorIndexRequest",
    "VectorIndexBatchRequest",
    "VectorSearchRequest",
    "HybridSearchRequest",
    "SearchResultItem",
    "VectorSearchResponse",
    "VectorIndexResponse",
    "VectorStatsResponse",
    "VectorDeleteRequest",
    "CollectionCreateRequest",
    "CollectionResponse",
    "CollectionsListResponse",
]
