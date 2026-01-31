"""搜索 API 模式

定义搜索相关的请求和响应模型。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============== 搜索请求 ==============


class SearchRequest(BaseModel):
    """搜索请求"""

    query: str = Field(..., description="搜索查询字符串", min_length=1)
    index: str = Field(..., description="索引名称")
    fields: list[str] | None = Field(None, description="搜索字段列表（默认全字段）")
    filters: dict[str, Any] | None = Field(None, description="过滤条件")
    size: int = Field(10, ge=1, le=100, description="返回结果数")
    page: int = Field(1, ge=1, description="页码")
    sort: list[str] | None = Field(None, description="排序，如 ['created_at:desc']")
    highlight: bool = Field(True, description="是否高亮")
    highlight_fields: list[str] | None = Field(None, description="高亮字段")


class HybridSearchRequest(BaseModel):
    """混合搜索请求（ES + 向量）"""

    query: str = Field(..., description="搜索查询字符串", min_length=1)
    index: str = Field(..., description="索引名称")
    vector_weight: float = Field(0.5, ge=0, le=1, description="向量检索权重")
    text_weight: float = Field(0.5, ge=0, le=1, description="全文检索权重")
    filters: dict[str, Any] | None = Field(None, description="过滤条件")
    size: int = Field(10, ge=1, le=100, description="返回结果数")
    page: int = Field(1, ge=1, description="页码")


class SuggestRequest(BaseModel):
    """搜索建议请求"""

    query: str = Field(..., description="查询前缀", min_length=1)
    index: str = Field(..., description="索引名称")
    field: str = Field("suggest", description="建议字段")
    size: int = Field(5, ge=1, le=20, description="建议数量")


class AggregationRequest(BaseModel):
    """聚合查询请求"""

    index: str = Field(..., description="索引名称")
    query: str | None = Field(None, description="查询字符串（可选）")
    aggregations: dict[str, Any] = Field(..., description="聚合配置")
    filters: dict[str, Any] | None = Field(None, description="过滤条件")


class IndexDocumentRequest(BaseModel):
    """索引文档请求"""

    index: str = Field(..., description="索引名称")
    documents: list[dict[str, Any]] = Field(..., description="文档列表")
    ids: list[str] | None = Field(None, description="文档 ID 列表")
    refresh: bool = Field(False, description="是否立即刷新")


# ============== 搜索响应 ==============


class Highlight(BaseModel):
    """高亮结果"""

    field: str = Field(..., description="字段名")
    fragments: list[str] = Field(default_factory=list, description="高亮片段")


class SearchResultItem(BaseModel):
    """搜索结果项"""

    id: str = Field(..., description="文档 ID")
    score: float = Field(..., description="相关度分数")
    source: dict[str, Any] = Field(default_factory=dict, description="文档内容")
    highlight: dict[str, list[str]] | None = Field(None, description="高亮结果")


class SearchResponse(BaseModel):
    """搜索响应"""

    success: bool = Field(..., description="是否成功")
    hits: list[SearchResultItem] = Field(default_factory=list, description="搜索结果")
    total: int = Field(..., description="总结果数")
    max_score: float | None = Field(None, description="最高分数")
    aggregations: dict[str, Any] | None = Field(None, description="聚合结果")
    took: int = Field(..., description="耗时（毫秒）")

    @classmethod
    def from_search_response(
        cls,
        response: "SearchResponse",
    ) -> "SearchResponse":
        """从内部搜索响应构建"""
        return cls(
            success=True,
            hits=[SearchResultItem(
                id=hit.id,
                score=hit.score,
                source=hit.source,
                highlight=hit.highlight,
            ) for hit in response.hits],
            total=response.total,
            max_score=response.max_score,
            aggregations=response.aggregations,
            took=response.took,
        )


class SuggestResponse(BaseModel):
    """搜索建议响应"""

    success: bool = Field(..., description="是否成功")
    suggestions: list[str] = Field(default_factory=list, description="建议列表")


class AggregationResponse(BaseModel):
    """聚合响应"""

    success: bool = Field(..., description="是否成功")
    aggregations: dict[str, Any] = Field(default_factory=dict, description="聚合结果")


class IndexResponse(BaseModel):
    """索引响应"""

    success: bool = Field(..., description="是否成功")
    indexed: int = Field(..., description="成功索引数")
    failed: int = Field(..., description="失败数")
    message: str | None = Field(None, description="提示信息")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="集群状态 (green/yellow/red)")
    cluster_name: str | None = Field(None, description="集群名称")
    number_of_nodes: int | None = Field(None, description="节点数量")
    active_shards: int | None = Field(None, description="活跃分片数")


class IndexInfo(BaseModel):
    """索引信息"""

    name: str = Field(..., description="索引名称")
    document_count: int = Field(..., description="文档数量")
    size: str | None = Field(None, description="索引大小")


class IndicesResponse(BaseModel):
    """索引列表响应"""

    success: bool = Field(..., description="是否成功")
    indices: list[IndexInfo] = Field(default_factory=list, description="索引列表")
