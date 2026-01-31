"""搜索服务

提供全文搜索、混合搜索和聚合查询的业务逻辑。

使用示例:
    ```python
    from app.services.search_service import get_search_service

    service = await get_search_service()

    # 全文搜索
    results = await service.search(
        index="documents",
        query="关键词",
        user_id=123
    )

    # 混合搜索
    results = await service.hybrid_search(
        index="documents",
        query="关键词",
        user_id=123
    )
    ```
"""

from __future__ import annotations

from typing import Any

from app.observability.logging import get_logger
from app.schemas.search import (
    AggregationRequest,
    AggregationResponse,
    HybridSearchRequest,
    IndexDocumentRequest,
    IndexResponse,
    SuggestRequest,
    SuggestResponse,
    SearchRequest as SearchRequestSchema,
    SearchResponse as SearchResponseSchema,
)
from app.services.search.elasticsearch import ElasticsearchClient, get_elasticsearch_client
from app.services.search.indexer import DocumentIndexer
from app.services.search.query import QueryBuilder, SearchResponse

logger = get_logger(__name__)


class SearchService:
    """搜索服务

    提供全文搜索、混合搜索、聚合查询等功能。
    支持多租户隔离。
    """

    def __init__(
        self,
        client: ElasticsearchClient | None = None,
        index_prefix: str = "",
    ):
        """初始化搜索服务

        Args:
            client: Elasticsearch 客户端
            index_prefix: 索引前缀（用于多租户）
        """
        self._client: ElasticsearchClient | None = client
        self._index_prefix = index_prefix
        self._indexer: DocumentIndexer | None = None

    @property
    async def client(self) -> ElasticsearchClient:
        """获取 Elasticsearch 客户端"""
        if self._client is None:
            self._client = await get_elasticsearch_client()
        return self._client

    @property
    def indexer(self) -> DocumentIndexer:
        """获取文档索引器"""
        if self._indexer is None:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            self._indexer = DocumentIndexer(self._client)
        return self._indexer

    def _get_tenant_index(self, index: str, tenant_id: int | None) -> str:
        """获取租户隔离的索引名

        Args:
            index: 原始索引名
            tenant_id: 租户 ID

        Returns:
            带租户前缀的索引名
        """
        if tenant_id:
            return f"tenant_{tenant_id}_{index}"
        return index

    # ============== 文档索引 ==============

    async def index_document(
        self,
        index: str,
        document: dict[str, Any],
        id: str | None = None,
        tenant_id: int | None = None,
        refresh: bool = False,
    ) -> str | None:
        """索引单个文档

        Args:
            index: 索引名称
            document: 文档内容
            id: 文档 ID
            tenant_id: 租户 ID
            refresh: 是否立即刷新

        Returns:
            文档 ID
        """
        client = await self.client
        resolved_index = self._get_tenant_index(index, tenant_id)

        logger.info("indexing_document", index=resolved_index, id=id)

        return await client.index_document(
            index=resolved_index,
            id=id,
            document=document,
            refresh=refresh,
        )

    async def bulk_index(
        self,
        request: IndexDocumentRequest,
        tenant_id: int | None = None,
    ) -> IndexResponse:
        """批量索引文档

        Args:
            request: 索引请求
            tenant_id: 租户 ID

        Returns:
            索引响应
        """
        client = await self.client
        resolved_index = self._get_tenant_index(request.index, tenant_id)

        # 确保索引存在
        if not await client.index_exists(resolved_index):
            await self._create_default_index(client, resolved_index)

        indexer = DocumentIndexer(client)
        success, failed = await indexer.bulk_index(
            index=resolved_index,
            documents=request.documents,
            ids=request.ids,
            refresh=request.refresh,
        )

        logger.info(
            "bulk_index_completed",
            index=resolved_index,
            success=success,
            failed=failed,
        )

        return IndexResponse(
            success=failed == 0,
            indexed=success,
            failed=failed,
            message=f"成功索引 {success} 个文档" + (f"，{failed} 个失败" if failed > 0 else ""),
        )

    # ============== 全文搜索 ==============

    async def search(
        self,
        request: SearchRequestSchema,
        tenant_id: int | None = None,
    ) -> SearchResponseSchema:
        """执行全文搜索

        Args:
            request: 搜索请求
            tenant_id: 租户 ID

        Returns:
            搜索响应
        """
        client = await self.client
        resolved_index = self._get_tenant_index(request.index, tenant_id)

        # 构建查询
        builder = QueryBuilder()

        # 设置搜索字段
        fields = request.fields or ["_all"]

        # 多字段搜索
        if len(fields) == 1:
            builder.must_match(fields[0], request.query)
        else:
            builder.multi_match(request.query, fields)

        # 添加过滤条件
        if request.filters:
            for field, value in request.filters.items():
                if isinstance(value, list):
                    builder.terms(field, value)
                elif isinstance(value, dict):
                    # 范围查询
                    if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                        builder.range_query(field, **value)
                    else:
                        builder.term(field, value)
                else:
                    builder.term(field, value)

        # 排序
        if request.sort:
            for sort_item in request.sort:
                parts = sort_item.split(":")
                field = parts[0]
                order = "desc" if len(parts) > 1 and parts[1].lower() == "desc" else "asc"
                builder.sort(field, order)
        else:
            builder.sort_score()

        # 高亮
        if request.highlight:
            highlight_fields = request.highlight_fields or fields
            builder.highlight(highlight_fields)

        # 分页
        builder.paginate(request.page, request.size)

        # 执行搜索
        query = builder.build()
        response = await client.search(
            index=resolved_index,
            query=query["query"],
            size=query.get("size"),
            from_=query.get("from", 0),
            sort=query.get("sort"),
            highlight=query.get("highlight"),
            source=query.get("_source"),
        )

        # 转换结果
        search_response = SearchResponse.from_es_response(response)

        logger.info(
            "search_completed",
            index=resolved_index,
            query=request.query,
            hits=len(search_response.hits),
            total=search_response.total,
        )

        return SearchResponseSchema.from_search_response(search_response)

    # ============== 混合搜索 ==============

    async def hybrid_search(
        self,
        request: HybridSearchRequest,
        tenant_id: int | None = None,
    ) -> SearchResponseSchema:
        """执行混合搜索（全文检索 + 向量检索）

        Args:
            request: 混合搜索请求
            tenant_id: 租户 ID

        Returns:
            搜索响应
        """
        client = await self.client
        resolved_index = self._get_tenant_index(request.index, tenant_id)

        # 1. 执行全文检索
        text_builder = QueryBuilder()
        text_builder.multi_match(request.query, ["title", "content"])
        if request.filters:
            for field, value in request.filters.items():
                if isinstance(value, list):
                    text_builder.terms(field, value)
                else:
                    text_builder.term(field, value)

        text_builder.sort_score()
        text_builder.paginate(request.page, request.size * 2)

        text_query = text_builder.build()
        text_response = await client.search(
            index=resolved_index,
            query=text_query["query"],
            size=text_query.get("size"),
            from_=text_query.get("from", 0),
        )

        text_results = SearchResponse.from_es_response(text_response)

        # 2. 执行向量检索（如果有向量字段）
        # TODO: 集成向量检索
        vector_results: SearchResponse | None = None

        # 3. 合并结果（RRF - Reciprocal Rank Fusion）
        merged_results = self._merge_results(
            text_results,
            vector_results,
            text_weight=request.text_weight,
            vector_weight=request.vector_weight,
            size=request.size,
        )

        logger.info(
            "hybrid_search_completed",
            index=resolved_index,
            query=request.query,
            hits=len(merged_results.hits),
        )

        return SearchResponseSchema.from_search_response(merged_results)

    def _merge_results(
        self,
        text_results: SearchResponse,
        vector_results: SearchResponse | None,
        text_weight: float,
        vector_weight: float,
        size: int,
    ) -> SearchResponse:
        """合并全文和向量搜索结果

        Args:
            text_results: 全文搜索结果
            vector_results: 向量搜索结果
            text_weight: 全文权重
            vector_weight: 向量权重
            size: 返回结果数

        Returns:
            合并后的搜索结果
        """
        if vector_results is None or not vector_results.hits:
            return text_results

        # RRF 算法
        k = 60  # RRF 常数

        scores: dict[str, float] = {}
        docs: dict[str, Any] = {}

        # 处理全文搜索结果
        for rank, hit in enumerate(text_results.hits):
            doc_id = hit.id
            rrf_score = 1 / (k + rank + 1)
            scores[doc_id] = text_weight * rrf_score
            docs[doc_id] = hit

        # 处理向量搜索结果
        for rank, hit in enumerate(vector_results.hits):
            doc_id = hit.id
            rrf_score = 1 / (k + rank + 1)
            if doc_id in scores:
                scores[doc_id] += vector_weight * rrf_score
            else:
                scores[doc_id] = vector_weight * rrf_score
                docs[doc_id] = hit

        # 排序并取前 N 个
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:size]

        from app.services.search.query import SearchResult

        merged_hits = [
            SearchResult(
                id=doc_id,
                score=score,
                source=docs[doc_id].source,
                highlight=docs[doc_id].highlight,
            )
            for doc_id, score in sorted_docs
        ]

        return SearchResponse(
            hits=merged_hits,
            total=len(merged_hits),
            max_score=max((s for _, s in sorted_docs), default=0.0),
            took=text_results.took + (vector_results.tok if vector_results else 0),
        )

    # ============== 搜索建议 ==============

    async def suggest(
        self,
        request: SuggestRequest,
        tenant_id: int | None = None,
    ) -> SuggestResponse:
        """获取搜索建议

        Args:
            request: 建议请求
            tenant_id: 租户 ID

        Returns:
            建议响应
        """
        client = await self.client
        resolved_index = self._get_tenant_index(request.index, tenant_id)

        try:
            # 使用 completion suggester
            response = await client.client.search(
                index=resolved_index,
                body={
                    "suggest": {
                        "prefix_suggest": {
                            "prefix": request.query,
                            "completion": {
                                "field": request.field,
                                "size": request.size,
                            },
                        }
                    }
                },
            )

            suggestions: list[str] = []
            suggest_result = response.get("suggest", {}).get("prefix_suggest", [])
            for option in suggest_result:
                for item in option.get("options", []):
                    text = item.get("text", "")
                    if text:
                        suggestions.append(text)

            logger.debug("suggest_completed", index=resolved_index, count=len(suggestions))

            return SuggestResponse(success=True, suggestions=suggestions[: request.size])

        except Exception as e:
            logger.warning("suggest_failed", index=resolved_index, error=str(e))

            # 降级：使用 prefix 查询
            builder = QueryBuilder().prefix(request.field, request.query).size(request.size)
            response = await client.search(
                index=resolved_index,
                query=builder.build_query_only(),
                size=request.size,
            )

            suggestions = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                value = source.get(request.field, "")
                if value:
                    suggestions.append(value)

            return SuggestResponse(success=True, suggestions=suggestions)

    # ============== 聚合查询 ==============

    async def aggregate(
        self,
        request: AggregationRequest,
        tenant_id: int | None = None,
    ) -> AggregationResponse:
        """执行聚合查询

        Args:
            request: 聚合请求
            tenant_id: 租户 ID

        Returns:
            聚合响应
        """
        client = await self.client
        resolved_index = self._get_tenant_index(request.index, tenant_id)

        # 构建查询
        builder = QueryBuilder()

        if request.query:
            builder.multi_match(request.query, ["_all"])

        if request.filters:
            for field, value in request.filters.items():
                if isinstance(value, list):
                    builder.terms(field, value)
                else:
                    builder.term(field, value)

        # 添加聚合
        for name, agg_config in request.aggregations.items():
            agg_type = agg_config.get("type")
            params = agg_config.get("params", {})

            if agg_type == "terms":
                builder.terms_aggregation(name, **params)
            elif agg_type == "date_histogram":
                builder.date_histogram_aggregation(name, **params)
            elif agg_type == "range":
                builder.range_aggregation(name, **params)
            elif agg_type == "stats":
                builder.stats_aggregation(name, **params)

        # 执行聚合
        query = builder.build_query_only()
        aggregations = await client.aggregate(
            index=resolved_index,
            aggs=builder._aggregations,
            query=query if request.query or request.filters else None,
            size=0,
        )

        logger.info("aggregate_completed", index=resolved_index)

        return AggregationResponse(success=True, aggregations=aggregations)

    # ============== 索引管理 ==============

    async def create_index(
        self,
        index: str,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        tenant_id: int | None = None,
    ) -> bool:
        """创建索引

        Args:
            index: 索引名称
            mappings: 索引映射
            settings: 索引设置
            tenant_id: 租户 ID

        Returns:
            是否成功
        """
        client = await self.client
        resolved_index = self._get_tenant_index(index, tenant_id)

        return await client.create_index(resolved_index, mappings, settings)

    async def delete_index(
        self,
        index: str,
        tenant_id: int | None = None,
    ) -> bool:
        """删除索引

        Args:
            index: 索引名称
            tenant_id: 租户 ID

        Returns:
            是否成功
        """
        client = await self.client
        resolved_index = self._get_tenant_index(index, tenant_id)

        return await client.delete_index(resolved_index)

    async def index_exists(
        self,
        index: str,
        tenant_id: int | None = None,
    ) -> bool:
        """检查索引是否存在

        Args:
            index: 索引名称
            tenant_id: 租户 ID

        Returns:
            是否存在
        """
        client = await self.client
        resolved_index = self._get_tenant_index(index, tenant_id)

        return await client.index_exists(resolved_index)

    # ============== 健康检查 ==============

    async def health(self) -> dict[str, Any]:
        """获取集群健康状态

        Returns:
            健康状态信息
        """
        client = await self.client
        return await client.cluster_health()

    # ============== 私有方法 ==============

    async def _create_default_index(
        self,
        client: ElasticsearchClient,
        index: str,
    ) -> None:
        """创建默认索引

        Args:
            client: ES 客户端
            index: 索引名称
        """
        mappings = {
            "properties": {
                "id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "tags": {"type": "keyword"},
                "category": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "tenant_id": {"type": "integer"},
                "suggest": {
                    "type": "completion",
                    "analyzer": "standard",
                },
            }
        }

        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 1,
        }

        await client.create_index(index, mappings, settings)

        logger.info("default_index_created", index=index)


# ============== 依赖注入工厂 ==============


_service: SearchService | None = None


async def get_search_service(
    tenant_id: int | None = None,
) -> SearchService:
    """获取搜索服务实例

    Args:
        tenant_id: 租户 ID（用于索引前缀）

    Returns:
        SearchService 实例
    """
    global _service

    if _service is None:
        index_prefix = f"tenant_{tenant_id}" if tenant_id else ""
        _service = SearchService(index_prefix=index_prefix)

    return _service


def reset_search_service() -> None:
    """重置搜索服务（主要用于测试）"""
    global _service
    _service = None
