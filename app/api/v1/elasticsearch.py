"""Elasticsearch API

提供 Elasticsearch 索引管理、文档操作、搜索等功能的 RESTful API。

使用示例:
```bash
# 创建索引
POST /api/v1/elasticsearch/indices
{
  "name": "documents",
  "dimension": 1024,
  "similarity": "cosine"
}

# 添加文档
POST /api/v1/elasticsearch/documents
{
  "index_name": "documents",
  "text": "文档内容",
  "metadata": {"title": "标题"}
}

# 搜索
POST /api/v1/elasticsearch/search
{
  "index_name": "documents",
  "query": "搜索内容",
  "k": 5
}

# 混合搜索
POST /api/v1/elasticsearch/search/hybrid
{
  "index_name": "documents",
  "query": "搜索内容",
  "text_weight": 0.3,
  "vector_weight": 0.7
}
```
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, status

from app.api.v1.auth import get_current_user
from app.models.database import User
from app.observability.logging import get_logger
from app.schemas.elasticsearch import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeToken,
    BulkOperationResponse,
    DocumentIndexBatchRequest,
    DocumentIndexRequest,
    DocumentResponse,
    DocumentUpdateRequest,
    ElasticsearchConfigResponse,
    ElasticsearchHealthResponse,
    ElasticsearchSearchRequest,
    ElasticsearchSearchResponse,
    ElasticsearchSearchResult,
    HybridSearchRequest,
    IndexCreateRequest,
    IndexListResponse,
    IndexStatsResponse,
    RawSearchRequest,
)
from app.schemas.response import ApiResponse
from app.services.elasticsearch_service import (
    BulkResult,
    ElasticsearchService,
    IndexMapping,
    SearchOptions,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/elasticsearch", tags=["Elasticsearch"])


# ============== 依赖注入 ==============


async def get_elasticsearch_service(
    current_user: Annotated[User, Depends(get_current_user)],
) -> ElasticsearchService:
    """获取 Elasticsearch 服务实例

    自动注入租户 ID 以实现多租户隔离。
    """
    tenant_id = current_user.tenant_id
    service = ElasticsearchService(tenant_id=tenant_id)
    await service.initialize()
    return service


# ============== 健康检查 ==============


@router.get(
    "/health",
    response_model=ApiResponse[ElasticsearchHealthResponse],
    summary="健康检查",
    description="检查 Elasticsearch 服务是否健康。",
)
async def health_check(
    request: Request,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[ElasticsearchHealthResponse]:
    """健康检查"""
    health_info = await service.health_check()

    return ApiResponse.ok(
        data=ElasticsearchHealthResponse(**health_info),
        message="Elasticsearch 服务正常" if health_info["status"] == "healthy" else "Elasticsearch 服务不可用",
    )


@router.get(
    "/config",
    response_model=ApiResponse[ElasticsearchConfigResponse],
    summary="获取配置",
    description="获取当前 Elasticsearch 配置信息。",
)
async def get_config(
    request: Request,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[ElasticsearchConfigResponse]:
    """获取配置"""
    config = service.config

    return ApiResponse.ok(
        data=ElasticsearchConfigResponse(
            url=config.url,
            cloud_id=config.cloud_id,
            index_name=config.index_name,
            strategy=config.strategy,
            dimension=config.dimension,
            similarity=config.similarity,
        )
    )


# ============== 索引管理 ==============


@router.post(
    "/indices",
    response_model=ApiResponse[IndexStatsResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建索引",
    description="创建新的 Elasticsearch 索引。",
)
async def create_index(
    request: Request,
    data: IndexCreateRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[IndexStatsResponse]:
    """创建索引"""
    mapping = IndexMapping(
        text_field=data.text_field,
        vector_field=data.vector_field,
        metadata_field=data.metadata_field,
        dimension=data.dimension,
        similarity=data.similarity,
    )

    success = await service.create_index(
        index_name=data.name,
        mapping=mapping,
        force=data.force,
    )

    if not success:
        return ApiResponse.fail(
            error="index_creation_failed",
            message=f"创建索引 {data.name} 失败",
        )

    # 获取统计
    stats = await service.get_index_stats(data.name)

    return ApiResponse.ok(
        data=IndexStatsResponse(
            index_name=data.name,
            doc_count=stats.doc_count if stats else 0,
            store_size=stats.store_size if stats else 0,
            dimension=data.dimension,
            health=stats.health if stats else "yellow",
            status=stats.status if stats else "open",
        ),
        message=f"索引 {data.name} 创建成功",
    )


@router.get(
    "/indices",
    response_model=ApiResponse[IndexListResponse],
    summary="列出索引",
    description="获取所有 Elasticsearch 索引列表。",
)
async def list_indices(
    request: Request,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[IndexListResponse]:
    """列出索引"""
    indices = await service.list_indices()

    return ApiResponse.ok(
        data=IndexListResponse(
            indices=indices,
            total=len(indices),
        )
    )


@router.get(
    "/indices/{index_name}",
    response_model=ApiResponse[IndexStatsResponse],
    summary="获取索引统计",
    description="获取指定索引的统计信息。",
)
async def get_index_stats(
    request: Request,
    index_name: str,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[IndexStatsResponse]:
    """获取索引统计"""
    stats = await service.get_index_stats(index_name)

    if stats is None:
        return ApiResponse.fail(
            error="index_not_found",
            message=f"索引 {index_name} 不存在",
        )

    return ApiResponse.ok(
        data=IndexStatsResponse(
            index_name=stats.index_name,
            doc_count=stats.doc_count,
            store_size=stats.store_size,
            dimension=stats.dimension,
            health=stats.health,
            status=stats.status,
        )
    )


@router.delete(
    "/indices/{index_name}",
    response_model=ApiResponse[dict[str, str]],
    summary="删除索引",
    description="删除指定的 Elasticsearch 索引。",
)
async def delete_index(
    request: Request,
    index_name: str,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[dict[str, str]]:
    """删除索引"""
    success = await service.delete_index(index_name)

    if not success:
        return ApiResponse.fail(
            error="index_not_found",
            message=f"索引 {index_name} 不存在或删除失败",
        )

    return ApiResponse.ok(
        data={"status": "deleted"},
        message=f"索引 {index_name} 已删除",
    )


# ============== 文档操作 ==============


@router.post(
    "/documents",
    response_model=ApiResponse[dict[str, str]],
    summary="添加文档",
    description="向索引添加单个文档。",
)
async def add_document(
    request: Request,
    data: DocumentIndexRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[dict[str, str]]:
    """添加文档"""
    import uuid

    doc_id = data.doc_id or str(uuid.uuid4())
    success = await service.add_document(
        index_name=data.index_name,
        doc_id=doc_id,
        text=data.text,
        metadata=data.metadata,
    )

    if not success:
        return ApiResponse.fail(
            error="document_index_failed",
            message="文档索引失败",
        )

    return ApiResponse.ok(
        data={"doc_id": doc_id},
        message="文档添加成功",
    )


@router.post(
    "/documents/batch",
    response_model=ApiResponse[BulkOperationResponse],
    summary="批量添加文档",
    description="批量向索引添加文档。",
)
async def add_documents_batch(
    request: Request,
    data: DocumentIndexBatchRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[BulkOperationResponse]:
    """批量添加文档"""
    result: BulkResult = await service.add_documents(
        index_name=data.index_name,
        documents=data.documents,
        ids=data.ids,
    )

    return ApiResponse.ok(
        data=BulkOperationResponse(
            total=result.total,
            successful=result.successful,
            failed=result.failed,
            errors=result.errors,
        ),
        message=f"成功索引 {result.successful} 个文档",
    )


@router.get(
    "/documents/{index_name}/{doc_id}",
    response_model=ApiResponse[DocumentResponse],
    summary="获取文档",
    description="根据 ID 获取文档内容。",
)
async def get_document(
    request: Request,
    index_name: str,
    doc_id: str,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[DocumentResponse]:
    """获取文档"""
    doc = await service.get_document(index_name, doc_id)

    if doc is None:
        return ApiResponse.fail(
            error="document_not_found",
            message=f"文档 {doc_id} 不存在",
        )

    return ApiResponse.ok(
        data=DocumentResponse(
            id=doc_id,
            text=doc.get("text", ""),
            metadata=doc.get("metadata", {}),
        )
    )


@router.put(
    "/documents/{index_name}/{doc_id}",
    response_model=ApiResponse[dict[str, str]],
    summary="更新文档",
    description="更新指定文档的内容或元数据。",
)
async def update_document(
    request: Request,
    index_name: str,
    doc_id: str,
    data: DocumentUpdateRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[dict[str, str]]:
    """更新文档"""
    success = await service.update_document(
        index_name=index_name,
        doc_id=doc_id,
        text=data.text,
        metadata=data.metadata,
    )

    if not success:
        return ApiResponse.fail(
            error="document_update_failed",
            message=f"文档 {doc_id} 更新失败",
        )

    return ApiResponse.ok(
        data={"doc_id": doc_id},
        message="文档更新成功",
    )


@router.delete(
    "/documents/{index_name}/{doc_id}",
    response_model=ApiResponse[dict[str, str]],
    summary="删除文档",
    description="删除指定文档。",
)
async def delete_document(
    request: Request,
    index_name: str,
    doc_id: str,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[dict[str, str]]:
    """删除文档"""
    success = await service.delete_document(index_name, doc_id)

    if not success:
        return ApiResponse.fail(
            error="document_delete_failed",
            message=f"文档 {doc_id} 删除失败",
        )

    return ApiResponse.ok(
        data={"doc_id": doc_id},
        message="文档删除成功",
    )


# ============== 搜索功能 ==============


@router.post(
    "/search",
    response_model=ApiResponse[ElasticsearchSearchResponse],
    summary="搜索文档",
    description="使用语义搜索、关键词搜索或混合搜索查找文档。",
)
async def search_documents(
    request: Request,
    data: ElasticsearchSearchRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[ElasticsearchSearchResponse]:
    """搜索文档"""
    start_time = time.time()

    options = SearchOptions(
        k=data.k,
        score_threshold=data.score_threshold,
        filter=data.filter,
        enable_highlight=data.enable_highlight,
    )

    results = await service.search(
        index_name=data.index_name,
        query=data.query,
        options=options,
    )

    response_time_ms = int((time.time() - start_time) * 1000)

    # 转换结果
    search_results = [
        ElasticsearchSearchResult(
            id=r.id,
            content=r.content,
            metadata=r.metadata,
            score=r.score,
            highlights=r.metadata.get("highlights") if data.enable_highlight else None,
        )
        for r in results
    ]

    return ApiResponse.ok(
        data=ElasticsearchSearchResponse(
            results=search_results,
            total=len(search_results),
            query=data.query,
            index_name=data.index_name,
            response_time_ms=response_time_ms,
        )
    )


@router.post(
    "/search/hybrid",
    response_model=ApiResponse[ElasticsearchSearchResponse],
    summary="混合搜索",
    description="结合文本搜索和向量搜索的混合搜索。",
)
async def hybrid_search_documents(
    request: Request,
    data: HybridSearchRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[ElasticsearchSearchResponse]:
    """混合搜索"""
    start_time = time.time()

    results = await service.hybrid_search(
        index_name=data.index_name,
        query=data.query,
        k=data.k,
        text_weight=data.text_weight,
        vector_weight=data.vector_weight,
        filter_dict=data.filter,
    )

    response_time_ms = int((time.time() - start_time) * 1000)

    # 转换结果
    search_results = [
        ElasticsearchSearchResult(
            id=r.id,
            content=r.content,
            metadata=r.metadata,
            score=r.score,
        )
        for r in results
    ]

    return ApiResponse.ok(
        data=ElasticsearchSearchResponse(
            results=search_results,
            total=len(search_results),
            query=data.query,
            index_name=data.index_name,
            response_time_ms=response_time_ms,
        )
    )


@router.post(
    "/search/raw",
    response_model=ApiResponse[dict[str, Any]],
    summary="原始查询",
    description="执行原始 Elasticsearch 查询 DSL。",
)
async def raw_search(
    request: Request,
    data: RawSearchRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[dict[str, Any]]:
    """执行原始查询"""
    result = await service.raw_search(
        index_name=data.index_name,
        query=data.query,
    )

    return ApiResponse.ok(
        data=result,
        message="查询执行成功",
    )


# ============== 分析器 ==============


@router.post(
    "/analyze",
    response_model=ApiResponse[AnalyzeResponse],
    summary="分析文本",
    description="使用 Elasticsearch 分析器分析文本。",
)
async def analyze_text(
    request: Request,
    data: AnalyzeRequest,
    service: Annotated[ElasticsearchService, Depends(get_elasticsearch_service)],
) -> ApiResponse[AnalyzeResponse]:
    """分析文本"""
    tokens = await service.analyze(
        index_name=data.index_name,
        text=data.text,
        analyzer=data.analyzer,
    )

    analyzed_tokens = [
        AnalyzeToken(
            token=t.get("token", ""),
            start_offset=t.get("start_offset"),
            end_offset=t.get("end_offset"),
            position=t.get("position"),
            type=t.get("type"),
        )
        for t in tokens
    ]

    return ApiResponse.ok(
        data=AnalyzeResponse(tokens=analyzed_tokens),
    )


__all__ = ["router"]
