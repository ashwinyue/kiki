"""向量 API

提供向量索引、搜索等功能的 RESTful API。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status

from app.api.v1.auth import get_current_user
from app.models.database import User
from app.observability.logging import get_logger
from app.schemas.response import ApiResponse
from app.schemas.vector import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionsListResponse,
    HybridSearchRequest,
    SearchResultItem,
    VectorIndexBatchRequest,
    VectorIndexRequest,
    VectorIndexResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStatsResponse,
)
from app.services.vector_service import VectorService
from app.vector_stores import VectorStoreConfig, create_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/vectors", tags=["Vector Store"])


# ============== 依赖注入 ==============


async def get_vector_service(
    current_user: Annotated[User, Depends(get_current_user)],
) -> VectorService:
    """获取向量服务实例

    自动注入租户 ID 以实现多租户隔离。
    """
    # 获取租户 ID
    tenant_id = current_user.tenant_id

    # 创建向量服务
    return VectorService(tenant_id=tenant_id)


# ============== 索引接口 ==============


@router.post(
    "/index",
    response_model=ApiResponse[VectorIndexResponse],
    summary="索引文档",
    description="将文档添加到向量存储，支持语义搜索。",
)
async def index_documents(
    request: Request,
    data: VectorIndexRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[VectorIndexResponse]:
    """索引文档

    将文档内容转换为向量并存储到向量数据库。
    """
    service = await get_vector_service(current_user)

    # 构建文档列表
    documents = [
        (content, data.metadatas[i] if data.metadatas else None)
        for i, content in enumerate(data.documents)
    ]

    # 索引
    ids = await service.index_documents(
        documents=documents,
        ids=data.ids,
        collection_name=data.collection_name,
    )

    return ApiResponse.ok(
        data=VectorIndexResponse(
            ids=ids,
            count=len(ids),
            collection_name=data.collection_name,
        ),
        message=f"成功索引 {len(ids)} 个文档",
    )


@router.post(
    "/index/batch",
    response_model=ApiResponse[VectorIndexResponse],
    summary="批量索引文档",
    description="批量索引大量文档，自动分块处理。",
)
async def index_documents_batch(
    request: Request,
    data: VectorIndexBatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[VectorIndexResponse]:
    """批量索引文档

    支持大规模文档索引，自动分块。
    """
    service = await get_vector_service(current_user)

    # TODO: 实现分块逻辑
    documents = [
        (content, {"index": i})
        for i, content in enumerate(data.documents)
    ]

    ids = await service.index_documents(
        documents=documents,
        collection_name=data.collection_name,
    )

    return ApiResponse.ok(
        data=VectorIndexResponse(
            ids=ids,
            count=len(ids),
            collection_name=data.collection_name,
        ),
        message=f"成功索引 {len(ids)} 个文档",
    )


# ============== 搜索接口 ==============


@router.post(
    "/search",
    response_model=ApiResponse[VectorSearchResponse],
    summary="向量搜索",
    description="基于语义相似度搜索文档。",
)
async def vector_search(
    request: Request,
    data: VectorSearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[VectorSearchResponse]:
    """向量搜索

    使用语义相似度搜索相关文档。
    """
    service = await get_vector_service(current_user)

    results, response_time_ms = await service.search(
        query=data.query,
        k=data.k,
        score_threshold=data.score_threshold,
        filter_dict=data.filter_dict,
        collection_name=data.collection_name,
    )

    # 转换结果
    search_results = [
        SearchResultItem(
            content=r.content,
            metadata=r.metadata,
            score=r.score,
            id=r.id,
        )
        for r in results
    ]

    return ApiResponse.ok(
        data=VectorSearchResponse(
            results=search_results,
            total=len(search_results),
            query=data.query,
            collection_name=data.collection_name,
            response_time_ms=response_time_ms,
        )
    )


@router.post(
    "/search/hybrid",
    response_model=ApiResponse[VectorSearchResponse],
    summary="混合搜索",
    description="结合向量搜索和关键词搜索的混合搜索。",
)
async def hybrid_search(
    request: Request,
    data: HybridSearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[VectorSearchResponse]:
    """混合搜索

    结合语义搜索和关键词匹配，提高搜索准确度。
    """
    service = await get_vector_service(current_user)

    results, response_time_ms = await service.hybrid_search(
        query=data.query,
        k=data.k,
        keyword_weight=data.keyword_weight,
        vector_weight=data.vector_weight,
        filter_dict=data.filter_dict,
        collection_name=data.collection_name,
    )

    # 转换结果
    search_results = [
        SearchResultItem(
            content=r.content,
            metadata=r.metadata,
            score=r.score,
            id=r.id,
        )
        for r in results
    ]

    return ApiResponse.ok(
        data=VectorSearchResponse(
            results=search_results,
            total=len(search_results),
            query=data.query,
            collection_name=data.collection_name,
            response_time_ms=response_time_ms,
        )
    )


# ============== 删除接口 ==============


@router.delete(
    "/{collection_name}",
    response_model=ApiResponse[dict[str, str]],
    summary="删除集合",
    description="删除指定集合的所有向量。",
)
async def delete_collection(
    request: Request,
    collection_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[dict[str, str]]:
    """删除集合

    删除整个集合及其所有向量数据。
    """
    service = await get_vector_service(current_user)

    success = await service.delete_collection(collection_name)

    if not success:
        return ApiResponse.fail(
            error="删除失败",
            message=f"无法删除集合 {collection_name}",
        )

    return ApiResponse.ok(
        data={"status": "deleted"},
        message=f"集合 {collection_name} 已删除",
    )


# ============== 统计接口 ==============


@router.get(
    "/stats",
    response_model=ApiResponse[VectorStatsResponse],
    summary="获取统计信息",
    description="获取向量存储的统计信息。",
)
async def get_stats(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    collection_name: str = Query(default="default", description="集合名称"),
) -> ApiResponse[VectorStatsResponse]:
    """获取统计信息

    返回向量存储的统计信息。
    """
    service = await get_vector_service(current_user)

    stats = await service.get_stats(collection_name)

    return ApiResponse.ok(
        data=VectorStatsResponse(**stats),
    )


@router.get(
    "/health",
    response_model=ApiResponse[dict[str, str]],
    summary="健康检查",
    description="检查向量存储服务是否健康。",
)
async def health_check(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[dict[str, str]]:
    """健康检查

    检查向量存储服务状态。
    """
    service = await get_vector_service(current_user)

    is_healthy = await service.health_check()

    if is_healthy:
        return ApiResponse.ok(
            data={"status": "healthy"},
            message="向量存储服务正常",
        )
    else:
        return ApiResponse.fail(
            error="unhealthy",
            message="向量存储服务不可用",
        )


# ============== 集合管理接口 ==============


@router.post(
    "/collections",
    response_model=ApiResponse[CollectionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建集合",
    description="创建新的向量集合。",
)
async def create_collection(
    request: Request,
    data: CollectionCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[CollectionResponse]:
    """创建集合

    创建新的向量集合。
    """
    # 获取租户 ID
    tenant_id = current_user.tenant_id or 0

    # 创建配置
    config = VectorStoreConfig(
        collection_name=data.name,
        dimension=data.dimension,
        metric=data.metric,
        tenant_id=tenant_id,
    )

    # 创建向量存储（会自动创建集合）
    store = create_vector_store("memory", config)  # 根据实际配置选择类型
    await store.initialize()

    return ApiResponse.ok(
        data=CollectionResponse(
            name=data.name,
            dimension=data.dimension,
            metric=data.metric,
            vector_count=0,
        ),
        message=f"集合 {data.name} 创建成功",
    )


@router.get(
    "/collections",
    response_model=ApiResponse[CollectionsListResponse],
    summary="列出集合",
    description="获取所有向量集合列表。",
)
async def list_collections(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ApiResponse[CollectionsListResponse]:
    """列出集合

    返回所有可用的向量集合。
    """
    # TODO: 根据实际向量存储实现获取集合列表
    return ApiResponse.ok(
        data=CollectionsListResponse(
            collections=[],
            total=0,
        )
    )


__all__ = ["router"]
