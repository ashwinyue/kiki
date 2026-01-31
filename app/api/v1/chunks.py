"""文档分块 API 路由

对齐 WeKnora99 分块管理 API
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.repositories.chunk import ChunkRepository
from app.schemas.chunk import (
    ChunkListResponse,
    ChunkResponse,
    ChunkTypes,
    DeleteQuestionRequest,
)
from app.schemas.response import ApiResponse, DataResponse

router = APIRouter(prefix="/chunks", tags=["chunks"])
logger = get_logger(__name__)


@router.get("/{knowledge_id}", response_model=DataResponse[ChunkListResponse])
async def list_knowledge_chunks(
    knowledge_id: str,
    db: Annotated[AsyncSession, Depends(get_session)],
    *,
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页条数"),
    tenant_id: TenantIdDep = None,
):
    """获取知识的分块列表

    对齐 WeKnora99 GET /chunks/{knowledge_id}
    """
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    repo = ChunkRepository(db)

    result = await repo.list_by_knowledge(
        knowledge_id=knowledge_id,
        tenant_id=tenant_id,
        params=PaginationParams(page=page, size=size),
        chunk_types=[ChunkTypes.TEXT],
    )

    response_data = ChunkListResponse(
        items=result.items,
        total=result.total,
        page=result.page,
        page_size=result.size,
    )

    logger.info(
        "chunks_listed",
        knowledge_id=knowledge_id,
        tenant_id=tenant_id,
        count=len(result.items),
    )

    return DataResponse(success=True, data=response_data)


@router.get("/by-id/{id}", response_model=DataResponse[ChunkResponse])
async def get_chunk_by_id(
    id: str,
    db: Annotated[AsyncSession, Depends(get_session)],
    *,
    tenant_id: TenantIdDep = None,
):
    """通过 ID 获取分块

    对齐 WeKnora99 GET /chunks/by-id/{id}
    """
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    repo = ChunkRepository(db)
    chunk = await repo.get_by_tenant(id, tenant_id)

    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    logger.info("chunk_retrieved", chunk_id=id, tenant_id=tenant_id)

    return DataResponse(success=True, data=chunk)


@router.put("/{knowledge_id}/{id}", response_model=DataResponse[ChunkResponse])
async def update_chunk(
    knowledge_id: str,
    id: str,
    data: dict,
    db: Annotated[AsyncSession, Depends(get_session)],
    *,
    tenant_id: TenantIdDep = None,
):
    """更新分块

    对齐 WeKnora99 PUT /chunks/{knowledge_id}/{id}
    """
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    repo = ChunkRepository(db)

    # 验证分块存在且属于该知识
    chunk = await repo.get_by_tenant(id, tenant_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    if chunk.knowledge_id != knowledge_id:
        raise HTTPException(status_code=403, detail="Chunk does not belong to this knowledge")

    # 构建更新字段
    update_fields = {}
    if "content" in data:
        update_fields["content"] = data["content"]
    if "is_enabled" in data:
        update_fields["is_enabled"] = data["is_enabled"]
    if "chunk_index" in data:
        update_fields["chunk_index"] = data["chunk_index"]
    if "start_at" in data:
        update_fields["start_at"] = data["start_at"]
    if "end_at" in data:
        update_fields["end_at"] = data["end_at"]
    if "image_info" in data:
        update_fields["image_info"] = data["image_info"]
    if "metadata" in data:
        update_fields["metadata"] = data["metadata"]

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = await repo.update_fields(id, tenant_id, **update_fields)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update chunk")

    logger.info(
        "chunk_updated",
        chunk_id=id,
        knowledge_id=knowledge_id,
        tenant_id=tenant_id,
        fields=list(update_fields.keys()),
    )

    return DataResponse(success=True, data=updated)


@router.delete("/{knowledge_id}/{id}", response_model=ApiResponse)
async def delete_chunk(
    knowledge_id: str,
    id: str,
    db: Annotated[AsyncSession, Depends(get_session)],
    *,
    tenant_id: TenantIdDep = None,
):
    """删除分块

    对齐 WeKnora99 DELETE /chunks/{knowledge_id}/{id}
    """
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    repo = ChunkRepository(db)

    # 验证分块存在且属于该知识
    chunk = await repo.get_by_tenant(id, tenant_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    if chunk.knowledge_id != knowledge_id:
        raise HTTPException(status_code=403, detail="Chunk does not belong to this knowledge")

    success = await repo.soft_delete(id, tenant_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete chunk")

    logger.info("chunk_deleted", chunk_id=id, knowledge_id=knowledge_id, tenant_id=tenant_id)

    return ApiResponse(success=True, message="Chunk deleted")


@router.delete("/{knowledge_id}", response_model=ApiResponse)
async def delete_chunks_by_knowledge(
    knowledge_id: str,
    db: Annotated[AsyncSession, Depends(get_session)],
    *,
    tenant_id: TenantIdDep = None,
):
    """删除知识下所有分块

    对齐 WeKnora99 DELETE /chunks/{knowledge_id}
    """
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    repo = ChunkRepository(db)

    count = await repo.delete_by_knowledge(knowledge_id, tenant_id)

    logger.info(
        "chunks_deleted_by_knowledge",
        knowledge_id=knowledge_id,
        tenant_id=tenant_id,
        count=count,
    )

    return ApiResponse(success=True, message=f"Deleted {count} chunks")


@router.delete("/by-id/{id}/questions", response_model=ApiResponse)
async def delete_generated_question(
    id: str,
    data: DeleteQuestionRequest,
    db: Annotated[AsyncSession, Depends(get_session)],
    *,
    tenant_id: TenantIdDep = None,
):
    """删除生成的问题

    对齐 WeKnora99 DELETE /chunks/by-id/{id}/questions
    """
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    repo = ChunkRepository(db)

    # 验证分块存在
    chunk = await repo.get_by_tenant(id, tenant_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    success = await repo.delete_generated_question(id, data.question_id, tenant_id)

    if not success:
        raise HTTPException(status_code=400, detail="Question not found or already deleted")

    logger.info(
        "generated_question_deleted",
        chunk_id=id,
        question_id=data.question_id,
        tenant_id=tenant_id,
    )

    return ApiResponse(success=True, message="Generated question deleted")


__all__ = ["router"]
