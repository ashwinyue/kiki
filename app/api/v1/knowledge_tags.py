"""知识标签 API 路由

对齐 WeKnora99 API 接口规范
使用 FastAPI 标准依赖注入模式。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.dependencies import get_session_dep
from app.middleware import TenantIdDep
from app.models.knowledge import KnowledgeTag
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.repositories.tag import TagRepository
from app.schemas.knowledge import (
    TagCreate,
    TagListResponse,
    TagResponse,
    TagUpdate,
)
from app.schemas.response import ApiResponse, DataResponse

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-tags"])
logger = get_logger(__name__)

# ============== 依赖类型别名 ==============

# 数据库会话依赖
DbDep = Annotated[AsyncSession, Depends(get_session_dep)]


@router.get("/{kb_id}/tags", response_model=DataResponse[TagListResponse])
async def list_tags(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页条数"),
    keyword: str | None = Query(None, description="标签名称关键字搜索"),
):
    """获取知识库标签列表

    对齐 WeKnora99 GET /knowledge-bases/:id/tags
    """
    repo = TagRepository(db)
    params = PaginationParams(page=page, size=size)

    result = await repo.list_by_kb(kb_id, tenant_id, params, keyword)

    # 获取每个标签的统计信息
    tags_with_counts = []
    for tag in result.items:
        knowledge_count = await repo.get_knowledge_count(tag.id, tenant_id)
        chunk_count = await repo.get_chunk_count(tag.id, tenant_id)

        tags_with_counts.append(
            TagResponse(
                id=tag.id,
                knowledge_base_id=tag.knowledge_base_id,
                name=tag.name,
                color=tag.color or "#1890ff",
                sort_order=tag.sort_order,
                knowledge_count=knowledge_count,
                chunk_count=chunk_count,
                created_at=tag.created_at,
                updated_at=tag.updated_at,
            )
        )

    return DataResponse(
        success=True,
        data=TagListResponse(
            total=result.total,
            page=result.page,
            page_size=result.size,
            data=tags_with_counts,
        ),
    )


@router.post("/{kb_id}/tags", response_model=DataResponse[TagResponse])
async def create_tag(
    kb_id: str,
    data: TagCreate,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """创建标签

    对齐 WeKnora99 POST /knowledge-bases/:id/tags
    """
    repo = TagRepository(db)

    # 检查同名标签
    from sqlalchemy import select

    stmt = select(KnowledgeTag).where(
        KnowledgeTag.knowledge_base_id == kb_id,
        KnowledgeTag.tenant_id == tenant_id,
        KnowledgeTag.name == data.name,
        KnowledgeTag.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=400, detail="Tag with this name already exists")

    tag_data = data.model_dump()
    tag_data["knowledge_base_id"] = kb_id

    tag = await repo.create_with_tenant(tag_data, tenant_id)

    logger.info(
        "tag_created",
        tag_id=tag.id,
        kb_id=kb_id,
        name=tag.name,
    )

    return DataResponse(
        success=True,
        data=TagResponse(
            id=tag.id,
            knowledge_base_id=tag.knowledge_base_id,
            name=tag.name,
            color=tag.color or "#1890ff",
            sort_order=tag.sort_order,
            knowledge_count=0,
            chunk_count=0,
            created_at=tag.created_at,
            updated_at=tag.updated_at,
        ),
    )


@router.put("/{kb_id}/tags/{tag_id}", response_model=DataResponse[TagResponse])
async def update_tag(
    kb_id: str,
    tag_id: str,
    data: TagUpdate,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新标签

    对齐 WeKnora99 PUT /knowledge-bases/:id/tags/:tag_id
    """
    repo = TagRepository(db)

    # 验证标签存在
    tag = await repo.get_by_tenant(tag_id, tenant_id)
    if not tag or tag.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="Tag not found")

    # 更新字段
    update_data: dict = {}
    if data.name is not None:
        update_data["name"] = data.name
    if data.color is not None:
        update_data["color"] = data.color
    if data.sort_order is not None:
        update_data["sort_order"] = data.sort_order

    if update_data:
        tag = await repo.update(tag_id, **update_data)

        logger.info(
            "tag_updated",
            tag_id=tag_id,
            update_data=update_data,
        )

    knowledge_count = await repo.get_knowledge_count(tag_id, tenant_id)
    chunk_count = await repo.get_chunk_count(tag_id, tenant_id)

    return DataResponse(
        success=True,
        data=TagResponse(
            id=tag.id,
            knowledge_base_id=tag.knowledge_base_id,
            name=tag.name,
            color=tag.color or "#1890ff",
            sort_order=tag.sort_order,
            knowledge_count=knowledge_count,
            chunk_count=chunk_count,
            created_at=tag.created_at,
            updated_at=tag.updated_at,
        ),
    )


@router.delete("/{kb_id}/tags/{tag_id}", response_model=ApiResponse)
async def delete_tag(
    kb_id: str,
    tag_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
    force: bool = Query(False, description="强制删除（即使标签被引用）"),
):
    """删除标签

    对齐 WeKnora99 DELETE /knowledge-bases/:id/tags/:tag_id
    """
    repo = TagRepository(db)

    # 验证标签存在
    tag = await repo.get_by_tenant(tag_id, tenant_id)
    if not tag or tag.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="Tag not found")

    # 检查是否有关联
    if not force:
        knowledge_count = await repo.get_knowledge_count(tag_id, tenant_id)
        chunk_count = await repo.get_chunk_count(tag_id, tenant_id)

        if knowledge_count > 0 or chunk_count > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Tag is in use (knowledge_count={knowledge_count}, chunk_count={chunk_count}). Use force=true to delete anyway.",
            )

    success = await repo.soft_delete(tag_id, tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Tag not found")

    logger.info(
        "tag_deleted",
        tag_id=tag_id,
        kb_id=kb_id,
        force=force,
    )

    return ApiResponse(success=True, message="Tag deleted")


__all__ = ["router"]
