"""知识库 FAQ 管理 API 路由

对齐 WeKnora99 知识库 FAQ API 规范
路径: /knowledge-bases/{id}/faq/*
使用 FastAPI 标准依赖注入模式。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.dependencies import get_session_dep
from app.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.schemas.response import ApiResponse, DataResponse

router = APIRouter(tags=["knowledge-faq"])
logger = get_logger(__name__)

# ============== 依赖类型别名 ==============

# 数据库会话依赖
DbDep = Annotated[AsyncSession, Depends(get_session_dep)]


@router.get("/{id}/faq/entries", response_model=DataResponse[dict])
async def list_faq_entries(
    id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量", alias="page_size"),
    tag_id: int | None = Query(None, description="标签ID筛选 (seq_id)"),
    keyword: str | None = Query(None, description="关键词搜索"),
    search_field: str | None = Query(None, description="搜索字段"),
    sort_order: str | None = Query(None, description="排序方式"),
):
    """获取知识库 FAQ 条目列表

    对齐 WeKnora99 GET /knowledge-bases/{id}/faq/entries
    """
    # TODO: 实现实际的 FAQ 列表查询
    result = {
        "items": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
    }

    logger.info(
        "faq_entries_listed",
        kb_id=id,
        tenant_id=tenant_id,
        tag_id=tag_id,
        keyword=keyword,
    )

    return DataResponse(success=True, data=result)


@router.get("/{id}/faq/entries/export")
async def export_faq_entries(
    id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """导出 FAQ 条目为 CSV

    对齐 WeKnora99 GET /knowledge-bases/{id}/faq/entries/export
    """
    # TODO: 实现 CSV 导出功能
    logger.info("faq_entries_exported", kb_id=id, tenant_id=tenant_id)

    from fastapi import Response

    csv_content = "id,question,answer\n"  # 预留实现
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="faq_export.csv"',
        },
    )


@router.get("/{id}/faq/entries/{entry_id}", response_model=DataResponse[dict])
async def get_faq_entry(
    id: str,
    entry_id: int,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取 FAQ 条目详情

    对齐 WeKnora99 GET /knowledge-bases/{id}/faq/entries/{entry_id}
    """
    # TODO: 实现实际的 FAQ 条目详情查询
    logger.info("faq_entry_retrieved", kb_id=id, entry_id=entry_id, tenant_id=tenant_id)

    raise HTTPException(status_code=501, detail="FAQ 详情查询功能（预留实现）")


@router.post("/{id}/faq/entries", response_model=DataResponse[dict])
async def upsert_faq_entries(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """批量创建/更新 FAQ 条目

    对齐 WeKnora99 POST /knowledge-bases/{id}/faq/entries

    支持 dry_run 模式进行验证
    """
    # TODO: 实现 FAQ 批量导入逻辑
    task_id = "task-placeholder"

    logger.info(
        "faq_entries_upserted",
        kb_id=id,
        tenant_id=tenant_id,
        dry_run=data.get("dry_run", False),
    )

    return DataResponse(success=True, data={"task_id": task_id})


@router.post("/{id}/faq/entry", response_model=DataResponse[dict])
async def create_faq_entry(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """创建单个 FAQ 条目

    对齐 WeKnora99 POST /knowledge-bases/{id}/faq/entry
    """
    # TODO: 实现单个 FAQ 条目创建
    logger.info(
        "faq_entry_created",
        kb_id=id,
        tenant_id=tenant_id,
    )

    return DataResponse(success=True, data={"message": "FAQ 条目创建成功（预留实现）"})


@router.put("/{id}/faq/entries/{entry_id}", response_model=DataResponse[dict])
async def update_faq_entry(
    id: str,
    entry_id: int,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新 FAQ 条目

    对齐 WeKnora99 PUT /knowledge-bases/{id}/faq/entries/{entry_id}
    """
    # TODO: 实现 FAQ 条目更新
    logger.info(
        "faq_entry_updated",
        kb_id=id,
        entry_id=entry_id,
        tenant_id=tenant_id,
    )

    return DataResponse(success=True, data={"message": "FAQ 条目更新成功（预留实现）"})


@router.post("/{id}/faq/entries/{entry_id}/similar-questions", response_model=ApiResponse)
async def add_similar_questions(
    id: str,
    entry_id: int,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """添加相似问题到 FAQ 条目

    对齐 WeKnora99 POST /knowledge-bases/{id}/faq/entries/{entry_id}/similar-questions
    """
    similar_questions = data.get("similar_questions", [])

    # TODO: 实现相似问题添加
    logger.info(
        "similar_questions_added",
        kb_id=id,
        entry_id=entry_id,
        count=len(similar_questions),
        tenant_id=tenant_id,
    )

    return ApiResponse(success=True, message="相似问题添加成功（预留实现）")


@router.put("/{id}/faq/entries/fields", response_model=ApiResponse)
async def update_entry_fields_batch(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """批量更新 FAQ 字段

    对齐 WeKnora99 PUT /knowledge-bases/{id}/faq/entries/fields

    支持: is_enabled, is_recommended, tag_id
    """
    # TODO: 实现批量字段更新
    logger.info(
        "faq_entry_fields_updated",
        kb_id=id,
        tenant_id=tenant_id,
        updates=list(data.keys()),
    )

    return ApiResponse(success=True, message="FAQ 字段批量更新成功（预留实现）")


@router.put("/{id}/faq/entries/tags", response_model=ApiResponse)
async def update_entry_tags_batch(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """批量更新 FAQ 标签

    对齐 WeKnora99 PUT /knowledge-bases/{id}/faq/entries/tags

    请求体: {"updates": {entry_seq_id: tag_seq_id, ...}}
    """
    # TODO: 实现批量标签更新
    logger.info(
        "faq_entry_tags_updated",
        kb_id=id,
        tenant_id=tenant_id,
        updates_count=len(data.get("updates", {})),
    )

    return ApiResponse(success=True, message="FAQ 标签批量更新成功（预留实现）")


@router.delete("/{id}/faq/entries", response_model=ApiResponse)
async def delete_faq_entries(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """批量删除 FAQ 条目

    对齐 WeKnora99 DELETE /knowledge-bases/{id}/faq/entries

    请求体: {"ids": [seq_id1, seq_id2, ...]}
    """
    ids = data.get("ids", [])

    # TODO: 实现批量删除
    logger.info(
        "faq_entries_deleted",
        kb_id=id,
        tenant_id=tenant_id,
        count=len(ids),
    )

    return ApiResponse(success=True, message=f"删除了 {len(ids)} 条 FAQ")


@router.post("/{id}/faq/search", response_model=DataResponse[dict])
async def search_faq(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """搜索 FAQ

    对齐 WeKnora99 POST /knowledge-bases/{id}/faq/search

    支持混合搜索和两级优先级标签召回
    """
    # TODO: 实现 FAQ 搜索
    logger.info(
        "faq_searched",
        kb_id=id,
        tenant_id=tenant_id,
        query=data.get("query_text", ""),
    )

    return DataResponse(success=True, data={"entries": [], "total": 0})


@router.get("/faq/import/progress/{task_id}", response_model=DataResponse[dict])
async def get_import_progress(
    task_id: str,
    db: DbDep,
):
    """获取 FAQ 导入进度

    对齐 WeKnora99 GET /faq/import/progress/{task_id}
    """
    # TODO: 实现导入进度查询
    logger.info("faq_import_progress_queried", task_id=task_id)

    return DataResponse(
        success=True,
        data={
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "导入进度查询（预留实现）",
        },
    )


@router.put("/{id}/faq/import/last-result/display", response_model=ApiResponse)
async def update_last_import_result_display_status(
    id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新 FAQ 最后一次导入结果显示状态

    对齐 WeKnora99 PUT /knowledge-bases/{id}/faq/import/last-result/display
    """
    display_status = data.get("display_status", "open")

    # TODO: 实现显示状态更新
    logger.info(
        "faq_import_display_status_updated",
        kb_id=id,
        tenant_id=tenant_id,
        display_status=display_status,
    )

    return ApiResponse(success=True, message="显示状态更新成功（预留实现）")


__all__ = ["router"]
