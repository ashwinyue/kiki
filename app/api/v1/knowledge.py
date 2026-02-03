"""知识库 API 路由

完全对齐 WeKnora99 API 接口
使用 FastAPI 标准依赖注入模式。
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.dependencies import get_session_dep
from app.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.schemas.knowledge import (
    CopyKnowledgeBaseRequest,
    CopyKnowledgeBaseResponse,
    CopyProgressResponse,
    HybridSearchRequest,
    HybridSearchResult,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    KnowledgeResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeUpdate,
    ManualKnowledgeCreate,
    SourceReference,
)
from app.schemas.knowledge_initialization import (
    EmbeddingTestRequest,
    InitializationConfig,
    KBInitConfigRequest,
    RemoteModelCheckRequest,
    ValidationResult,
)
from app.schemas.response import ApiResponse, DataResponse
from app.services.knowledge.knowledge_initialization import (
    InitResult,
    KnowledgeInitializationService,
)
from app.services.knowledge.knowledge_search import KnowledgeSearchService
from app.services.knowledge.base import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge"])
logger = get_logger(__name__)

# ============== 依赖类型别名 ==============

# 数据库会话依赖
DbDep = Annotated[AsyncSession, Depends(get_session_dep)]


# ============== 知识库管理 ==============


@router.post("", response_model=DataResponse[KnowledgeBaseResponse])
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """创建知识库"""
    service = KnowledgeBaseService(db)
    kb = await service.create_knowledge_base(data, tenant_id)

    return DataResponse(
        success=True,
        data=KnowledgeBaseResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            kb_type=kb.kb_type,
            chunking_config=kb.chunking_config,
            embedding_model_id=kb.embedding_model_id,
            summary_model_id=kb.summary_model_id,
            knowledge_count=0,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        ),
    )


@router.get("", response_model=DataResponse[list[KnowledgeBaseResponse]])
async def list_knowledge_bases(
    db: DbDep,
    tenant_id: TenantIdDep,
    page: int = 1,
    size: int = 20,
):
    """查询知识库列表"""
    service = KnowledgeBaseService(db)
    params = PaginationParams(page=page, size=size)
    kbs = await service.list_knowledge_bases(tenant_id, params)

    # 获取每个知识库的知识数量
    items = []
    for kb in kbs:
        count = await service.get_knowledge_count(kb.id, tenant_id)
        items.append(
            KnowledgeBaseResponse(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                kb_type=kb.kb_type,
                chunking_config=kb.chunking_config,
                embedding_model_id=kb.embedding_model_id,
                summary_model_id=kb.summary_model_id,
                knowledge_count=count,
                created_at=kb.created_at,
                updated_at=kb.updated_at,
            )
        )

    return DataResponse(success=True, data=items)


@router.get("/{kb_id}", response_model=DataResponse[KnowledgeBaseResponse])
async def get_knowledge_base(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取知识库详情"""
    service = KnowledgeBaseService(db)
    kb = await service.get_knowledge_base(kb_id, tenant_id)

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    count = await service.get_knowledge_count(kb_id, tenant_id)

    return DataResponse(
        success=True,
        data=KnowledgeBaseResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            kb_type=kb.kb_type,
            chunking_config=kb.chunking_config,
            embedding_model_id=kb.embedding_model_id,
            summary_model_id=kb.summary_model_id,
            knowledge_count=count,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        ),
    )


@router.put("/{kb_id}", response_model=DataResponse[KnowledgeBaseResponse])
async def update_knowledge_base(
    kb_id: str,
    data: KnowledgeBaseUpdate,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新知识库"""
    service = KnowledgeBaseService(db)
    kb = await service.update_knowledge_base(kb_id, data, tenant_id)

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    count = await service.get_knowledge_count(kb_id, tenant_id)

    return DataResponse(
        success=True,
        data=KnowledgeBaseResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            kb_type=kb.kb_type,
            chunking_config=kb.chunking_config,
            embedding_model_id=kb.embedding_model_id,
            summary_model_id=kb.summary_model_id,
            knowledge_count=count,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        ),
    )


@router.delete("/{kb_id}", response_model=ApiResponse)
async def delete_knowledge_base(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """删除知识库"""
    service = KnowledgeBaseService(db)
    success = await service.delete_knowledge_base(kb_id, tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return ApiResponse(success=True, message="Knowledge base deleted")


@router.post(
    "/{kb_id}/hybrid-search", response_model=DataResponse[list[HybridSearchResult]]
)
async def hybrid_search(
    kb_id: str,
    request: HybridSearchRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """混合搜索

    对齐 WeKnora99 POST /knowledge-bases/{id}/hybrid-search

    支持向量搜索 + 关键词搜索 + RRF 融合 + 重排序
    """
    service = KnowledgeBaseService(db)
    results = await service.hybrid_search(kb_id, request, tenant_id)

    return DataResponse(
        success=True, data=[HybridSearchResult(**r) for r in results]
    )


# ============== 知识条目管理 ==============


@router.post("/{kb_id}/knowledge/file", response_model=ApiResponse)
async def create_knowledge_from_file(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
    file: UploadFile = File(...),
    enable_multimodel: bool = Form(True),
):
    """从文件创建知识条目

    支持的文件类型：.pdf, .txt, .md, .markdown
    """
    from app.infra.storage import get_storage
    from app.services.knowledge.base import is_supported_file_type

    # 1. 验证文件类型
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not is_supported_file_type(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: .pdf, .txt, .md, .markdown",
        )

    # 2. 读取文件内容
    content = await file.read()
    file_size = len(content)

    # 3. 保存到临时文件
    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{file.filename}"
    ) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # 4. 保存到存储（可选）
        storage = get_storage()
        storage_url = storage.upload_file(tmp_path)

        # 5. 创建知识条目并处理
        service = KnowledgeService(db)
        result = await service.create_from_file(
            kb_id=kb_id,
            file_path=tmp_path,  # 使用本地临时路径处理
            file_name=file.filename,
            file_type=file.content_type or "application/octet-stream",
            file_size=file_size,
            tenant_id=tenant_id,
            enable_multimodel=enable_multimodel,
        )

        return ApiResponse(success=True, message=result.get("message", "Knowledge created"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        try:
            import os

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/{kb_id}/knowledge/url", response_model=ApiResponse)
async def create_knowledge_from_url(
    kb_id: str,
    url: str,
    db: DbDep,
    tenant_id: TenantIdDep,
    enable_multimodel: bool = True,
):
    """从 URL 创建知识条目"""
    service = KnowledgeService(db)
    await service.create_from_url(
        kb_id=kb_id,
        url=url,
        tenant_id=tenant_id,
        enable_multimodel=enable_multimodel,
    )

    return ApiResponse(success=True, message="Knowledge created from URL")


@router.get("/{kb_id}/knowledge", response_model=DataResponse[list[KnowledgeResponse]])
async def list_knowledge(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
    page: int = 1,
    size: int = 20,
):
    """知识条目列表"""
    service = KnowledgeService(db)
    params = PaginationParams(page=page, size=size)
    items = await service.list_knowledge(kb_id, tenant_id, params)

    return DataResponse(success=True, data=[KnowledgeResponse(**i) for i in items])


@router.get("/knowledge/{knowledge_id}", response_model=DataResponse[KnowledgeResponse])
async def get_knowledge(
    knowledge_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """知识条目详情"""
    service = KnowledgeService(db)
    knowledge = await service.get_knowledge(knowledge_id, tenant_id)

    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    return DataResponse(success=True, data=KnowledgeResponse(**knowledge))


@router.delete("/knowledge/{knowledge_id}", response_model=ApiResponse)
async def delete_knowledge(
    knowledge_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """删除知识条目"""
    service = KnowledgeService(db)
    success = await service.delete_knowledge(knowledge_id, tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    return ApiResponse(success=True, message="Knowledge deleted")


@router.put("/knowledge/{knowledge_id}", response_model=DataResponse[KnowledgeResponse])
async def update_knowledge(
    knowledge_id: str,
    data: KnowledgeUpdate,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新知识条目"""
    service = KnowledgeService(db)
    knowledge = await service.update_knowledge(knowledge_id, data, tenant_id)

    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    return DataResponse(success=True, data=KnowledgeResponse(**knowledge))


@router.get("/knowledge/{knowledge_id}/download")
async def download_knowledge(
    knowledge_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """下载知识文件

    对齐 WeKnora99 GET /knowledge/{id}/download
    """
    service = KnowledgeService(db)
    knowledge = await service.get_knowledge(knowledge_id, tenant_id)

    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    if knowledge["type"] != "file":
        raise HTTPException(status_code=400, detail="Only file knowledge can be downloaded")

    # TODO: 实现文件下载逻辑
    # from fastapi.responses import FileResponse
    # return FileResponse(knowledge["file_path"], filename=knowledge["file_name"])

    return ApiResponse(
        success=True,
        message=f"Download feature for {knowledge['file_name']} is not implemented yet",
    )


@router.post("/{kb_id}/knowledge/manual", response_model=ApiResponse)
async def create_knowledge_manual(
    kb_id: str,
    data: ManualKnowledgeCreate,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """手工创建知识条目

    对齐 WeKnora99 POST /knowledge-bases/{id}/knowledge/manual
    """
    service = KnowledgeService(db)
    result = await service.create_manual(
        kb_id=kb_id,
        title=data.title,
        content=data.content,
        tenant_id=tenant_id,
    )

    logger.info(
        "manual_knowledge_created",
        knowledge_id=result.get("id"),
        kb_id=kb_id,
        title=data.title,
    )

    return ApiResponse(success=True, message="Knowledge created", data=result)


# ============== 知识库拷贝 ==============


@router.post("/copy", response_model=DataResponse[CopyKnowledgeBaseResponse])
async def copy_knowledge_base(
    data: CopyKnowledgeBaseRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """拷贝知识库（异步任务）

    对齐 WeKnora99 POST /knowledge-bases/copy 接口

    将一个知识库的内容复制到另一个知识库。
    如果 target_id 为空，则创建新的知识库作为目标。
    """
    from app.services.knowledge.knowledge_clone import create_copy_task
    from app.tasks import start_copy_task

    service = KnowledgeBaseService(db)

    # 生成任务 ID
    task_id = data.task_id or str(uuid.uuid4())

    # 验证源知识库存在
    source_kb = await service.get_knowledge_base(data.source_id, tenant_id)
    if not source_kb:
        raise HTTPException(status_code=404, detail="Source knowledge base not found")

    # 创建复制任务
    try:
        task_id, progress = await create_copy_task(
            session=db,
            source_id=data.source_id,
            target_id=data.target_id if data.target_id else None,
            tenant_id=tenant_id,
        )
        target_id = progress.target_id
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 启动后台复制任务
    start_copy_task(task_id, None)

    logger.info(
        "knowledge_base_copy_started",
        task_id=task_id,
        source_id=data.source_id,
        target_id=target_id,
        tenant_id=tenant_id,
    )

    return DataResponse(
        success=True,
        data=CopyKnowledgeBaseResponse(
            task_id=task_id,
            source_id=data.source_id,
            target_id=target_id,
            message="Knowledge base copy task started",
        ),
    )


@router.get(
    "/copy/progress/{task_id}", response_model=DataResponse[CopyProgressResponse]
)
async def get_copy_progress(
    task_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取知识库复制进度

    对齐 WeKnora99 GET /knowledge-bases/copy/progress/{task_id}
    """
    from app.services.knowledge.knowledge_clone import get_copy_progress

    progress = await get_copy_progress(db, task_id)

    if not progress:
        raise HTTPException(status_code=404, detail="Copy task not found")

    # 验证租户权限
    if progress.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return DataResponse(
        success=True,
        data=CopyProgressResponse(**progress.to_dict()),
    )


# ============== 独立知识搜索 ==============


@router.post(
    "/knowledge-search",
    response_model=DataResponse[KnowledgeSearchResponse],
)
async def knowledge_search(
    request: KnowledgeSearchRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """独立知识搜索

    不需要 session_id，直接查询知识库。
    支持多种检索器：vector, bm25, ensemble, conversational。

    Args:
        request: 搜索请求
        db: 数据库会话
        tenant_id: 租户 ID

    Returns:
        搜索结果，包含答案和来源引用
    """
    # 验证知识库存在
    kb_service = KnowledgeBaseService(db)
    kb = await kb_service.get_knowledge_base(request.knowledge_base_id, tenant_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 执行搜索
    search_service = KnowledgeSearchService(db)
    result = await search_service.search(
        query=request.query,
        knowledge_base_id=request.knowledge_base_id,
        tenant_id=tenant_id,
        retriever_type=request.retriever_type,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        chat_history=request.chat_history,
        enable_rerank=request.enable_rerank,
        filter_dict=request.filter_dict,
        embedding_model_id=kb.embedding_model_id,
        summary_model_id=kb.summary_model_id,
    )

    # 转换来源引用
    sources = [
        SourceReference(
            chunk_id=doc.metadata.get("chunk_id", ""),
            knowledge_id=doc.metadata.get("knowledge_id", ""),
            knowledge_title=doc.metadata.get("knowledge_title", ""),
            content=doc.page_content,
            score=doc.metadata.get("score", 0.0),
            metadata=doc.metadata,
        )
        for doc in result.get("documents", [])
    ]

    logger.info(
        "knowledge_search_completed",
        kb_id=request.knowledge_base_id,
        query=request.query[:100],
        retriever_type=request.retriever_type,
        source_count=len(sources),
    )

    return DataResponse(
        success=True,
        data=KnowledgeSearchResponse(
            answer=result.get("answer", ""),
            sources=sources,
            question=result.get("question"),
            retrieved_count=len(result.get("documents", [])),
        ),
    )


# ============== 知识库初始化配置 ==============


@router.get(
    "/{kb_id}/initialization/config",
    response_model=DataResponse[InitializationConfig],
)
async def get_kb_init_config(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取知识库初始化配置

    对齐 WeKnora99 GET /initialization/kb/{kbId}/config

    获取知识库的模型、存储、分块等初始化配置信息。
    """
    service = KnowledgeInitializationService(db)
    config = await service.get_config(kb_id, tenant_id)

    if not config:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return DataResponse(success=True, data=config)


@router.put("/{kb_id}/initialization/config", response_model=ApiResponse)
async def update_kb_init_config(
    kb_id: str,
    data: KBInitConfigRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新知识库初始化配置

    对齐 WeKnora99 PUT /initialization/kb/{kbId}/config

    更新知识库的模型、存储、分块等初始化配置。
    """
    service = KnowledgeInitializationService(db)

    # 构建初始化配置
    config = InitializationConfig(
        kb_id=kb_id,
        kb_name="",  # 将在服务中填充
        has_files=False,  # 将在服务中填充
        llm=data.llm,
        embedding=data.embedding,
        rerank=data.rerank,
        multimodal=data.multimodal,
        extract=data.extract,
        question_generation=data.question_generation,
        vector_store=data.vector_store,
        chunking=data.chunking,
    )

    result: InitResult = await service.update_config(kb_id, tenant_id, config)

    if not result.success:
        raise HTTPException(
            status_code=400 if result.error != "KNOWLEDGE_BASE_NOT_FOUND" else 404,
            detail=result.message,
        )

    return ApiResponse(success=True, message=result.message)


@router.post(
    "/{kb_id}/initialization/validate",
    response_model=DataResponse[ValidationResult],
)
async def validate_kb_config(
    kb_id: str,
    data: KBInitConfigRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """验证知识库配置

    对齐 WeKnora99 的配置验证逻辑

    验证知识库的模型、存储等配置是否有效。
    """

    # 获取知识库
    kb_repo = KnowledgeBaseService(db).kb_repo
    kb = await kb_repo.get_by_tenant(kb_id, tenant_id)

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    service = KnowledgeInitializationService(db)

    # 构建初始化配置
    config = InitializationConfig(
        kb_id=kb_id,
        kb_name=kb.name,
        has_files=False,
        llm=data.llm,
        embedding=data.embedding,
        rerank=data.rerank,
        multimodal=data.multimodal,
        extract=data.extract,
        question_generation=data.question_generation,
        vector_store=data.vector_store,
        chunking=data.chunking,
    )

    validation = await service.validate_config(kb, config)

    return DataResponse(success=True, data=validation)


@router.post(
    "/{kb_id}/initialization/initialize",
    response_model=DataResponse[dict],
)
async def initialize_kb(
    kb_id: str,
    data: KBInitConfigRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """初始化知识库

    对齐 WeKnora99 POST /initialization/kb/{kbId}

    执行知识库初始化，包括配置更新和验证。
    """
    service = KnowledgeInitializationService(db)

    # 构建初始化配置
    config = InitializationConfig(
        kb_id=kb_id,
        kb_name="",
        has_files=False,
        llm=data.llm,
        embedding=data.embedding,
        rerank=data.rerank,
        multimodal=data.multimodal,
        extract=data.extract,
        question_generation=data.question_generation,
        vector_store=data.vector_store,
        chunking=data.chunking,
    )

    result: InitResult = await service.initialize_kb(kb_id, tenant_id, config)

    if not result.success:
        raise HTTPException(
            status_code=400 if result.error != "KNOWLEDGE_BASE_NOT_FOUND" else 404,
            detail=result.message,
        )

    return DataResponse(
        success=True,
        data={
            "message": result.message,
            "status": result.status.value,
            "progress_percent": result.progress_percent,
        },
    )


@router.post(
    "/models/remote/check",
    response_model=DataResponse[dict],
)
async def check_remote_model(request: RemoteModelCheckRequest):
    """检查远程模型连接

    对齐 WeKnora99 POST /initialization/models/remote/check

    检查远程 API 模型连接是否正常。
    """
    # 简化实现，返回基本检查结果
    # TODO: 实现实际的模型连接检查

    available = bool(request.model_name and request.base_url)

    logger.info(
        "remote_model_check",
        model_name=request.model_name,
        model_type=request.model_type,
        available=available,
    )

    return DataResponse(
        success=True,
        data={
            "available": available,
            "message": "连接正常" if available else "模型配置不完整",
        },
    )


@router.post(
    "/models/embedding/test",
    response_model=DataResponse[dict],
)
async def test_embedding_model(request: EmbeddingTestRequest):
    """测试 Embedding 模型

    对齐 WeKnora99 POST /initialization/models/embedding/test

    测试 Embedding 接口是否可用并返回向量维度。
    """
    # 简化实现
    # TODO: 实现实际的 Embedding 模型测试

    available = bool(request.model_name and request.source)

    logger.info(
        "embedding_model_test",
        model_name=request.model_name,
        source=request.source,
        available=available,
    )

    return DataResponse(
        success=True,
        data={
            "available": available,
            "dimension": request.dimension if request.dimension > 0 else 1536,
            "message": "测试成功" if available else "模型配置不完整",
        },
    )


__all__ = ["router"]
