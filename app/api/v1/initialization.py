"""系统初始化 API 路由

对齐 WeKnora99 系统初始化 API 规范
使用 FastAPI 标准依赖注入模式。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.dependencies import get_session_dep
from app.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.schemas.initialization import (
    EmbeddingTestRequest,
    KBModelConfigRequest,
    LLMTestRequest,
    ModelTestResponse,
    MultimodalTestRequest,
    OllamaStatusResponse,
    RemoteModelCheckRequest,
    RerankTestRequest,
    TestResultResponse,
)
from app.schemas.response import ApiResponse, DataResponse
from app.services.initialization_service import InitializationService
from app.services.model_test import ModelTestService
from app.services.ollama import get_ollama_service

router = APIRouter(prefix="/initialization", tags=["initialization"])
logger = get_logger(__name__)

# ============== 依赖类型别名 ==============

# 数据库会话依赖
DbDep = Annotated[AsyncSession, Depends(get_session_dep)]

router = APIRouter(prefix="/initialization", tags=["initialization"])
logger = get_logger(__name__)


@router.get("/kb/{kb_id}/config", response_model=DataResponse[dict])
async def get_kb_config(
    kb_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取知识库配置

    对齐 WeKnora99 GET /initialization/kb/{kbId}/config
    """
    service = InitializationService(db)
    config = await service.get_kb_config(kb_id, tenant_id)

    if not config:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return DataResponse(success=True, data=config)


@router.put("/kb/{kb_id}/config", response_model=ApiResponse)
async def update_kb_config(
    kb_id: str,
    data: KBModelConfigRequest,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新知识库配置

    对齐 WeKnora99 PUT /initialization/kb/{kbId}/config
    """
    service = InitializationService(db)

    try:
        config = await service.update_kb_config(kb_id, data, tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    if not config:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    logger.info(
        "kb_config_updated",
        kb_id=kb_id,
        tenant_id=tenant_id,
    )

    return ApiResponse(success=True, message="配置更新成功", data=config)


@router.post("/kb/{kb_id}", response_model=ApiResponse)
async def initialize_kb(
    kb_id: str,
    data: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """初始化知识库

    对齐 WeKnora99 POST /initialization/kb/{kbId}
    创建或更新知识库的模型和分块配置
    """
    service = InitializationService(db)

    kb = await service.kb_repo.get_by_tenant(kb_id, tenant_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    logger.info(
        "kb_initialization",
        kb_id=kb_id,
        tenant_id=tenant_id,
    )

    # TODO: 实现完整的初始化逻辑（创建模型、更新配置等）
    return ApiResponse(success=True, message="知识库初始化成功")


@router.get("/ollama/status", response_model=DataResponse[OllamaStatusResponse])
async def check_ollama_status(
    db: DbDep,
):
    """检查 Ollama 服务状态

    对齐 WeKnora99 GET /initialization/ollama/status
    """
    service = InitializationService(db)
    status = await service.check_ollama_status()

    return DataResponse(success=True, data=status)


@router.post("/models/embedding/test", response_model=DataResponse[ModelTestResponse])
async def test_embedding_model(
    data: EmbeddingTestRequest,
    db: DbDep,
):
    """测试 Embedding 模型

    对齐 WeKnora99 POST /initialization/models/embedding/test

    测试 Embedding 接口是否可用并返回向量维度
    """
    service = InitializationService(db)
    result = await service.test_embedding_model(data)

    return DataResponse(success=True, data=result)


@router.post("/models/rerank/check", response_model=DataResponse[ModelTestResponse])
async def check_rerank_model(
    data: RerankTestRequest,
    db: DbDep,
):
    """检查 Rerank 模型

    对齐 WeKnora99 POST /initialization/models/rerank/check

    检查 Rerank 模型连接和功能是否正常
    """
    service = InitializationService(db)
    result = await service.check_rerank_model(data)

    return DataResponse(success=True, data=result)


@router.post("/models/remote/check", response_model=DataResponse[ModelTestResponse])
async def check_remote_model(
    data: RemoteModelCheckRequest,
    db: DbDep,
):
    """检查远程模型连接

    对齐 WeKnora99 POST /initialization/models/remote/check

    检查远程 API 模型连接是否正常
    """
    service = InitializationService(db)
    result = await service.check_remote_model(data)

    return DataResponse(success=True, data=result)


# ============== Ollama 模型管理接口 ==============


@router.get("/ollama/models", response_model=DataResponse[dict])
async def list_ollama_models(
    db: DbDep,
):
    """列出已安装的 Ollama 模型

    对齐 WeKnora99 GET /initialization/ollama/models
    """
    ollama = await get_ollama_service()
    models = await ollama.list_models()

    logger.info("ollama_models_listed", count=len(models))

    return DataResponse(success=True, data={"models": models})


@router.post("/ollama/models/check", response_model=DataResponse[dict])
async def check_ollama_models(
    data: dict,
    db: DbDep,
):
    """检查指定的 Ollama 模型是否已安装

    对齐 WeKnora99 POST /initialization/ollama/models/check

    请求体: {"models": ["model1", "model2", ...]}
    """
    ollama = await get_ollama_service()
    models = data.get("models", [])

    if not models:
        raise HTTPException(status_code=400, detail="Models list is required")

    model_status = {}
    for model_name in models:
        model_status[model_name] = await ollama.is_model_available(model_name)

    logger.info("ollama_models_checked", models=model_status)

    return DataResponse(success=True, data={"models": model_status})


@router.post("/ollama/models/download", response_model=DataResponse[dict])
async def download_ollama_model(
    data: dict,
    db: DbDep,
):
    """异步下载指定的 Ollama 模型

    对齐 WeKnora99 POST /initialization/ollama/models/download

    请求体: {"modelName": "model_name"}
    """
    ollama = await get_ollama_service()
    model_name = data.get("modelName")

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    result = await ollama.create_download_task(model_name)

    logger.info(
        "ollama_download_started",
        model_name=model_name,
        task_id=result.get("id"),
    )

    return DataResponse(success=True, data=result)


@router.get("/ollama/download/progress/{task_id}", response_model=DataResponse[dict])
async def get_download_progress(
    task_id: str,
    db: DbDep,
):
    """获取 Ollama 模型下载任务的进度

    对齐 WeKnora99 GET /initialization/ollama/download/progress/{taskId}
    """
    ollama = await get_ollama_service()
    result = await ollama.get_download_progress(task_id)

    if not result:
        raise HTTPException(status_code=404, detail="Download task not found")

    return DataResponse(success=True, data=result)


@router.get("/ollama/download/tasks", response_model=DataResponse[list])
async def list_download_tasks(
    db: DbDep,
):
    """列出所有 Ollama 模型下载任务

    对齐 WeKnora99 GET /initialization/ollama/download/tasks
    """
    ollama = await get_ollama_service()
    result = await ollama.list_download_tasks()

    return DataResponse(success=True, data=result)


@router.delete("/ollama/models/{model_name:path}", response_model=ApiResponse)
async def delete_ollama_model(
    model_name: str,
    db: DbDep,
):
    """删除指定的 Ollama 模型

    对齐 WeKnora99 DELETE /initialization/ollama/models/{model}

    Args:
        model_name: 模型名称（可包含命名空间，如 llama3:latest）
    """
    ollama = await get_ollama_service()
    success = await ollama.delete_model(model_name)

    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {model_name}",
        )

    logger.info("ollama_model_deleted", model_name=model_name)

    return ApiResponse(success=True, message=f"模型 {model_name} 删除成功")


@router.get("/ollama/models/{model_name:path}/info", response_model=DataResponse[dict])
async def get_ollama_model_info(
    model_name: str,
    db: DbDep,
):
    """获取 Ollama 模型详细信息

    对齐 WeKnora99 获取模型信息功能

    Args:
        model_name: 模型名称（可包含命名空间，如 llama3:latest）
    """
    ollama = await get_ollama_service()
    info = await ollama.get_model_info(model_name)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name}",
        )

    return DataResponse(success=True, data=info)


# ============== 模型测试接口（新增）=============


@router.post("/models/llm/test", response_model=DataResponse[TestResultResponse])
async def test_llm_model(
    data: LLMTestRequest,
    db: DbDep,
):
    """测试 LLM 模型

    对齐 WeKnora99 POST /initialization/models/llm/test

    测试 LLM 模型连接和基本功能
    """
    test_service = ModelTestService(db)

    result = await test_service.test_llm(
        model_name=data.model_name,
        base_url=data.base_url,
        api_key=data.api_key,
        provider=data.provider,
    )

    return DataResponse(
        success=True,
        data=TestResultResponse(
            status=result.status,
            message=result.message,
            latency_ms=result.latency_ms,
            details=result.details,
        ),
    )


@router.post("/models/multimodal/test", response_model=DataResponse[TestResultResponse])
async def test_multimodal_model(
    data: MultimodalTestRequest,
    db: DbDep,
):
    """测试多模态模型

    对齐 WeKnora99 POST /initialization/multimodal/test

    测试多模态模型（VLM）的文本和图片理解能力
    """
    test_service = ModelTestService(db)

    result = await test_service.test_multimodal(
        model_name=data.model_name,
        base_url=data.base_url,
        api_key=data.api_key,
        image_base64=data.image_base64,
    )

    return DataResponse(
        success=True,
        data=TestResultResponse(
            status=result.status,
            message=result.message,
            latency_ms=result.latency_ms,
            details=result.details,
        ),
    )


@router.post("/models/by-id/{model_id}/test", response_model=DataResponse[TestResultResponse])
async def test_model_by_id(
    model_id: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """根据模型 ID 测试模型

    对齐 WeKnora99 模型测试功能

    Args:
        model_id: 模型 ID
    """
    test_service = ModelTestService(db)

    result = await test_service.test_model_by_id(model_id, tenant_id)

    return DataResponse(
        success=True,
        data=TestResultResponse(
            status=result.status,
            message=result.message,
            latency_ms=result.latency_ms,
            details=result.details,
        ),
    )


__all__ = ["router"]
