"""模型 API 路由

完全对齐 WeKnora99 API 接口
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_tenant_id
from app.repositories.base import PaginationParams
from app.schemas.model import (
    ModelCreate,
    ModelProvidersResponse,
    ModelResponse,
    ModelUpdate,
)
from app.schemas.response import ApiResponse, DataResponse
from app.services.model_service import ModelService, get_providers

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=DataResponse[ModelResponse])
async def create_model(
    data: ModelCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: int = Depends(get_tenant_id),
):
    """创建模型"""
    service = ModelService(db)
    model = await service.create_model(data, tenant_id)

    return DataResponse(
        success=True,
        data=ModelResponse(
            id=model.id,
            name=model.name,
            type=model.type,
            source=model.source,
            description=model.description,
            parameters=model.parameters,
            is_default=model.is_default,
            is_builtin=model.is_builtin,
            status=model.status,
            created_at=model.created_at,
        ),
    )


@router.get("", response_model=DataResponse[list[ModelResponse]])
async def list_models(
    type: str | None = None,
    page: int = 1,
    size: int = 20,
    db: AsyncSession = Depends(get_db),
    tenant_id: int = Depends(get_tenant_id),
):
    """模型列表"""
    service = ModelService(db)
    params = PaginationParams(page=page, size=size)
    models = await service.list_models(tenant_id, type, params)

    return DataResponse(
        success=True,
        data=[
            ModelResponse(
                id=m.id,
                name=m.name,
                type=m.type,
                source=m.source,
                description=m.description,
                parameters=m.parameters,
                is_default=m.is_default,
                is_builtin=m.is_builtin,
                status=m.status,
                created_at=m.created_at,
            )
            for m in models
        ],
    )


@router.get("/{model_id}", response_model=DataResponse[ModelResponse])
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: int = Depends(get_tenant_id),
):
    """模型详情"""
    service = ModelService(db)
    model = await service.get_model(model_id, tenant_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return DataResponse(
        success=True,
        data=ModelResponse(
            id=model.id,
            name=model.name,
            type=model.type,
            source=model.source,
            description=model.description,
            parameters=model.parameters,
            is_default=model.is_default,
            is_builtin=model.is_builtin,
            status=model.status,
            created_at=model.created_at,
        ),
    )


@router.patch("/{model_id}", response_model=DataResponse[ModelResponse])
async def update_model(
    model_id: str,
    data: ModelUpdate,
    db: AsyncSession = Depends(get_db),
    tenant_id: int = Depends(get_tenant_id),
):
    """更新模型"""
    service = ModelService(db)
    model = await service.update_model(model_id, data, tenant_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return DataResponse(
        success=True,
        data=ModelResponse(
            id=model.id,
            name=model.name,
            type=model.type,
            source=model.source,
            description=model.description,
            parameters=model.parameters,
            is_default=model.is_default,
            is_builtin=model.is_builtin,
            status=model.status,
            created_at=model.created_at,
        ),
    )


@router.delete("/{model_id}", response_model=ApiResponse)
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: int = Depends(get_tenant_id),
):
    """删除模型"""
    service = ModelService(db)
    success = await service.delete_model(model_id, tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return ApiResponse(success=True, message="Model deleted")


@router.get("/providers", response_model=DataResponse[list[dict[str, Any]]])
async def get_model_providers(
    model_type: str | None = Query(
        None, description="模型类型：chat, embedding, rerank, vllm"
    ),
):
    """获取模型服务商列表

    对齐 WeKnora99 GET /models/providers 接口

    返回支持的 AI 模型服务商及配置信息。
    """
    providers = get_providers(model_type)

    return DataResponse(
        success=True,
        data=[
            {
                "value": p.value,
                "label": p.label,
                "description": p.description,
                "defaultUrls": p.default_urls,
                "modelTypes": p.model_types,
            }
            for p in providers
        ],
    )


__all__ = ["router"]
