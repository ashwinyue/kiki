"""模型服务

提供模型的业务逻辑
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.llm_model import Model
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.repositories.model import ModelRepository
from app.schemas.model import ModelCreate, ModelUpdate, ProviderInfo

logger = get_logger(__name__)


# WeKnora99 支持的服务商配置
# 参考: WeKnora99/docs/api/model.md
_MODEL_PROVIDERS: list[dict[str, Any]] = [
    {
        "value": "generic",
        "label": "自定义 (OpenAI兼容接口)",
        "description": "通用 OpenAI 兼容接口",
        "default_urls": {
            "chat": "https://api.openai.com/v1",
            "embedding": "https://api.openai.com/v1",
            "rerank": "https://api.openai.com/v1",
        },
        "model_types": ["chat", "embedding", "rerank", "vllm"],
    },
    {
        "value": "openai",
        "label": "OpenAI",
        "description": "gpt-4o, gpt-4o-mini, text-embedding-3, etc.",
        "default_urls": {
            "chat": "https://api.openai.com/v1",
            "embedding": "https://api.openai.com/v1",
            "rerank": "https://api.openai.com/v1",
        },
        "model_types": ["chat", "embedding", "rerank", "vllm"],
    },
    {
        "value": "aliyun",
        "label": "阿里云 DashScope",
        "description": "qwen-plus, tongyi-embedding-vision-plus, qwen3-rerank, etc.",
        "default_urls": {
            "chat": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "embedding": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "rerank": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        },
        "model_types": ["chat", "embedding", "rerank", "vllm"],
    },
    {
        "value": "zhipu",
        "label": "智谱 BigModel",
        "description": "glm-4.7, embedding-3, rerank, etc.",
        "default_urls": {
            "chat": "https://open.bigmodel.cn/api/paas/v4",
            "embedding": "https://open.bigmodel.cn/api/paas/v4/embeddings",
            "rerank": "https://open.bigmodel.cn/api/paas/v4/rerank",
        },
        "model_types": ["chat", "embedding", "rerank", "vllm"],
    },
    {
        "value": "deepseek",
        "label": "DeepSeek",
        "description": "deepseek-chat, deepseek-coder",
        "default_urls": {
            "chat": "https://api.deepseek.com/v1",
        },
        "model_types": ["chat"],
    },
    {
        "value": "jina",
        "label": "Jina AI",
        "description": "jina-embeddings-v3, jina-reranker-v2",
        "default_urls": {
            "embedding": "https://api.jina.ai/v1",
            "rerank": "https://api.jina.ai/v1",
        },
        "model_types": ["embedding", "rerank"],
    },
    {
        "value": "gemini",
        "label": "Google Gemini",
        "description": "gemini-2.0-flash, gemini-exp",
        "default_urls": {
            "chat": "https://generativelanguage.googleapis.com/v1beta",
        },
        "model_types": ["chat"],
    },
    {
        "value": "volcengine",
        "label": "火山引擎 Volcengine",
        "description": "doubao-pro, doubao-embedding",
        "default_urls": {
            "chat": "https://ark.cn-beijing.volces.com/api/v3",
            "embedding": "https://ark.cn-beijing.volces.com/api/v3",
        },
        "model_types": ["chat", "embedding", "vllm"],
    },
    {
        "value": "hunyuan",
        "label": "腾讯混元 Hunyuan",
        "description": "hunyuan-pro, hunyuan-embedding",
        "default_urls": {
            "chat": "https://hunyuan.tencentcloudapi.com/v1",
            "embedding": "https://hunyuan.tencentcloudapi.com/v1",
        },
        "model_types": ["chat", "embedding"],
    },
    {
        "value": "siliconflow",
        "label": "硅基流动 SiliconFlow",
        "description": "Qwen, Llama, DeepSeek 等多种模型",
        "default_urls": {
            "chat": "https://api.siliconflow.cn/v1",
            "embedding": "https://api.siliconflow.cn/v1",
            "rerank": "https://api.siliconflow.cn/v1",
        },
        "model_types": ["chat", "embedding", "rerank", "vllm"],
    },
    {
        "value": "moonshot",
        "label": "月之暗面 Moonshot",
        "description": "moonshot-v1-8k, moonshot-v1-32k",
        "default_urls": {
            "chat": "https://api.moonshot.cn/v1",
        },
        "model_types": ["chat", "vllm"],
    },
]


def get_providers(model_type: str | None = None) -> list[ProviderInfo]:
    """获取支持的模型服务商列表

    对齐 WeKnora99 GET /models/providers 接口

    Args:
        model_type: 可选，按模型类型过滤（chat/embedding/rerank/vllm）

    Returns:
        服务商信息列表
    """
    providers = []

    for p in _MODEL_PROVIDERS:
        # 按模型类型过滤
        if model_type:
            model_type_key = model_type.lower()
            if model_type_key not in p["model_types"]:
                continue

        providers.append(
            ProviderInfo(
                value=p["value"],
                label=p["label"],
                description=p["description"],
                default_urls=p.get("default_urls"),
                model_types=p["model_types"],
            )
        )

    return providers


class ModelService:
    """模型服务"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._model_repo: ModelRepository | None = None

    @property
    def model_repo(self) -> ModelRepository:
        """延迟初始化模型仓储"""
        if self._model_repo is None:
            self._model_repo = ModelRepository(self.session)
        return self._model_repo

    async def create_model(
        self, data: ModelCreate, tenant_id: int
    ) -> Model:
        """创建模型

        Args:
            data: 创建请求
            tenant_id: 租户 ID

        Returns:
            创建的模型
        """
        create_data = {
            "name": data.name,
            "type": data.type.value,
            "source": data.source.value,
            "description": data.description,
            "is_default": data.is_default,
            "is_builtin": False,
            "status": "active",
            "parameters": data.parameters.model_dump(),
        }

        # 如果设置为默认，取消同类型的其他默认
        if data.is_default:
            existing = await self.model_repo.get_default(data.type.value, tenant_id)
            if existing:
                existing.is_default = False

        model = await self.model_repo.create_with_tenant(create_data, tenant_id)

        logger.info(
            "model_created",
            model_id=model.id,
            tenant_id=tenant_id,
            name=model.name,
            type=model.type,
        )
        return model

    async def get_model(
        self, model_id: str, tenant_id: int
    ) -> Model | None:
        """获取模型详情

        Args:
            model_id: 模型 ID
            tenant_id: 租户 ID

        Returns:
            模型实例
        """
        return await self.model_repo.get_by_tenant(model_id, tenant_id)

    async def list_models(
        self,
        tenant_id: int,
        model_type: str | None = None,
        params: PaginationParams | None = None,
    ) -> list[Model]:
        """模型列表

        Args:
            tenant_id: 租户 ID
            model_type: 模型类型
            params: 分页参数

        Returns:
            模型列表
        """
        params = params or PaginationParams()
        result = await self.model_repo.list_by_type(model_type, tenant_id, params)
        return result.items

    async def update_model(
        self, model_id: str, data: ModelUpdate, tenant_id: int
    ) -> Model | None:
        """更新模型

        Args:
            model_id: 模型 ID
            data: 更新请求
            tenant_id: 租户 ID

        Returns:
            更新后的模型
        """
        update_data: dict = {}
        if data.name is not None:
            update_data["name"] = data.name
        if data.description is not None:
            update_data["description"] = data.description
        if data.parameters is not None:
            update_data["parameters"] = data.parameters.model_dump()
        if data.is_default is not None and data.is_default:
            # 取消同类型的其他默认
            model = await self.get_model(model_id, tenant_id)
            if model:
                existing = await self.model_repo.get_default(model.type, tenant_id)
                if existing and existing.id != model_id:
                    existing.is_default = False
            update_data["is_default"] = True

        if not update_data:
            return await self.get_model(model_id, tenant_id)

        return await self.model_repo.update(model_id, **update_data)

    async def delete_model(
        self, model_id: str, tenant_id: int
    ) -> bool:
        """删除模型

        Args:
            model_id: 模型 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        return await self.model_repo.soft_delete(model_id, tenant_id)

    async def set_default(
        self, model_id: str, tenant_id: int
    ) -> bool:
        """设置默认模型

        Args:
            model_id: 模型 ID
            tenant_id: 租户 ID

        Returns:
            是否设置成功
        """
        return await self.model_repo.set_default(model_id, tenant_id)

    async def get_default_model(
        self, model_type: str, tenant_id: int
    ) -> Model | None:
        """获取默认模型

        Args:
            model_type: 模型类型
            tenant_id: 租户 ID

        Returns:
            默认模型
        """
        return await self.model_repo.get_default(model_type, tenant_id)


__all__ = ["ModelService", "get_providers"]
