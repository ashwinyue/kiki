"""模型相关 Schema

对齐 WeKnora99 API 接口规范
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """模型服务商信息

    对齐 WeKnora99 GET /models/providers 响应格式
    """

    value: str = Field(..., description="服务商标识（如：aliyun, zhipu）")
    label: str = Field(..., description="服务商显示名称（如：阿里云 DashScope）")
    description: str = Field(..., description="服务商描述")
    default_urls: dict[str, str] | None = Field(
        default=None, description="默认 API 地址（按类型区分）"
    )
    model_types: list[str] = Field(
        default_factory=list, description="支持的模型类型列表"
    )


class ModelProvidersResponse(BaseModel):
    """模型服务商列表响应"""

    providers: list[ProviderInfo]


class ModelType(str, Enum):
    """模型类型"""

    EMBEDDING = "Embedding"
    RERANK = "Rerank"
    KNOWLEDGE_QA = "KnowledgeQA"
    CHAT = "Chat"
    VLLM = "VLLM"


class ModelSource(str, Enum):
    """模型来源"""

    LOCAL = "local"
    OPENAI = "openai"
    ALIYUN = "aliyun"
    ZHIPU = "zhipu"
    REMOTE = "remote"


class EmbeddingParameters(BaseModel):
    """嵌入模型专用参数"""

    dimension: int = Field(default=0, description="向量维度（如：768, 1024）")
    truncate_prompt_tokens: int = Field(default=0, description="截断 Token 数（0 表示不截断）")


class ModelParameters(BaseModel):
    """模型参数

    对齐 WeKnora99 ModelParameters 结构
    """

    base_url: str = Field(default="", description="API 服务地址")
    api_key: str = Field(default="", description="API 密钥")
    provider: str = Field(default="", description="服务商标识（可选，用于选择特定的 API 适配器）")
    embedding_parameters: EmbeddingParameters | None = Field(
        default=None, description="Embedding 模型专用参数"
    )
    extra_config: dict[str, object] | None = Field(
        default=None, description="服务商特定的额外配置"
    )


class ModelCreate(BaseModel):
    """创建模型请求"""

    name: str = Field(..., min_length=1, max_length=255, description="模型名称")
    type: ModelType = Field(..., description="模型类型")
    source: ModelSource = Field(default=ModelSource.LOCAL, description="模型来源")
    description: str = Field(..., min_length=1, max_length=500, description="模型描述")
    parameters: ModelParameters = Field(
        default_factory=ModelParameters, description="模型参数"
    )
    is_default: bool = Field(default=False, description="是否默认")


class ModelUpdate(BaseModel):
    """更新模型请求"""

    name: str | None = Field(None, min_length=1, max_length=255, description="模型名称")
    description: str | None = Field(None, max_length=500, description="模型描述")
    parameters: ModelParameters | None = Field(None, description="模型参数")
    is_default: bool | None = Field(None, description="是否默认")


class ModelResponse(BaseModel):
    """模型响应"""

    id: str
    name: str
    type: str
    source: str
    description: str
    parameters: dict
    is_default: bool
    is_builtin: bool
    status: str
    created_at: datetime


__all__ = [
    "ModelType",
    "ModelSource",
    "ModelParameters",
    "EmbeddingParameters",
    "ModelCreate",
    "ModelUpdate",
    "ModelResponse",
    "ProviderInfo",
    "ModelProvidersResponse",
]
