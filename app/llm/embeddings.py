"""Embedding 模型集成

支持多种 Embedding 提供商：
- OpenAI: text-embedding-3-small/large
- DashScope: text-embedding-v4 (Qwen)
- VoyageAI: voyage-3-large
- 本地模型: Ollama embeddings
"""

from enum import Enum
from typing import Literal

from langchain_openai import OpenAIEmbeddings

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingProvider(str, Enum):
    """Embedding 提供商"""

    OPENAI = "openai"
    DASHSCOPE = "dashscope"
    VOYAGE = "voyage"
    OLLAMA = "ollama"


class DashScopeEmbeddings(OpenAIEmbeddings):
    """DashScope Qwen Embedding V4

    使用 OpenAI 兼容模式调用 DashScope API。
    文档: https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

    模型参数:
    - text-embedding-v4: 1024 维（默认），支持 256-2048 维
    - 单行最大 Token: 8,192
    - 最大批处理行数: 10
    - 支持语种: 中文、英语等 100+ 语种
    """

    def __init__(
        self,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    ) -> None:
        """初始化 DashScope Embeddings

        Args:
            model: 模型名称，默认 text-embedding-v4
            dimensions: 向量维度，可选 64/128/256/512/768/1024/1536/2048
            api_key: DashScope API Key
            base_url: API 基础 URL
            **kwargs: 其他参数
        """
        # 验证维度参数
        valid_dimensions = {64, 128, 256, 512, 768, 1024, 1536, 2048}
        if dimensions not in valid_dimensions:
            logger.warning(
                "invalid_dimensions",
                dimensions=dimensions,
                valid=list(valid_dimensions),
            )
            dimensions = 1024

        # 使用环境变量中的 API Key
        if api_key is None:
            api_key = settings.dashscope_api_key

        if not api_key:
            raise ValueError(
                "DashScope API Key is required. "
                "Set DASHSCOPE_API_KEY environment variable or pass api_key parameter."
            )

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
            **kwargs,
        )

        logger.info(
            "dashscope_embeddings_initialized",
            model=model,
            dimensions=dimensions,
        )


def get_embeddings(
    provider: Literal["openai", "dashscope", "voyage", "ollama"] | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> OpenAIEmbeddings:
    """获取 Embeddings 实例

    Args:
        provider: 提供商名称，默认从配置读取
        model: 模型名称
        dimensions: 向量维度（仅部分提供商支持）

    Returns:
        Embeddings 实例
    """
    # 从配置获取默认提供商
    if provider is None:
        provider = getattr(settings, "embedding_provider", "openai")

    if provider == "dashscope":
        return DashScopeEmbeddings(
            model=model or "text-embedding-v4",
            dimensions=dimensions or getattr(settings, "embedding_dimensions", 1024),
        )

    # OpenAI 兼容的提供商
    api_key = settings.llm_api_key
    base_url = settings.llm_base_url

    if provider == "openai":
        return OpenAIEmbeddings(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


# 导出
__all__ = [
    "EmbeddingProvider",
    "DashScopeEmbeddings",
    "get_embeddings",
]
