"""Embedding 模型集成"""

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
    """DashScope Qwen Embedding V4"""

    def __init__(
        self,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    ) -> None:
        """初始化 DashScope Embeddings"""
        valid_dimensions = {64, 128, 256, 512, 768, 1024, 1536, 2048}
        if dimensions not in valid_dimensions:
            logger.warning(
                "invalid_dimensions",
                dimensions=dimensions,
                valid=list(valid_dimensions),
            )
            dimensions = 1024

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
    """获取 Embeddings 实例"""
    if provider is None:
        provider = getattr(settings, "embedding_provider", "openai")

    if provider == "dashscope":
        return DashScopeEmbeddings(
            model=model or "text-embedding-v4",
            dimensions=dimensions or getattr(settings, "embedding_dimensions", 1024),
        )

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


__all__ = [
    "EmbeddingProvider",
    "DashScopeEmbeddings",
    "get_embeddings",
]
