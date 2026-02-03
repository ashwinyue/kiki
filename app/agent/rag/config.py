"""RAG 配置管理

支持多种 RAG 后端的配置加载和检索器创建。
参考 DeerFlow 的 YAML 配置 + 环境变量合并模式。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.rag.retrievers.faiss import FAISSRetriever
from app.agent.rag.retrievers.ragflow import RAGFlowRetriever, RAGFlowConfig
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 支持的 RAG 后端类型
RAGBackend = Literal["faiss", "ragflow", "qdrant", "dify"]


@dataclass
class KnowledgeBaseConfig:
    """知识库配置

    Attributes:
        name: 知识库名称（用于标识和隔离）
        backend: RAG 后端类型
        backend_config: 后端特定配置
        description: 知识库描述
    """
    name: str
    backend: RAGBackend = "faiss"
    backend_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        """标准化知识库名称"""
        self.name = self.name.strip().lower()


@dataclass
class RAGConfig:
    """RAG 配置

    Attributes:
        default_backend: 默认后端类型
        retrieval_size: 默认检索结果数量
        backends: 各后端的具体配置
    """
    default_backend: RAGBackend = "faiss"
    retrieval_size: int = 5
    backends: dict[str, dict[str, object]] = field(default_factory=dict)

    def get_backend_config(self, backend: RAGBackend) -> dict[str, object]:
        """获取指定后端的配置"""
        return self.backends.get(backend, {})


def load_rag_config_from_env() -> RAGConfig:
    """从环境变量加载 RAG 配置

    支持的环境变量：
    - RAG_DEFAULT_BACKEND: 默认后端（faiss, ragflow, qdrant, dify）
    - RAG_RETRIEVAL_SIZE: 默认检索结果数量
    - RAGFLOW_API_URL: RAGFlow API 地址
    - RAGFLOW_API_KEY: RAGFlow API 密钥
    - RAGFLOW_DATASET_ID: RAGFlow 数据集 ID
    - QDRANT_URL: Qdrant 服务地址
    - QDRANT_API_KEY: Qdrant API 密钥
    - DIFY_API_URL: Dify API 地址
    - DIFY_API_KEY: Dify API 密钥

    Returns:
        RAG 配置对象
    """
    default_backend: RAGBackend = os.getenv(
        "RAG_DEFAULT_BACKEND",
        "faiss",
    )  # type: ignore

    retrieval_size = int(os.getenv("RAG_RETRIEVAL_SIZE", "5"))

    backends: dict[str, dict[str, object]] = {}

    # RAGFlow 配置
    if os.getenv("RAGFLOW_API_URL"):
        backends["ragflow"] = {
            "api_url": os.getenv("RAGFLOW_API_URL"),
            "api_key": os.getenv("RAGFLOW_API_KEY", ""),
            "dataset_id": os.getenv("RAGFLOW_DATASET_ID"),
        }

    # Qdrant 配置（占位）
    if os.getenv("QDRANT_URL"):
        backends["qdrant"] = {
            "url": os.getenv("QDRANT_URL"),
            "api_key": os.getenv("QDRANT_API_KEY"),
            "collection": os.getenv("QDRANT_COLLECTION", "kiki_knowledge"),
        }

    # Dify 配置（占位）
    if os.getenv("DIFY_API_URL"):
        backends["dify"] = {
            "api_url": os.getenv("DIFY_API_URL"),
            "api_key": os.getenv("DIFY_API_KEY"),
        }

    return RAGConfig(
        default_backend=default_backend,
        retrieval_size=retrieval_size,
        backends=backends,
    )


def load_rag_config_from_yaml(config_dict: dict[str, object] | None) -> RAGConfig:
    """从 YAML 配置字典加载 RAG 配置

    Args:
        config_dict: YAML 配置解析后的字典

    Returns:
        RAG 配置对象
    """
    if config_dict is None:
        return load_rag_config_from_env()

    rag_dict = config_dict.get("RAG", {})

    default_backend: RAGBackend = rag_dict.get(
        "default_backend",
        "faiss",
    )  # type: ignore

    retrieval_size = rag_dict.get("retrieval_size", 5)

    backends: dict[str, dict[str, object]] = {}

    # 解析各后端配置
    for backend_name in ["faiss", "ragflow", "qdrant", "dify"]:
        backend_config = rag_dict.get(backend_name)
        if backend_config:
            backends[backend_name] = backend_config

    return RAGConfig(
        default_backend=default_backend,
        retrieval_size=retrieval_size,
        backends=backends,
    )


def create_retriever(
    backend: RAGBackend | None = None,
    config: RAGConfig | None = None,
) -> "FAISSRetriever | RAGFlowRetriever":
    """创建检索器实例

    Args:
        backend: 后端类型（不指定则使用配置中的默认值）
        config: RAG 配置（不指定则从环境变量加载）

    Returns:
        检索器实例

    Raises:
        ValueError: 不支持的后端类型
    """
    if config is None:
        config = load_rag_config_from_env()

    if backend is None:
        backend = config.default_backend

    backend_config = config.get_backend_config(backend)

    match backend:
        case "faiss":
            # FAISS 不需要额外配置
            return FAISSRetriever()

        case "ragflow":
            if not backend_config.get("api_url"):
                raise ValueError("RAGFlow 需要 api_url 配置")

            ragflow_config = RAGFlowConfig(
                api_url=str(backend_config["api_url"]),
                api_key=str(backend_config.get("api_key", "")),
                dataset_id=str(backend_config["dataset_id"]) if backend_config.get("dataset_id") else None,
            )
            return RAGFlowRetriever(ragflow_config)

        case "qdrant":
            # Qdrant 待实现
            raise NotImplementedError("Qdrant 后端尚未实现")

        case "dify":
            # Dify 待实现
            raise NotImplementedError("Dify 后端尚未实现")

        case _:
            raise ValueError(f"不支持的后端类型: {backend}")


__all__ = [
    "RAGBackend",
    "RAGConfig",
    "KnowledgeBaseConfig",
    "load_rag_config_from_env",
    "load_rag_config_from_yaml",
    "create_retriever",
]
