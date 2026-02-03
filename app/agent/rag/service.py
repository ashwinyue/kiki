"""RAG 服务管理器

提供租户级别/Agent 级别的知识库隔离和管理。
参考 DeerFlow 的服务层设计理念，RAG 作为独立服务而非全局工具。

核心特性：
- 多租户隔离
- Agent 级别知识库分离
- 检索器生命周期管理
- 配置驱动的后端选择
"""

from asyncio import Lock
from dataclasses import dataclass
from typing import Any

from app.agent.rag.config import KnowledgeBaseConfig, RAGBackend, RAGConfig, load_rag_config_from_env
from app.agent.rag.retrievers.base import BaseRetriever, RetrievalError, RetrievalOptions
from app.agent.rag.retrievers.faiss import FAISSRetriever
from app.agent.rag.retrievers.ragflow import RAGFlowConfig, RAGFlowRetriever
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalContext:
    """检索上下文

    用于单次检索的上下文信息。

    Attributes:
        tenant_id: 租户 ID
        agent_name: Agent 名称
        knowledge_base: 知识库名称
    """
    tenant_id: int | None = None
    agent_name: str = "default"
    knowledge_base: str = "default"


class RAGService:
    """RAG 服务管理器

    管理多个租户和 Agent 的知识库，提供隔离的检索服务。

    设计原则：
    1. 服务层独立于工具层
    2. 支持多租户/多 Agent 隔离
    3. 配置驱动的后端选择
    4. 检索器生命周期管理

    Example:
        ```python
        # 创建服务实例
        service = RAGService()

        # 获取或创建租户的知识库检索器
        retriever = service.get_retriever(
            tenant_id=123,
            knowledge_base="research_docs",
        )

        # 执行检索
        results = await retriever.retrieve("查询内容")

        # 为 Agent 创建专属工具
        tool = service.create_tool_for_agent(
            agent_name="researcher",
            tenant_id=123,
            knowledge_base="research_docs",
        )
        ```
    """

    def __init__(self, config: RAGConfig | None = None):
        """初始化 RAG 服务

        Args:
            config: RAG 全局配置（可选，默认从环境变量加载）
        """
        self._config = config or load_rag_config_from_env()
        self._retrievers: dict[str, BaseRetriever] = {}
        self._kb_configs: dict[str, KnowledgeBaseConfig] = {}
        self._lock = Lock()

        logger.info(
            "rag_service_initialized",
            default_backend=self._config.default_backend,
        )

    def _make_key(
        self,
        tenant_id: int | None,
        knowledge_base: str,
    ) -> str:
        """生成检索器缓存键

        Args:
            tenant_id: 租户 ID
            knowledge_base: 知识库名称

        Returns:
            缓存键（格式: tenant_id:kb_name 或 global:kb_name）
        """
        tenant_part = str(tenant_id) if tenant_id is not None else "global"
        return f"{tenant_part}:{knowledge_base.lower()}"

    def register_knowledge_base(
        self,
        config: KnowledgeBaseConfig,
        tenant_id: int | None = None,
    ) -> None:
        """注册知识库配置

        Args:
            config: 知识库配置
            tenant_id: 租户 ID（None 表示全局知识库）
        """
        key = self._make_key(tenant_id, config.name)
        self._kb_configs[key] = config

        logger.info(
            "knowledge_base_registered",
            knowledge_base=config.name,
            backend=config.backend,
            tenant_id=tenant_id,
        )

    def get_retriever(
        self,
        tenant_id: int | None = None,
        knowledge_base: str = "default",
        context: RetrievalContext | None = None,
    ) -> BaseRetriever:
        """获取或创建检索器

        Args:
            tenant_id: 租户 ID
            knowledge_base: 知识库名称
            context: 检索上下文（可选，优先使用上下文中的参数）

        Returns:
            检索器实例

        Raises:
            RetrievalError: 检索器创建失败
        """
        # 使用上下文参数
        if context:
            tenant_id = context.tenant_id
            knowledge_base = context.knowledge_base

        key = self._make_key(tenant_id, knowledge_base)

        # 检查缓存
        if key in self._retrievers:
            logger.debug(
                "retriever_cache_hit",
                key=key,
            )
            return self._retrievers[key]

        # 获取知识库配置
        kb_config = self._kb_configs.get(key)
        backend = kb_config.backend if kb_config else self._config.default_backend
        backend_config = kb_config.backend_config if kb_config else {}

        # 创建检索器
        retriever = self._create_retriever(
            backend=backend,
            backend_config=backend_config,
        )

        # 缓存检索器
        self._retrievers[key] = retriever

        logger.info(
            "retriever_created",
            key=key,
            backend=backend,
        )

        return retriever

    def _create_retriever(
        self,
        backend: RAGBackend,
        backend_config: dict[str, Any],
    ) -> BaseRetriever:
        """创建检索器实例

        Args:
            backend: 后端类型
            backend_config: 后端配置

        Returns:
            检索器实例

        Raises:
            RetrievalError: 不支持的后端或创建失败
        """
        try:
            match backend:
                case "faiss":
                    return FAISSRetriever()

                case "ragflow":
                    ragflow_config = RAGFlowConfig(
                        api_url=str(backend_config.get("api_url", "")),
                        api_key=str(backend_config.get("api_key", "")),
                        dataset_id=str(backend_config.get("dataset_id")) if backend_config.get("dataset_id") else None,
                    )
                    return RAGFlowRetriever(ragflow_config)

                case _:
                    raise RetrievalError(
                        message=f"不支持的后端类型: {backend}",
                        retriever_type=backend,
                    )

        except Exception as e:
            raise RetrievalError(
                message=f"创建检索器失败: {e}",
                retriever_type=backend,
                cause=e,
            )

    async def retrieve(
        self,
        query: str,
        tenant_id: int | None = None,
        knowledge_base: str = "default",
        options: RetrievalOptions | None = None,
        context: RetrievalContext | None = None,
    ) -> list:
        """执行检索

        Args:
            query: 查询文本
            tenant_id: 租户 ID
            knowledge_base: 知识库名称
            options: 检索选项
            context: 检索上下文（可选）

        Returns:
            检索结果列表
        """
        retriever = self.get_retriever(
            tenant_id=tenant_id,
            knowledge_base=knowledge_base,
            context=context,
        )

        return await retriever.retrieve(query, options)

    def create_tool_for_agent(
        self,
        agent_name: str,
        tenant_id: int | None = None,
        knowledge_base: str = "default",
        tool_name: str | None = None,
    ):
        """为 Agent 创建专属的 RAG 工具

        这是一个工厂函数，动态创建与 Agent 绑定的 RAG 工具。

        Args:
            agent_name: Agent 名称
            tenant_id: 租户 ID
            knowledge_base: 知识库名称
            tool_name: 工具名称（默认: search_{agent_name}_knowledge）

        Returns:
            LangChain 工具实例

        Example:
            ```python
            service = RAGService()

            # 为研究员 Agent 创建工具
            researcher_tool = service.create_tool_for_agent(
                agent_name="researcher",
                tenant_id=123,
                knowledge_base="research_docs",
            )

            # 创建 Agent
            agent = create_react_agent(
                agent_name="researcher",
                tools=[search_web, researcher_tool],
            )
            ```
        """
        from langchain_core.tools import tool

        if tool_name is None:
            tool_name = f"search_{agent_name}_knowledge"

        # 创建检索上下文
        context = RetrievalContext(
            tenant_id=tenant_id,
            agent_name=agent_name,
            knowledge_base=knowledge_base,
        )

        @tool
        async def rag_tool(query: str, top_k: int = 5) -> str:
            """搜索知识库以获取相关文档

            在向量数据库中检索与查询相关的文档。
            此工具专属于 {agent_name} Agent，使用 {knowledge_base} 知识库。

            Args:
                query: 搜索查询或问题
                top_k: 返回结果数量（1-20）

            Returns:
                检索到的相关文档内容
            """
            top_k = max(1, min(20, int(top_k)))

            try:
                results = await self.retrieve(
                    query=query,
                    context=context,
                    options=RetrievalOptions(top_k=top_k),
                )

                if not results:
                    return "未找到相关文档。"

                # 格式化结果
                formatted_parts = []
                for i, doc in enumerate(results, 1):
                    formatted_parts.append(
                        f"## {i}. {doc.title}\n"
                        f"**来源**: {doc.source}\n"
                        f"**相似度**: {doc.score:.4f}\n\n"
                        f"{doc.content}\n"
                    )

                return "\n".join(formatted_parts)

            except Exception as e:
                logger.error(
                    "rag_tool_failed",
                    agent_name=agent_name,
                    query=query[:50],
                    error=str(e),
                )
                return f"检索知识库时出错: {str(e)}"

        # 设置工具名称
        rag_tool.name = tool_name
        rag_tool.description = f"""搜索知识库以获取相关文档。

此工具专属于 {agent_name} Agent，使用 {knowledge_base} 知识库。
在向量数据库中检索与查询相关的文档。

Args:
    query: 搜索查询或问题
    top_k: 返回结果数量（1-20）
"""

        logger.info(
            "rag_tool_created_for_agent",
            agent_name=agent_name,
            tool_name=tool_name,
            knowledge_base=knowledge_base,
            tenant_id=tenant_id,
        )

        return rag_tool

    def clear_tenant_cache(self, tenant_id: int | None) -> None:
        """清除租户的检索器缓存

        Args:
            tenant_id: 租户 ID
        """
        prefix = f"{tenant_id if tenant_id is not None else 'global'}:"

        keys_to_remove = [k for k in self._retrievers if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._retrievers[key]

        logger.info(
            "tenant_cache_cleared",
            tenant_id=tenant_id,
            count=len(keys_to_remove),
        )

    def list_knowledge_bases(
        self,
        tenant_id: int | None = None,
    ) -> list[KnowledgeBaseConfig]:
        """列出知识库配置

        Args:
            tenant_id: 租户 ID（None 列出所有）

        Returns:
            知识库配置列表
        """
        if tenant_id is None:
            return list(self._kb_configs.values())

        prefix = f"{tenant_id}:"
        return [
            config for key, config in self._kb_configs.items()
            if key.startswith(prefix)
        ]


# 全局服务实例
_global_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """获取全局 RAG 服务实例

    Returns:
        RAG 服务实例
    """
    global _global_service
    if _global_service is None:
        _global_service = RAGService()
    return _global_service


__all__ = [
    # 数据模型
    "RetrievalContext",
    # 服务管理
    "RAGService",
    "get_rag_service",
]
