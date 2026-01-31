"""知识搜索服务

提供独立知识搜索功能，支持多种检索器。
"""

from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.llm.embeddings import get_embeddings
from app.llm.providers import LLMPriority, get_llm_for_task
from app.observability.logging import get_logger
from app.repositories.chunk import ChunkRepository
from app.vector_stores import (
    BaseVectorStore,
    VectorStoreConfig,
    create_vector_store,
)

logger = get_logger(__name__)
settings = get_settings()


class KnowledgeSearchService:
    """知识搜索服务

    支持多种检索器进行知识搜索和问答。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self._chunk_repo: ChunkRepository | None = None

    @property
    def chunk_repo(self) -> ChunkRepository:
        """延迟初始化分块仓储"""
        if self._chunk_repo is None:
            self._chunk_repo = ChunkRepository(self.session)
        return self._chunk_repo

    async def search(
        self,
        query: str,
        knowledge_base_id: str,
        tenant_id: int,
        retriever_type: Literal["vector", "bm25", "ensemble", "conversational"] = "vector",
        top_k: int = 5,
        score_threshold: float | None = None,
        chat_history: list[dict[str, str]] | None = None,
        enable_rerank: bool = False,
        filter_dict: dict[str, Any] | None = None,
        embedding_model_id: str | None = None,
        summary_model_id: str | None = None,
    ) -> dict[str, Any]:
        """执行知识搜索

        Args:
            query: 查询问题
            knowledge_base_id: 知识库 ID
            tenant_id: 租户 ID
            retriever_type: 检索器类型
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            chat_history: 聊天历史
            enable_rerank: 是否启用重排序
            filter_dict: 过滤条件
            embedding_model_id: 嵌入模型 ID
            summary_model_id: 摘要模型 ID

        Returns:
            搜索结果字典，包含：
            - answer: 生成的答案
            - documents: 检索到的文档
            - question: 重述后的问题（conversational 模式）
        """
        logger.info(
            "knowledge_search_start",
            kb_id=knowledge_base_id,
            query=query[:100],
            retriever_type=retriever_type,
            top_k=top_k,
        )

        try:
            # 1. 创建检索器
            retriever = await self._create_retriever(
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                retriever_type=retriever_type,
                top_k=top_k,
                score_threshold=score_threshold,
                embedding_model_id=embedding_model_id,
            )

            # 2. 执行检索
            if retriever_type == "conversational":
                result = await self._conversational_search(
                    retriever=retriever,
                    query=query,
                    chat_history=chat_history or [],
                    summary_model_id=summary_model_id,
                )
            else:
                result = await self._direct_search(
                    retriever=retriever,
                    query=query,
                    summary_model_id=summary_model_id,
                )

            # 3. 重排序（可选）
            if enable_rerank:
                result["documents"] = await self._rerank_documents(
                    query=query,
                    documents=result["documents"],
                    top_k=top_k,
                )

            logger.info(
                "knowledge_search_complete",
                kb_id=knowledge_base_id,
                query=query[:100],
                retriever_type=retriever_type,
                document_count=len(result.get("documents", [])),
            )

            return result

        except Exception as e:
            logger.error(
                "knowledge_search_failed",
                kb_id=knowledge_base_id,
                query=query[:100],
                error=str(e),
            )
            raise

    async def _create_retriever(
        self,
        knowledge_base_id: str,
        tenant_id: int,
        retriever_type: Literal["vector", "bm25", "ensemble", "conversational"],
        top_k: int,
        score_threshold: float | None,
        embedding_model_id: str | None,
    ) -> BaseRetriever:
        """创建检索器

        Args:
            knowledge_base_id: 知识库 ID
            tenant_id: 租户 ID
            retriever_type: 检索器类型
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            embedding_model_id: 嵌入模型 ID

        Returns:
            检索器实例
        """
        from app.retrievers import (
            ConversationalRetriever,
            ConversationalRetrieverConfig,
        )

        if retriever_type == "vector":
            return await self._create_vector_retriever(
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                top_k=top_k,
                score_threshold=score_threshold,
                embedding_model_id=embedding_model_id,
            )

        elif retriever_type == "bm25":
            return await self._create_bm25_retriever(
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                top_k=top_k,
                score_threshold=score_threshold,
            )

        elif retriever_type == "ensemble":
            return await self._create_ensemble_retriever(
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                top_k=top_k,
                score_threshold=score_threshold,
                embedding_model_id=embedding_model_id,
            )

        elif retriever_type == "conversational":
            # 创建底层向量检索器
            vector_retriever = await self._create_vector_retriever(
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                top_k=top_k,
                score_threshold=score_threshold,
                embedding_model_id=embedding_model_id,
            )

            # 创建对话式检索器
            config = ConversationalRetrieverConfig(
                k=top_k,
                score_threshold=score_threshold,
            )
            return ConversationalRetriever(
                config=config,
                retriever=vector_retriever,
            )

        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

    async def _create_vector_retriever(
        self,
        knowledge_base_id: str,
        tenant_id: int,
        top_k: int,
        score_threshold: float | None,
        embedding_model_id: str | None,
    ) -> BaseRetriever:
        """创建向量检索器

        Args:
            knowledge_base_id: 知识库 ID
            tenant_id: 租户 ID
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            embedding_model_id: 嵌入模型 ID

        Returns:
            向量检索器
        """
        # 获取嵌入模型
        embeddings = get_embeddings()

        # 创建向量存储
        config = VectorStoreConfig(
            collection_name=f"kb_{knowledge_base_id}",
            tenant_id=tenant_id,
        )
        vector_store: BaseVectorStore = create_vector_store(
            settings.vector_store_type,
            config,
            embeddings,
        )

        # 确保初始化
        await vector_store.ensure_initialized()

        # 创建检索器
        retriever = await vector_store.as_retriever(
            k=top_k,
            score_threshold=score_threshold,
        )

        logger.info(
            "vector_retriever_created",
            kb_id=knowledge_base_id,
            store_type=settings.vector_store_type,
        )

        return retriever

    async def _create_bm25_retriever(
        self,
        knowledge_base_id: str,
        tenant_id: int,
        top_k: int,
        score_threshold: float | None,
    ) -> BaseRetriever:
        """创建 BM25 检索器

        Args:
            knowledge_base_id: 知识库 ID
            tenant_id: 租户 ID
            top_k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            BM25 检索器
        """
        from app.retrievers import BM25Retriever, BM25RetrieverConfig

        # 获取知识库文档
        chunks = await self.chunk_repo.list_by_knowledge(
            knowledge_id=knowledge_base_id,
            tenant_id=tenant_id,
            params=type("Params", (), {"page": 1, "size": 10000})(),  # type: ignore
        )

        # 创建 BM25 检索器
        config = BM25RetrieverConfig(
            k=top_k,
            score_threshold=score_threshold,
        )
        retriever = BM25Retriever(config)

        # 构建索引
        texts = [c.content for c in chunks.items]
        metadatas = [
            {
                "chunk_id": c.id,
                "knowledge_id": c.knowledge_id,
                "knowledge_title": c.knowledge_id,  # 简化
            }
            for c in chunks.items
        ]
        await retriever.from_texts(texts, metadatas)

        logger.info(
            "bm25_retriever_created",
            kb_id=knowledge_base_id,
            document_count=len(texts),
        )

        return retriever

    async def _create_ensemble_retriever(
        self,
        knowledge_base_id: str,
        tenant_id: int,
        top_k: int,
        score_threshold: float | None,
        embedding_model_id: str | None,
    ) -> BaseRetriever:
        """创建集成检索器

        Args:
            knowledge_base_id: 知识库 ID
            tenant_id: 租户 ID
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            embedding_model_id: 嵌入模型 ID

        Returns:
            集成检索器
        """
        from app.retrievers import (
            EnsembleRetriever,
            EnsembleRetrieverConfig,
        )

        # 创建向量检索器
        vector_retriever = await self._create_vector_retriever(
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
            top_k=top_k,
            score_threshold=score_threshold,
            embedding_model_id=embedding_model_id,
        )

        # 创建 BM25 检索器
        bm25_retriever = await self._create_bm25_retriever(
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 创建集成检索器
        config = EnsembleRetrieverConfig(
            weights=[0.7, 0.3],  # 向量 70%, BM25 30%
        )
        retriever = EnsembleRetriever(
            config=config,
            retrievers=[vector_retriever, bm25_retriever],
        )

        logger.info(
            "ensemble_retriever_created",
            kb_id=knowledge_base_id,
        )

        return retriever

    async def _conversational_search(
        self,
        retriever: BaseRetriever,
        query: str,
        chat_history: list[dict[str, str]],
        summary_model_id: str | None,
    ) -> dict[str, Any]:
        """对话式搜索

        Args:
            retriever: 检索器
            query: 查询问题
            chat_history: 聊天历史
            summary_model_id: 摘要模型 ID

        Returns:
            搜索结果
        """
        from app.retrievers import SearchResult

        # 设置 LLM
        llm = get_llm_for_task(priority=LLMPriority.BALANCED)

        if hasattr(retriever, "set_llm"):
            retriever.set_llm(llm)

        # 转换历史格式
        history_tuples = []
        if chat_history:
            for item in chat_history:
                if "question" in item and "answer" in item:
                    history_tuples.append((item["question"], item["answer"]))

        # 执行搜索
        result: SearchResult = await retriever.asearch(
            query=query,
            chat_history=history_tuples,
        )

        return {
            "answer": result.answer,
            "documents": result.source_documents,
            "question": result.question,
        }

    async def _direct_search(
        self,
        retriever: BaseRetriever,
        query: str,
        summary_model_id: str | None,
    ) -> dict[str, Any]:
        """直接搜索

        Args:
            retriever: 检索器
            query: 查询问题
            summary_model_id: 摘要模型 ID

        Returns:
            搜索结果
        """
        # 执行检索
        if hasattr(retriever, "aretrieve"):
            documents = await retriever.aretrieve(query)
        else:
            documents = retriever.get_relevant_documents(query)

        # 生成答案
        answer = await self._generate_answer(
            query=query,
            documents=documents,
            model_id=summary_model_id,
        )

        return {
            "answer": answer,
            "documents": documents,
        }

    async def _generate_answer(
        self,
        query: str,
        documents: list[Document],
        model_id: str | None,
    ) -> str:
        """生成答案

        Args:
            query: 查询问题
            documents: 检索到的文档
            model_id: 模型 ID

        Returns:
            生成的答案
        """
        if not documents:
            return "抱歉，没有找到相关信息。"

        try:
            llm = get_llm_for_task(priority=LLMPriority.BALANCED)

            # 构建上下文
            context_parts = []
            for doc in documents[:5]:  # 最多使用5个文档
                context_parts.append(doc.page_content[:500])

            context = "\n\n".join(context_parts)

            # 构建提示
            prompt = (
                "基于以下上下文回答问题。如果上下文中没有相关信息，"
                "请直接说明不知道，不要编造答案。\n\n"
                f"上下文：\n{context}\n\n"
                f"问题：{query}\n\n"
                "答案："
            )

            # 生成答案
            answer = await llm.apredict(prompt)

            return answer.strip()

        except Exception as e:
            logger.error(
                "generate_answer_failed",
                query=query[:100],
                error=str(e),
            )

            # 降级：返回文档摘要
            return self._summarize_documents(documents)

    def _summarize_documents(self, documents: list[Document]) -> str:
        """摘要文档

        Args:
            documents: 文档列表

        Returns:
            文档摘要
        """
        if not documents:
            return "抱歉，没有找到相关信息。"

        summary_parts = []
        for i, doc in enumerate(documents[:3], 1):
            content = doc.page_content[:200]
            summary_parts.append(f"{i}. {content}...")

        return "根据知识库找到以下相关信息：\n\n" + "\n".join(summary_parts)

    async def _rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """重排序文档

        Args:
            query: 查询问题
            documents: 文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档
        """
        # 简单实现：按原分数排序
        # 实际项目中可以接入专门的重排序模型
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get("score", 0.0),
            reverse=True,
        )

        return sorted_docs[:top_k]


__all__ = ["KnowledgeSearchService"]
