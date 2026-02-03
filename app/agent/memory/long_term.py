"""长期记忆实现

基于向量存储的长期记忆，支持语义检索。
"""

from typing import Any

from langchain_core.embeddings import Embeddings

from app.agent.memory.base import BaseLongTermMemory
from app.config.settings import get_settings
from app.llm.embeddings import get_embeddings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LongTermMemory(BaseLongTermMemory):
    """长期记忆

    使用向量存储实现语义检索的长期记忆功能。
    支持多种向量存储后端（需要相应依赖）。
    """

    def __init__(
        self,
        collection_name: str = "kiki_memories",
        embeddings: Embeddings | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        """初始化长期记忆

        Args:
            collection_name: 集合名称
            embeddings: 自定义 Embeddings 实例（优先使用）
            embedding_provider: Embedding 提供商（openai/dashscope/voyage/ollama）
        """
        self.collection_name = collection_name
        self._custom_embeddings = embeddings
        self._embedding_provider = embedding_provider
        self._vector_store = None
        self._embeddings = None

        logger.debug(
            "long_term_memory_initialized",
            collection_name=collection_name,
            has_custom_embeddings=embeddings is not None,
        )

    def _get_embeddings(self) -> Embeddings:
        """获取 Embeddings 实例"""
        if self._embeddings is not None:
            return self._embeddings

        if self._custom_embeddings is not None:
            self._embeddings = self._custom_embeddings
        else:
            provider = self._embedding_provider or settings.embedding_provider
            self._embeddings = get_embeddings(provider=provider)

        return self._embeddings

    async def _get_vector_store(self):
        """获取向量存储实例（延迟初始化）"""
        if self._vector_store is not None:
            return self._vector_store

        # 尝试使用 LangChain 的向量存储集成
        try:
            storage_type = getattr(settings, "vector_store_type", "memory")

            if storage_type == "pgvector":
                return await self._init_pgvector()
            elif storage_type == "pinecone":
                return await self._init_pinecone()
            elif storage_type == "chroma":
                return await self._init_chroma()
            else:
                # 默认使用内存向量存储
                return await self._init_memory_store()

        except Exception as e:
            logger.warning("vector_store_init_failed", error=str(e))
            return None

    async def _init_memory_store(self):
        """初始化内存向量存储"""
        try:
            from langchain_community.vectorstores import InMemoryVectorStore

            embeddings = self._get_embeddings()
            self._vector_store = InMemoryVectorStore(embeddings)
            logger.info("memory_vector_store_initialized")
            return self._vector_store

        except ImportError:
            logger.warning("memory_vector_store_dependencies_missing")
            return None

    async def _init_pgvector(self):
        """初始化 PGVector"""
        try:
            from langchain_community.vectorstores import PGVector

            embeddings = self._get_embeddings()
            connection_string = settings.database_url
            if connection_string.startswith("postgresql+asyncpg://"):
                connection_string = connection_string.replace(
                    "postgresql+asyncpg://", "postgresql://"
                )

            self._vector_store = PGVector(
                collection_name=self.collection_name,
                connection_string=connection_string,
                embedding_function=embeddings,
            )
            logger.info("pgvector_initialized")
            return self._vector_store

        except ImportError:
            logger.warning("pgvector_dependencies_missing")
            return await self._init_memory_store()

    async def _init_pinecone(self):
        """初始化 Pinecone"""
        try:
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone

            # 需要 PINECONE_API_KEY 配置
            api_key = settings.pinecone_api_key
            if not api_key:
                logger.warning("pinecone_api_key_not_configured")
                return await self._init_memory_store()

            pc = Pinecone(api_key=api_key)
            index_name = settings.pinecone_index_name
            index = pc.Index(index_name)

            embeddings = self._get_embeddings()
            self._vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
            )
            logger.info("pinecone_initialized")
            return self._vector_store

        except ImportError:
            logger.warning("pinecone_dependencies_missing")
            return await self._init_memory_store()

    async def _init_chroma(self):
        """初始化 Chroma"""
        try:
            from langchain_chroma import Chroma

            embeddings = self._get_embeddings()
            persist_directory = settings.chroma_persist_directory

            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            logger.info("chroma_initialized")
            return self._vector_store

        except ImportError:
            logger.warning("chroma_dependencies_missing")
            return await self._init_memory_store()

    async def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加长期记忆

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            记忆 ID
        """
        vector_store = await self._get_vector_store()
        if not vector_store:
            logger.warning("vector_store_not_available")
            return ""

        metadata = metadata or {}
        memory_id = f"mem_{hash(content)}_{id(content)}"

        # 添加到向量存储
        await vector_store.aadd_texts(
            texts=[content],
            metadatas=[{**metadata, "memory_id": memory_id}],
        )

        logger.debug("memory_added", memory_id=memory_id)
        return memory_id

    async def search_memories(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索记忆

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            记忆列表
        """
        vector_store = await self._get_vector_store()
        if not vector_store:
            return []

        try:
            if filter:
                # 支持过滤的向量存储
                results = await vector_store.asimilarity_search_with_score(
                    query,
                    k=k,
                    filter=filter,
                )
            else:
                results = await vector_store.asimilarity_search_with_score(
                    query,
                    k=k,
                )

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in results
            ]

        except Exception as e:
            logger.warning("memory_search_failed", error=str(e))
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            是否成功删除
        """
        vector_store = await self._get_vector_store()
        if not vector_store:
            return False

        try:
            await vector_store.adelete([memory_id])
            logger.info("memory_deleted", memory_id=memory_id)
            return True
        except Exception as e:
            logger.warning("memory_delete_failed", error=str(e))
            return False

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆 ID
            content: 新内容
            metadata: 新元数据

        Returns:
            是否成功更新
        """
        # 向量存储通常不支持原地更新，需要删除后重新添加
        if content:
            await self.delete_memory(memory_id)
            await self.add_memory(content, metadata)
            return True
        return False
