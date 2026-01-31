"""pgvector 向量存储

基于 pgvector 的 PostgreSQL 向量存储实现。
支持 HNSW 索引和余弦相似度搜索。

依赖安装:
    uv add pgvector langchain-postgres

参考实现:
    - WeKnora99/internal/application/repository/retriever/postgres/repository.go
    - https://github.com/pgvector/pgvector
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.observability.logging import get_logger
from app.vector_stores.base import (
    BaseVectorStore,
    IndexResult,
    SearchResult,
    VectorStats,
    VectorStoreConfig,
)

logger = get_logger(__name__)


# ============== pgvector 配置 ==============


@dataclass
class PgVectorConfig(VectorStoreConfig):
    """pgvector 配置

    Attributes:
        connection_string: PostgreSQL 连接字符串
        table_name: 向量表名称 (默认: embeddings)
        hnsw_m: HNSW 索引参数 M (默认: 16)
        hnsw_ef_construction: HNSW 索引构建参数 ef_construction (默认: 64)
        hnsw_ef_search: HNSW 搜索参数 ef (默认: 40)
        batch_size: 批量插入大小 (默认: 100)
    """

    connection_string: str | None = None
    table_name: str = "embeddings"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    batch_size: int = 100


# ============== pgvector 向量存储 ==============


class PgVectorStore(BaseVectorStore):
    """pgvector 向量存储

    使用 PostgreSQL + pgvector 扩展进行向量存储和搜索。

    Features:
        - HNSW 索引支持 (高效的近似最近邻搜索)
        - 余弦距离 (<=>) 运算符
        - 批量操作优化
        - 租户隔离支持
    """

    def __init__(
        self,
        config: PgVectorConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化 pgvector 向量存储

        Args:
            config: pgvector 配置
            embeddings: Embedding 实例
        """
        super().__init__(config, embeddings)
        self.pg_config: PgVectorConfig = config or PgVectorConfig()
        self._client: Any = None
        self._engine: Any = None

    async def initialize(self) -> None:
        """初始化 pgvector 存储

        - 创建 PostgreSQL 连接池
        - 创建向量表 (如不存在)
        - 创建 HNSW 索引 (如不存在)
        """
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine

            # 构建连接字符串
            conn_str = self.pg_config.connection_string
            if not conn_str:
                from app.config.settings import get_settings

                settings = get_settings()
                conn_str = settings.database_url

            # 创建异步引擎
            self._engine = create_async_engine(
                conn_str,
                pool_size=10,
                max_overflow=20,
                echo=False,
            )

            # 测试连接
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            # 确保 pgvector 扩展已安装
            async with self._engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # 创建向量表
            await self._create_table()

            # 创建 HNSW 索引
            await self._create_hnsw_index()

            self._initialized = True
            logger.info(
                "pgvector_initialized",
                table=self.pg_config.table_name,
                dimension=self.pg_config.dimension,
            )

        except ImportError as e:
            logger.error("pgvector_dependencies_not_installed")
            raise ImportError(
                "请安装 pgvector 依赖: uv add pgvector langchain-postgres"
            ) from e
        except Exception as e:
            logger.error("pgvector_init_failed", error=str(e))
            raise

    async def _create_table(self) -> None:
        """创建向量表"""
        from sqlalchemy import text

        table = self.pg_config.table_name
        dim = self.pg_config.dimension

        # 创建表的 SQL (对齐 WeKnora99 embeddings 表结构)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            source_id VARCHAR(64) NOT NULL,
            source_type INTEGER NOT NULL,
            chunk_id VARCHAR(64),
            knowledge_id VARCHAR(64),
            knowledge_base_id VARCHAR(64),
            content TEXT,
            dimension INTEGER NOT NULL,
            embedding VECTOR({dim}),
            is_enabled BOOLEAN DEFAULT TRUE,
            tag_id VARCHAR(36),
            tenant_id INTEGER
        );
        """

        # 创建索引 SQL
        create_indexes_sql = f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_kb_id ON {table}(knowledge_base_id);
        CREATE INDEX IF NOT EXISTS idx_{table}_chunk_id ON {table}(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_{table}_knowledge_id ON {table}(knowledge_id);
        CREATE INDEX IF NOT EXISTS idx_{table}_tag_id ON {table}(tag_id);
        CREATE INDEX IF NOT EXISTS idx_{table}_tenant_id ON {table}(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_{table}_enabled ON {table}(is_enabled);
        """

        async with self._engine.begin() as conn:
            await conn.execute(text(create_table_sql))
            await conn.execute(text(create_indexes_sql))

        logger.info("pgvector_table_created", table=table)

    async def _create_hnsw_index(self) -> None:
        """创建 HNSW 向量索引"""
        from sqlalchemy import text

        table = self.pg_config.table_name
        dim = self.pg_config.dimension
        m = self.pg_config.hnsw_m
        ef_const = self.pg_config.hnsw_ef_construction

        # HNSW 索引 SQL
        # 注意: pgvector 的 HNSW 索引需要指定维度
        create_index_sql = f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_embedding_hnsw
        ON {table} USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_const});
        """

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text(create_index_sql))
            logger.info("pgvector_hnsw_index_created", table=table, dim=dim)
        except Exception as e:
            # 如果 HNSW 索引已存在或不支持，跳过
            logger.warning("pgvector_hnsw_index_warning", error=str(e))

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            from sqlalchemy import text

            if self._engine is None:
                return False

            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning("pgvector_health_check_failed", error=str(e))
            return False

    async def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文档"""
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        import time

        from sqlalchemy import text

        # 生成 ID
        if ids is None:
            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # 嵌入文档
        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)

        # 批量插入
        table = self.pg_config.table_name
        batch_size = self.pg_config.batch_size

        async with self._engine.begin() as conn:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                batch_vectors = vectors[i : i + batch_size]

                # 构建批量插入 SQL
                values_placeholders = []
                params_list = []

                for j, (doc, doc_id, vector) in enumerate(
                    zip(batch_docs, batch_ids, batch_vectors, strict=True)
                ):
                    placeholder = f"(:id_{j}, :source_id_{j}, :source_type_{j}, :chunk_id_{j}, :knowledge_id_{j}, :knowledge_base_id_{j}, :content_{j}, :dimension_{j}, :embedding_{j}, :is_enabled_{j}, :tag_id_{j}, :tenant_id_{j})"
                    values_placeholders.append(placeholder)

                    metadata = doc.metadata or {}
                    params_list.append({
                        f"id_{j}": doc_id,
                        f"source_id_{j}": doc_id,
                        f"source_type_{j}": 1,  # chunk
                        f"chunk_id_{j}": doc_id,
                        f"knowledge_id_{j}": metadata.get("knowledge_id"),
                        f"knowledge_base_id_{j}": metadata.get("knowledge_base_id"),
                        f"content_{j}": doc.page_content,
                        f"dimension_{j}": len(vector),
                        f"embedding_{j}": vector,
                        f"is_enabled_{j}": True,
                        f"tag_id_{j}": metadata.get("tag_id"),
                        f"tenant_id_{j}": metadata.get("tenant_id"),
                    })

                # 合并参数字典
                merged_params = {}
                for p in params_list:
                    merged_params.update(p)

                # ruff: noqa: S608 (table 来自配置，非用户输入)
                insert_sql = f"""
                INSERT INTO {table}
                (id, source_id, source_type, chunk_id, knowledge_id, knowledge_base_id, content, dimension, embedding, is_enabled, tag_id, tenant_id)
                VALUES {', '.join(values_placeholders)}
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """

                await conn.execute(text(insert_sql), merged_params)

        logger.info(
            "pgvector_documents_added",
            count=len(documents),
            table=self.pg_config.table_name,
        )

        return IndexResult(ids=ids, count=len(documents))

    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文本"""
        documents = [
            Document(
                page_content=text,
                metadata=(metadatas or [{}])[i] if metadatas else {},
            )
            for i, text in enumerate(texts)
        ]
        return await self.add_documents(documents, ids)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """相似度搜索"""
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        # 嵌入查询
        query_vector = await self.embeddings.aembed_query(query)

        return await self.search_by_vector(query_vector, k, score_threshold, filter_dict)

    async def search_by_vector(
        self,
        vector: list[float],
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """通过向量搜索

        使用 pgvector 的余弦距离运算符 (<=>) 进行搜索。
        HNSW 索引会自动被优化器使用。
        """
        from sqlalchemy import text

        await self.ensure_initialized()

        table = self.pg_config.table_name
        dim = self.pg_config.dimension

        # 构建过滤条件
        where_clauses = ["is_enabled = TRUE"]
        params: dict[str, Any] = {
            "query_vector": vector,
            "limit": k,
        }

        if filter_dict:
            if "knowledge_base_id" in filter_dict:
                where_clauses.append("knowledge_base_id = :kb_id")
                params["kb_id"] = filter_dict["knowledge_base_id"]
            if "knowledge_id" in filter_dict:
                where_clauses.append("knowledge_id = :kid")
                params["kid"] = filter_dict["knowledge_id"]
            if "tag_id" in filter_dict:
                where_clauses.append("tag_id = :tid")
                params["tid"] = filter_dict["tag_id"]
            if "tenant_id" in filter_dict:
                where_clauses.append("tenant_id = :tid2")
                params["tid2"] = filter_dict["tenant_id"]

        # 应用距离阈值过滤
        if score_threshold is not None:
            # 余弦距离 = 1 - 余弦相似度
            # 如果 score_threshold 是相似度，则转换为距离
            distance_threshold = 1 - score_threshold
            where_clauses.append("distance <= :distance_threshold")
            params["distance_threshold"] = distance_threshold

        where_sql = " AND ".join(where_clauses)

        # 使用子查询策略 (对齐 WeKnora99)
        # 先获取候选集，再过滤阈值，最后限制结果数量
        # ruff: noqa: S608 (table 和 where_sql 来自配置，非用户输入)
        search_sql = f"""
        SELECT
            id, content, chunk_id, knowledge_id, knowledge_base_id, tag_id,
            (1 - distance) as score
        FROM (
            SELECT
                id, content, chunk_id, knowledge_id, knowledge_base_id, tag_id,
                embedding::vector({dim}) <=> :query_vector::vector({dim}) as distance
            FROM {table}
            WHERE {where_sql}
            ORDER BY embedding::vector({dim}) <=> :query_vector::vector({dim})
            LIMIT :limit
        ) AS candidates
        ORDER BY distance ASC
        LIMIT :limit
        """

        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text(search_sql), params)
                rows = result.fetchall()

            # 转换结果
            search_results = []
            for row in rows:
                score = float(row.score) if row.score else 0.0
                search_results.append(
                    SearchResult(
                        content=row.content or "",
                        metadata={
                            "chunk_id": row.chunk_id,
                            "knowledge_id": row.knowledge_id,
                            "knowledge_base_id": row.knowledge_base_id,
                            "tag_id": row.tag_id,
                        },
                        score=score,
                        id=str(row.id),
                    )
                )

            logger.info(
                "pgvector_search_completed",
                query_vector_dim=len(vector),
                result_count=len(search_results),
            )

            return search_results

        except Exception as e:
            logger.error("pgvector_search_failed", error=str(e))
            return []

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        await self.ensure_initialized()

        from sqlalchemy import text

        table = self.pg_config.table_name
        # ruff: noqa: S608 (table 来自配置，非用户输入)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {table} WHERE id = ANY(:ids)"),
                {"ids": ids},
            )

        logger.info(
            "pgvector_documents_deleted",
            count=len(ids),
            table=self.pg_config.table_name,
        )
        return True

    async def delete_collection(self) -> bool:
        """删除整个集合 (清空表)"""
        await self.ensure_initialized()

        from sqlalchemy import text

        table = self.pg_config.table_name
        # ruff: noqa: S608 (table 来自配置，非用户输入)
        async with self._engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE TABLE {table}"))

        logger.info("pgvector_collection_cleared", table=self.pg_config.table_name)
        return True

    async def get_stats(self) -> VectorStats:
        """获取统计信息"""
        await self.ensure_initialized()

        from sqlalchemy import text

        table = self.pg_config.table_name

        try:
            async with self._engine.begin() as conn:
                # 获取向量总数
                # ruff: noqa: S608 (table 来自配置，非用户输入)
                count_result = await conn.execute(
                    text(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL")
                )
                total_vectors = count_result.scalar() or 0

                # 获取集合数量 (按 knowledge_base_id 分组)
                # ruff: noqa: S608 (table 来自配置，非用户输入)
                kb_result = await conn.execute(
                    text(f"SELECT COUNT(DISTINCT knowledge_base_id) FROM {table}")
                )
                collections = kb_result.scalar() or 0

            return VectorStats(
                total_vectors=total_vectors,
                collections=collections,
                dimension=self.pg_config.dimension,
                metric=self.pg_config.metric,
            )
        except Exception as e:
            logger.error("pgvector_get_stats_failed", error=str(e))
            return VectorStats(
                total_vectors=0,
                collections=0,
                dimension=self.pg_config.dimension,
                metric=self.pg_config.metric,
            )


__all__ = [
    "PgVectorConfig",
    "PgVectorStore",
]
