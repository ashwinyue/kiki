"""文档索引器

提供批量索引、索引更新和删除功能。

使用示例:
    ```python
    from app.services.search.indexer import DocumentIndexer
    from app.services.search.elasticsearch import get_elasticsearch_client

    es = await get_elasticsearch_client()
    indexer = DocumentIndexer(es)

    # 批量索引
    await indexer.bulk_index("my_index", documents)

    # 更新文档
    await indexer.update_document("my_index", "doc1", {"title": "New Title"})
    ```
"""

from __future__ import annotations

from typing import Any

from app.observability.logging import get_logger
from app.services.search.elasticsearch import ElasticsearchClient

logger = get_logger(__name__)


class DocumentIndexer:
    """文档索引器

    提供批量索引、更新和删除文档的功能。
    支持多种文档格式的自动转换。
    """

    def __init__(
        self,
        client: ElasticsearchClient,
        default_refresh: bool = False,
        bulk_size: int = 500,
    ):
        """初始化文档索引器

        Args:
            client: Elasticsearch 客户端
            default_refresh: 是否默认立即刷新
            bulk_size: 批量操作大小
        """
        self.client = client
        self.default_refresh = default_refresh
        self.bulk_size = bulk_size
        self._pending_actions: list[dict[str, Any]] = []

    # ============== 单文档操作 ==============

    async def index_document(
        self,
        index: str,
        document: dict[str, Any],
        id: str | None = None,
        refresh: bool | None = None,
    ) -> str | None:
        """索引单个文档

        Args:
            index: 索引名称
            document: 文档内容
            id: 文档 ID（None 自动生成）
            refresh: 是否立即刷新

        Returns:
            文档 ID
        """
        refresh = refresh if refresh is not None else self.default_refresh

        # 预处理文档
        processed = self._preprocess_document(document)

        return await self.client.index_document(
            index=index,
            id=id,
            document=processed,
            refresh=refresh,
        )

    async def update_document(
        self,
        index: str,
        id: str,
        updates: dict[str, Any],
        refresh: bool | None = None,
    ) -> bool:
        """更新文档

        Args:
            index: 索引名称
            id: 文档 ID
            updates: 更新内容
            refresh: 是否立即刷新

        Returns:
            是否成功
        """
        refresh = refresh if refresh is not None else self.default_refresh

        return await self.client.update_document(
            index=index,
            id=id,
            updates=updates,
            refresh=refresh,
        )

    async def upsert_document(
        self,
        index: str,
        document: dict[str, Any],
        id: str,
        refresh: bool | None = None,
    ) -> str | None:
        """插入或更新文档（如果存在则更新）

        Args:
            index: 索引名称
            document: 文档内容
            id: 文档 ID
            refresh: 是否立即刷新

        Returns:
            文档 ID
        """
        refresh = refresh if refresh is not None else self.default_refresh

        # 使用 doc_as_upsert 实现插入或更新
        processed = self._preprocess_document(document)

        try:
            resolved_index = self.client._resolve_index(index)
            await self.client.client.update(
                index=resolved_index,
                id=id,
                doc=processed,
                doc_as_upsert=True,
                refresh=refresh,
            )

            logger.debug("document_upserted", index=resolved_index, id=id)
            return id

        except Exception as e:
            logger.error("document_upsert_failed", index=index, id=id, error=str(e))
            return None

    async def delete_document(
        self,
        index: str,
        id: str,
        refresh: bool | None = None,
    ) -> bool:
        """删除文档

        Args:
            index: 索引名称
            id: 文档 ID
            refresh: 是否立即刷新

        Returns:
            是否成功
        """
        refresh = refresh if refresh is not None else self.default_refresh

        return await self.client.delete_document(
            index=index,
            id=id,
            refresh=refresh,
        )

    # ============== 批量操作 ==============

    async def bulk_index(
        self,
        index: str,
        documents: list[dict[str, Any]],
        ids: list[str] | None = None,
        op_type: str = "index",
        refresh: bool | None = None,
    ) -> tuple[int, int]:
        """批量索引文档

        Args:
            index: 索引名称
            documents: 文档列表
            ids: 文档 ID 列表（可选）
            op_type: 操作类型 (index/create/update/delete)
            refresh: 是否立即刷新

        Returns:
            (成功数, 失败数)
        """
        refresh = refresh if refresh is not None else self.default_refresh

        actions = []
        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids and i < len(ids) else None
            processed = self._preprocess_document(doc)

            action = {
                "_op_type": op_type,
                "_index": index,
                "_id": doc_id,
                "_source": processed if op_type in ("index", "create") else None,
            }

            if op_type == "update":
                action["doc"] = processed

            actions.append(action)

        return await self.client.bulk_index(actions, refresh=refresh)

    async def bulk_delete(
        self,
        index: str,
        ids: list[str],
        refresh: bool | None = None,
    ) -> tuple[int, int]:
        """批量删除文档

        Args:
            index: 索引名称
            ids: 文档 ID 列表
            refresh: 是否立即刷新

        Returns:
            (成功数, 失败数)
        """
        refresh = refresh if refresh is not None else self.default_refresh

        actions = [
            {
                "_op_type": "delete",
                "_index": index,
                "_id": doc_id,
            }
            for doc_id in ids
        ]

        return await self.client.bulk_index(actions, refresh=refresh)

    async def bulk_update(
        self,
        index: str,
        updates: list[tuple[str, dict[str, Any]]],
        refresh: bool | None = None,
    ) -> tuple[int, int]:
        """批量更新文档

        Args:
            index: 索引名称
            updates: (文档 ID, 更新内容) 列表
            refresh: 是否立即刷新

        Returns:
            (成功数, 失败数)
        """
        refresh = refresh if refresh is not None else self.default_refresh

        actions = []
        for doc_id, update in updates:
            actions.append({
                "_op_type": "update",
                "_index": index,
                "_id": doc_id,
                "doc": update,
            })

        return await self.client.bulk_index(actions, refresh=refresh)

    # ============== 增量索引 ==============

    def add_to_batch(
        self,
        index: str,
        document: dict[str, Any],
        id: str | None = None,
        op_type: str = "index",
    ) -> None:
        """添加到批量操作缓冲区

        Args:
            index: 索引名称
            document: 文档内容
            id: 文档 ID
            op_type: 操作类型
        """
        processed = self._preprocess_document(document)

        self._pending_actions.append({
            "_op_type": op_type,
            "_index": index,
            "_id": id,
            "_source": processed if op_type in ("index", "create") else None,
        })

        # 达到批量大小时自动提交
        if len(self._pending_actions) >= self.bulk_size:
            raise BufferFullError

    async def flush_batch(self, refresh: bool | None = None) -> tuple[int, int]:
        """提交批量操作缓冲区

        Args:
            refresh: 是否立即刷新

        Returns:
            (成功数, 失败数)
        """
        if not self._pending_actions:
            return 0, 0

        refresh = refresh if refresh is not None else self.default_refresh

        actions = self._pending_actions.copy()
        self._pending_actions.clear()

        return await self.client.bulk_index(actions, refresh=refresh)

    def clear_batch(self) -> None:
        """清空批量操作缓冲区"""
        self._pending_actions.clear()

    @property
    def pending_count(self) -> int:
        """获取缓冲区操作数量"""
        return len(self._pending_actions)

    # ============== 删除操作 ==============

    async def delete_by_query(
        self,
        index: str,
        query: dict[str, Any],
        refresh: bool | None = None,
    ) -> int:
        """按查询删除文档

        Args:
            index: 索引名称
            query: 删除查询
            refresh: 是否立即刷新

        Returns:
            删除的文档数
        """
        refresh = refresh if refresh is not None else self.default_refresh

        return await self.client.delete_by_query(index, query, refresh)

    async def delete_all(
        self,
        index: str,
        refresh: bool | None = None,
    ) -> int:
        """删除索引中的所有文档

        Args:
            index: 索引名称
            refresh: 是否立即刷新

        Returns:
            删除的文档数
        """
        query = {"match_all": {}}
        return await self.delete_by_query(index, query, refresh)

    # ============== 重建索引 ==============

    async def reindex(
        self,
        source_index: str,
        target_index: str,
        query: dict[str, Any] | None = None,
        refresh: bool | None = None,
    ) -> int:
        """重建索引

        Args:
            source_index: 源索引
            target_index: 目标索引
            query: 过滤查询（None 表示全部）
            refresh: 是否立即刷新

        Returns:
            重建的文档数
        """
        refresh = refresh if refresh is not None else self.default_refresh

        count = 0
        async for doc in self.client.scroll_search(
            index=source_index,
            query=query or {"match_all": {}},
            size=100,
        ):
            source = doc.get("_source", {})
            doc_id = doc.get("_id")

            result = await self.index_document(
                index=target_index,
                document=source,
                id=doc_id,
                refresh=False,
            )

            if result:
                count += 1

        if refresh:
            await self.client.refresh_index(target_index)

        logger.info(
            "reindex_completed",
            source=source_index,
            target=target_index,
            count=count,
        )

        return count

    # ============== 私有方法 ==============

    def _preprocess_document(self, document: dict[str, Any]) -> dict[str, Any]:
        """预处理文档

        Args:
            document: 原始文档

        Returns:
            处理后的文档
        """
        processed = {}

        for key, value in document.items():
            # 跳过 None 值
            if value is None:
                continue

            # 转换列表为字符串（用于搜索）
            if isinstance(value, list):
                if value and isinstance(value[0], (str, int, float, bool)):
                    processed[key] = " ".join(str(v) for v in value)
                else:
                    processed[key] = value
            else:
                processed[key] = value

        return processed


class BufferFullError(Exception):
    """批量缓冲区已满"""
    pass


# ============== 便捷函数 ==============


async def create_document_indexer(
    client: ElasticsearchClient | None = None,
    **kwargs,
) -> DocumentIndexer:
    """创建文档索引器

    Args:
        client: Elasticsearch 客户端（None 则使用全局实例）
        **kwargs: 其他参数

    Returns:
        DocumentIndexer 实例
    """
    if client is None:
        from app.services.search.elasticsearch import get_elasticsearch_client

        client = await get_elasticsearch_client()

    return DocumentIndexer(client, **kwargs)
