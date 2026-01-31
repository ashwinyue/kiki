"""Elasticsearch 客户端封装

提供异步 Elasticsearch 客户端，支持：
- 索引管理（创建、删除、刷新）
- 文档 CRUD 操作
- 全文搜索
- 聚合查询
- 多租户隔离

使用示例:
    ```python
    from app.services.search.elasticsearch import get_elasticsearch_client

    # 获取客户端
    es = await get_elasticsearch_client()

    # 创建索引
    await es.create_index("my_index", {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"}
            }
        }
    })

    # 索引文档
    await es.index_document("my_index", "1", {
        "title": "Hello",
        "content": "World"
    })

    # 搜索
    results = await es.search("my_index", {
        "query": {
            "match": {"content": "hello"}
        }
    })
    ```
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.observability.logging import get_logger

logger = get_logger(__name__)

# Elasticsearch 异步客户端
try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk

    _elasticsearch_available = True
except ImportError:
    _elasticsearch_available = False
    AsyncElasticsearch = None  # type: ignore
    async_bulk = None  # type: ignore

    logger.warning("elasticsearch_not_installed")


# ============== Elasticsearch 客户端 ==============


class ElasticsearchClient:
    """Elasticsearch 异步客户端封装

    提供生产级的 Elasticsearch 操作接口，支持：
    - 连接池管理
    - 自动重连
    - 多租户隔离（通过索引命名空间）
    - 批量操作
    - 错误处理和日志记录
    """

    def __init__(
        self,
        hosts: str | list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        index_prefix: str = "",
        verify_certs: bool = True,
        **kwargs,
    ):
        """初始化 Elasticsearch 客户端

        Args:
            hosts: Elasticsearch 主机地址，支持多个
            username: 用户名
            password: 密码
            api_key: API Key（替代用户名密码）
            index_prefix: 索引前缀（用于多租户隔离）
            verify_certs: 是否验证证书
            **kwargs: 其他 elasticsearch-py 参数
        """
        if not _elasticsearch_available:
            raise ImportError(
                "elasticsearch is not installed. "
                "Install with: uv add 'elasticsearch[async]'"
            )

        self.hosts = hosts or ["http://localhost:9200"]
        self.username = username
        self.password = password
        self.api_key = api_key
        self.index_prefix = index_prefix
        self.verify_certs = verify_certs
        self._client: AsyncElasticsearch | None = None
        self._extra_kwargs = kwargs

    async def connect(self) -> None:
        """连接 Elasticsearch（幂等）"""
        if self._client is not None:
            return

        # 构建认证信息
        if self.api_key:
            auth = {"api_key": self.api_key}
        elif self.username and self.password:
            auth = {"basic_auth": (self.username, self.password)}
        else:
            auth = {}

        try:
            self._client = AsyncElasticsearch(
                hosts=self.hosts,
                verify_certs=self.verify_certs,
                **auth,
                **self._extra_kwargs,
            )

            # 测试连接
            info = await self._client.info()
            logger.info(
                "elasticsearch_connected",
                cluster_name=info.get("cluster_name"),
                version=info.get("version", {}).get("number"),
            )

        except Exception as e:
            logger.error("elasticsearch_connect_failed", error=str(e))
            raise

    async def close(self) -> None:
        """关闭客户端连接"""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("elasticsearch_closed")

    @property
    def client(self) -> AsyncElasticsearch:
        """获取底层客户端（确保已连接）"""
        if self._client is None:
            raise RuntimeError("Elasticsearch client is not connected. Call connect() first.")
        return self._client

    def _resolve_index(self, index: str) -> str:
        """解析索引名称（添加前缀）

        Args:
            index: 原始索引名

        Returns:
            带前缀的索引名
        """
        if self.index_prefix and not index.startswith(self.index_prefix):
            return f"{self.index_prefix}_{index}"
        return index

    # ============== 索引管理 ==============

    async def index_exists(self, index: str) -> bool:
        """检查索引是否存在

        Args:
            index: 索引名称

        Returns:
            是否存在
        """
        try:
            resolved_index = self._resolve_index(index)
            return await self.client.indices.exists(index=resolved_index)
        except Exception as e:
            logger.error("index_exists_check_failed", index=index, error=str(e))
            return False

    async def create_index(
        self,
        index: str,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> bool:
        """创建索引

        Args:
            index: 索引名称
            mappings: 索引映射
            settings: 索引设置

        Returns:
            是否成功
        """
        try:
            resolved_index = self._resolve_index(index)

            body: dict[str, Any] = {}
            if mappings:
                body["mappings"] = mappings
            if settings:
                body["settings"] = settings

            await self.client.indices.create(index=resolved_index, body=body)

            logger.info("index_created", index=resolved_index)
            return True

        except Exception as e:
            logger.error("index_create_failed", index=index, error=str(e))
            return False

    async def delete_index(self, index: str) -> bool:
        """删除索引

        Args:
            index: 索引名称

        Returns:
            是否成功
        """
        try:
            resolved_index = self._resolve_index(index)
            await self.client.indices.delete(index=resolved_index)
            logger.info("index_deleted", index=resolved_index)
            return True

        except Exception as e:
            logger.error("index_delete_failed", index=index, error=str(e))
            return False

    async def refresh_index(self, index: str) -> bool:
        """刷新索引（使文档立即可搜索）

        Args:
            index: 索引名称

        Returns:
            是否成功
        """
        try:
            resolved_index = self._resolve_index(index)
            await self.client.indices.refresh(index=resolved_index)
            return True

        except Exception as e:
            logger.error("index_refresh_failed", index=index, error=str(e))
            return False

    async def get_index_mapping(self, index: str) -> dict[str, Any] | None:
        """获取索引映射

        Args:
            index: 索引名称

        Returns:
            映射字典
        """
        try:
            resolved_index = self._resolve_index(index)
            response = await self.client.indices.get_mapping(index=resolved_index)
            return response.get(resolved_index, {}).get("mappings")

        except Exception as e:
            logger.error("get_mapping_failed", index=index, error=str(e))
            return None

    async def put_index_mapping(
        self,
        index: str,
        mappings: dict[str, Any],
    ) -> bool:
        """更新索引映射

        Args:
            index: 索引名称
            mappings: 新的映射

        Returns:
            是否成功
        """
        try:
            resolved_index = self._resolve_index(index)
            await self.client.indices.put_mapping(
                index=resolved_index,
                body=mappings,
            )
            logger.info("mapping_updated", index=resolved_index)
            return True

        except Exception as e:
            logger.error("mapping_update_failed", index=index, error=str(e))
            return False

    # ============== 文档操作 ==============

    async def index_document(
        self,
        index: str,
        id: str | None,
        document: dict[str, Any],
        refresh: bool = False,
    ) -> str | None:
        """索引单个文档

        Args:
            index: 索引名称
            id: 文档 ID（None 自动生成）
            document: 文档内容
            refresh: 是否立即刷新

        Returns:
            文档 ID
        """
        try:
            resolved_index = self._resolve_index(index)
            response = await self.client.index(
                index=resolved_index,
                id=id,
                document=document,
                refresh=refresh,
            )

            doc_id = response.get("_id")
            logger.debug("document_indexed", index=resolved_index, id=doc_id)
            return doc_id

        except Exception as e:
            logger.error("document_index_failed", index=index, id=id, error=str(e))
            return None

    async def bulk_index(
        self,
        actions: list[dict[str, Any]],
        refresh: bool = False,
    ) -> tuple[int, int]:
        """批量索引文档

        Args:
            actions: 操作列表，每个操作包含 _op_type, _index, _id, _source
            refresh: 是否立即刷新

        Returns:
            (成功数, 失败数)
        """
        try:
            # 添加索引前缀
            resolved_actions = []
            for action in actions:
                resolved_action = action.copy()
                resolved_action["_index"] = self._resolve_index(action["_index"])
                resolved_actions.append(resolved_action)

            success, failed = await async_bulk(
                self.client,
                resolved_actions,
                refresh=refresh,
                raise_on_error=False,
            )

            logger.info("bulk_index_completed", success=success, failed=len(failed))
            return success, len(failed)

        except Exception as e:
            logger.error("bulk_index_failed", error=str(e))
            return 0, len(actions)

    async def get_document(
        self,
        index: str,
        id: str,
    ) -> dict[str, Any] | None:
        """获取文档

        Args:
            index: 索引名称
            id: 文档 ID

        Returns:
            文档内容（包含 _source）
        """
        try:
            resolved_index = self._resolve_index(index)
            response = await self.client.get(index=resolved_index, id=id)
            return response.get("_source")

        except Exception as e:
            logger.debug("document_not_found", index=index, id=id, error=str(e))
            return None

    async def update_document(
        self,
        index: str,
        id: str,
        updates: dict[str, Any],
        refresh: bool = False,
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
        try:
            resolved_index = self._resolve_index(index)
            await self.client.update(
                index=resolved_index,
                id=id,
                doc=updates,
                refresh=refresh,
            )
            logger.debug("document_updated", index=resolved_index, id=id)
            return True

        except Exception as e:
            logger.error("document_update_failed", index=index, id=id, error=str(e))
            return False

    async def delete_document(
        self,
        index: str,
        id: str,
        refresh: bool = False,
    ) -> bool:
        """删除文档

        Args:
            index: 索引名称
            id: 文档 ID
            refresh: 是否立即刷新

        Returns:
            是否成功
        """
        try:
            resolved_index = self._resolve_index(index)
            await self.client.delete(index=resolved_index, id=id, refresh=refresh)
            logger.debug("document_deleted", index=resolved_index, id=id)
            return True

        except Exception as e:
            logger.error("document_delete_failed", index=index, id=id, error=str(e))
            return False

    async def delete_by_query(
        self,
        index: str,
        query: dict[str, Any],
        refresh: bool = False,
    ) -> int:
        """按查询删除文档

        Args:
            index: 索引名称
            query: 删除查询
            refresh: 是否立即刷新

        Returns:
            删除的文档数
        """
        try:
            resolved_index = self._resolve_index(index)
            response = await self.client.delete_by_query(
                index=resolved_index,
                query=query,
                refresh=refresh,
            )
            deleted = response.get("deleted", 0)
            logger.info("documents_deleted_by_query", index=resolved_index, count=deleted)
            return deleted

        except Exception as e:
            logger.error("delete_by_query_failed", index=index, error=str(e))
            return 0

    # ============== 搜索操作 ==============

    async def search(
        self,
        index: str,
        query: dict[str, Any],
        size: int = 10,
        from_: int = 0,
        sort: list[str] | dict[str, Any] | None = None,
        highlight: dict[str, Any] | None = None,
        source: bool | list[str] | dict[str, Any] | None = None,
        track_total_hits: bool | int = True,
    ) -> dict[str, Any]:
        """执行搜索

        Args:
            index: 索引名称
            query: 搜索查询 DSL
            size: 返回结果数
            from_: 偏移量
            sort: 排序
            highlight: 高亮配置
            source: 返回字段配置
            track_total_hits: 是否精确统计总数

        Returns:
            搜索结果
        """
        try:
            resolved_index = self._resolve_index(index)

            body = {"query": query}

            if sort:
                body["sort"] = sort
            if highlight:
                body["highlight"] = highlight
            if source is not None:
                body["_source"] = source

            response = await self.client.search(
                index=resolved_index,
                size=size,
                from_=from_,
                body=body,
                track_total_hits=track_total_hits,
            )

            logger.debug("search_executed", index=resolved_index, hits=len(response.get("hits", {}).get("hits", [])))
            return response

        except Exception as e:
            logger.error("search_failed", index=index, error=str(e))
            return {"hits": {"hits": [], "total": {"value": 0}}}

    async def multi_search(
        self,
        searches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """执行多搜索

        Args:
            searches: 搜索列表，每个包含 index 和 body

        Returns:
            搜索结果列表
        """
        try:
            resolved_searches = []
            for search in searches:
                resolved_search = {
                    "index": self._resolve_index(search["index"]),
                    "body": search.get("body", {}),
                }
                resolved_searches.append(resolved_search)

            response = await self.client.msearch(searches=resolved_searches)
            return response.get("responses", [])

        except Exception as e:
            logger.error("multi_search_failed", error=str(e))
            return []

    async def count(
        self,
        index: str,
        query: dict[str, Any] | None = None,
    ) -> int:
        """统计文档数

        Args:
            index: 索引名称
            query: 查询条件（None 统计全部）

        Returns:
            文档数量
        """
        try:
            resolved_index = self._resolve_index(index)
            if query:
                response = await self.client.count(index=resolved_index, query=query)
            else:
                response = await self.client.count(index=resolved_index)
            return response.get("count", 0)

        except Exception as e:
            logger.error("count_failed", index=index, error=str(e))
            return 0

    async def aggregate(
        self,
        index: str,
        aggs: dict[str, Any],
        query: dict[str, Any] | None = None,
        size: int = 0,
    ) -> dict[str, Any]:
        """执行聚合查询

        Args:
            index: 索引名称
            aggs: 聚合 DSL
            query: 过滤查询
            size: 返回文档数（0 表示不返回文档）

        Returns:
            聚合结果
        """
        try:
            resolved_index = self._resolve_index(index)

            body: dict[str, Any] = {"aggs": aggs, "size": size}
            if query:
                body["query"] = query

            response = await self.client.search(index=resolved_index, body=body)
            return response.get("aggregations", {})

        except Exception as e:
            logger.error("aggregate_failed", index=index, error=str(e))
            return {}

    async def scroll_search(
        self,
        index: str,
        query: dict[str, Any],
        scroll_time: str = "1m",
        size: int = 100,
    ) -> AsyncIterator[dict[str, Any]]:
        """滚动搜索（获取大量结果）

        Args:
            index: 索引名称
            query: 搜索查询
            scroll_time: 滚动窗口时间
            size: 每批大小

        Yields:
            文档
        """
        try:
            resolved_index = self._resolve_index(index)

            response = await self.client.search(
                index=resolved_index,
                query=query,
                scroll=scroll_time,
                size=size,
            )

            scroll_id = response.get("_scroll_id")
            hits = response.get("hits", {}).get("hits", [])

            while hits:
                for hit in hits:
                    yield hit

                # 获取下一批
                if scroll_id:
                    response = await self.client.scroll(
                        scroll_id=scroll_id,
                        scroll=scroll_time,
                    )
                    scroll_id = response.get("_scroll_id")
                    hits = response.get("hits", {}).get("hits", [])
                else:
                    break

            # 清理滚动上下文
            if scroll_id:
                await self.client.clear_scroll(scroll_id=scroll_id)

        except Exception as e:
            logger.error("scroll_search_failed", index=index, error=str(e))

    # ============== 健康检查 ==============

    async def ping(self) -> bool:
        """检查连接是否正常

        Returns:
            是否正常
        """
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def cluster_health(self) -> dict[str, Any]:
        """获取集群健康状态

        Returns:
            健康状态信息
        """
        try:
            return await self.client.cluster.health()
        except Exception as e:
            logger.error("cluster_health_failed", error=str(e))
            return {}

    async def cluster_stats(self) -> dict[str, Any]:
        """获取集群统计信息

        Returns:
            统计信息
        """
        try:
            return await self.client.cluster.stats()
        except Exception as e:
            logger.error("cluster_stats_failed", error=str(e))
            return {}


# ============== 全局客户端实例 ==============

_client: ElasticsearchClient | None = None


async def get_elasticsearch_client(
    hosts: str | list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
    index_prefix: str = "",
) -> ElasticsearchClient:
    """获取 Elasticsearch 客户端实例（单例）

    Args:
        hosts: Elasticsearch 主机地址
        username: 用户名
        password: 密码
        api_key: API Key
        index_prefix: 索引前缀

    Returns:
        ElasticsearchClient 实例
    """
    global _client

    if _client is None:
        # 从配置读取
        if hosts is None:
            from app.config.settings import get_settings

            settings = get_settings()
            hosts = getattr(settings, "elasticsearch_hosts", ["http://localhost:9200"])
            username = getattr(settings, "elasticsearch_username", None)
            password = getattr(settings, "elasticsearch_password", None)
            api_key = getattr(settings, "elasticsearch_api_key", None)
            index_prefix = getattr(settings, "elasticsearch_index_prefix", "")

        _client = ElasticsearchClient(
            hosts=hosts,
            username=username,
            password=password,
            api_key=api_key,
            index_prefix=index_prefix,
        )
        await _client.connect()

    return _client


async def close_elasticsearch_client() -> None:
    """关闭全局客户端"""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


# ============== 上下文管理器 ==============


@asynccontextmanager
async def elasticsearch_context(
    hosts: str | list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
    index_prefix: str = "",
) -> AsyncIterator[ElasticsearchClient]:
    """Elasticsearch 上下文管理器

    Args:
        hosts: Elasticsearch 主机地址
        username: 用户名
        password: 密码
        api_key: API Key
        index_prefix: 索引前缀

    Yields:
        ElasticsearchClient 实例
    """
    client = ElasticsearchClient(
        hosts=hosts,
        username=username,
        password=password,
        api_key=api_key,
        index_prefix=index_prefix,
    )
    try:
        await client.connect()
        yield client
    finally:
        await client.close()
