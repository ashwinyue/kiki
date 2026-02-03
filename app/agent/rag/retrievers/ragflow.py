"""RAGFlow 远程服务检索器

通过 API 调用 RAGFlow 远程检索服务。
参考 DeerFlow 的 RAGFlow 集成设计。

RAGFlow 是一个开源的 RAG 引擎，提供：
- 文档解析（PDF、Word、Markdown 等）
- 向量存储
- 语义检索
- 多语言支持

官网: https://ragflow.io
GitHub: https://github.com/infiniflow/ragflow
"""

from dataclasses import dataclass
from typing import Any

import httpx

from app.observability.logging import get_logger
from app.agent.rag.retrievers.base import (
    BaseRetriever,
    RetrievedDocument,
    RetrievalOptions,
    RetrievalError,
)

logger = get_logger(__name__)


@dataclass
class RAGFlowConfig:
    """RAGFlow 配置

    Attributes:
        api_url: RAGFlow API 地址（如 http://localhost:9388）
        api_key: API 密钥
        dataset_id: 数据集 ID（可选，不指定则检索所有数据集）
        timeout: 请求超时时间（秒）
    """
    api_url: str
    api_key: str
    dataset_id: str | None = None
    timeout: int = 30

    def __post_init__(self):
        """标准化 API URL"""
        # 移除尾部斜杠
        self.api_url = self.api_url.rstrip("/")


class RAGFlowRetriever(BaseRetriever):
    """RAGFlow 远程检索器

    通过 HTTP API 调用 RAGFlow 进行文档检索。

    Example:
        ```python
        # 创建检索器
        config = RAGFlowConfig(
            api_url="http://localhost:9388",
            api_key="ragflow-xxx",
            dataset_id="dataset-123",
        )
        retriever = RAGFlowRetriever(config)

        # 检索
        results = await retriever.retrieve("Python 异步编程")
        ```
    """

    def __init__(self, config: RAGFlowConfig):
        """初始化 RAGFlow 检索器

        Args:
            config: RAGFlow 配置
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "ragflow_retriever_initialized",
            api_url=config.api_url,
            dataset_id=config.dataset_id,
        )

    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    async def retrieve(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedDocument]:
        """检索相关文档

        Args:
            query: 查询文本
            options: 检索选项（可选）

        Returns:
            检索到的文档列表

        Raises:
            RetrievalError: 检索失败时抛出
        """
        if options is None:
            options = RetrievalOptions()

        client = self._get_client()

        try:
            # 构建请求体
            payload: dict[str, Any] = {
                "question": query,
                "top_k": options.top_k,
            }

            # 添加数据集过滤
            if self.config.dataset_id:
                payload["dataset_ids"] = [self.config.dataset_id]

            # 发送检索请求
            # 注意：RAGFlow API 端点可能因版本而异
            response = await client.post("/api/retrieval", json=payload)
            response.raise_for_status()

            data = response.json()

            # 解析响应
            # RAGFlow 响应格式可能如下：
            # {
            #   "code": 0,
            #   "data": [
            #     {
            #       "chunk_id": "xxx",
            #       "content": "文档内容",
            #       "doc_name": "文档名称",
            #       "similarity": 0.95,
            #       ...
            #     }
            #   ]
            # }
            if data.get("code") != 0:
                raise RetrievalError(
                    message=f"API 返回错误: {data.get('message', 'Unknown error')}",
                    retriever_type="RAGFlow",
                )

            chunks = data.get("data", [])

            # 转换为 RetrievedDocument
            retrieved_docs = []
            for chunk in chunks:
                # 应用分数阈值
                score = chunk.get("similarity", chunk.get("score", 0.0))
                if options.score_threshold is not None and score < options.score_threshold:
                    continue

                retrieved_docs.append(
                    RetrievedDocument(
                        title=chunk.get("doc_name", chunk.get("title", "Untitled")),
                        content=chunk.get("content", ""),
                        source=chunk.get("doc_id", chunk.get("source", "unknown")),
                        score=float(score),
                        metadata={
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                        },
                    )
                )

            logger.debug(
                "ragflow_retrieval_completed",
                query=query[:50],
                result_count=len(retrieved_docs),
            )

            return retrieved_docs

        except httpx.HTTPStatusError as e:
            logger.error(
                "ragflow_http_error",
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise RetrievalError(
                message=f"HTTP 错误: {e.response.status_code}",
                retriever_type="RAGFlow",
                cause=e,
            )
        except httpx.RequestError as e:
            logger.error(
                "ragflow_request_error",
                error=str(e),
            )
            raise RetrievalError(
                message=f"请求失败: {e}",
                retriever_type="RAGFlow",
                cause=e,
            )
        except Exception as e:
            logger.error(
                "ragflow_retrieval_failed",
                query=query[:50],
                error=str(e),
            )
            raise RetrievalError(
                message=f"检索失败: {e}",
                retriever_type="RAGFlow",
                cause=e,
            )

    def health_check(self) -> bool:
        """健康检查

        Returns:
            RAGFlow 服务是否可用
        """
        try:
            import asyncio

            async def check() -> bool:
                client = self._get_client()
                # 尝试获取服务状态
                response = await client.get("/api/health")
                return response.status_code == 200

            return asyncio.run(check())
        except Exception as e:
            logger.warning(
                "ragflow_health_check_failed",
                error=str(e),
            )
            return False

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


__all__ = [
    "RAGFlowConfig",
    "RAGFlowRetriever",
]
