"""模型测试服务

对齐 WeKnora99 的模型测试功能：
- Embedding 模型测试
- Rerank 模型测试
- LLM 模型测试
- 多模态模型测试

参考:
- WeKnora99 internal/handler/initialization.go
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.schemas.initialization import (
    ModelTestResponse,
)

if TYPE_CHECKING:
    from app.repositories.model import ModelRepository

logger = get_logger(__name__)


# ============== 测试结果数据类 ==============


class TestStatus(str, Enum):
    """测试状态"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TestResult:
    """测试结果"""

    status: TestStatus
    message: str
    latency_ms: int = 0
    details: dict | None = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


# ============== 模型测试服务 ==============


class ModelTestService:
    """模型测试服务

    提供各类模型的测试功能：
    - Embedding 模型连接测试和维度检测
    - Rerank 模型功能测试
    - LLM 模型连接测试
    - 多模态模型测试
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self._model_repo: ModelRepository | None = None

    @property
    def model_repo(self) -> "ModelRepository":
        """延迟初始化模型仓储（避免循环导入）"""
        if self._model_repo is None:
            from app.repositories.model import ModelRepository

            self._model_repo = ModelRepository(self.session)
        return self._model_repo

    # ============== Embedding 测试 ==============

    async def test_embedding(
        self,
        source: str,
        model_name: str,
        base_url: str = "",
        api_key: str = "",
        dimension: int = 0,
        provider: str = "",
    ) -> ModelTestResponse:
        """测试 Embedding 模型

        对齐 WeKnora99 POST /initialization/embedding/test

        Args:
            source: 模型来源 (local, remote, openai, aliyun, zhipu)
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥
            dimension: 期望的向量维度
            provider: 服务商标识

        Returns:
            测试结果
        """
        start_time = time.time()

        logger.info(
            "embedding_test_start",
            source=source,
            model_name=model_name,
            provider=provider,
        )

        try:
            # 验证输入参数
            if not model_name:
                return ModelTestResponse(
                    available=False,
                    message="模型名称不能为空",
                    dimension=0,
                )

            # 根据来源处理
            if source == "local":
                return await self._test_local_embedding(model_name)
            elif source == "remote":
                return await self._test_remote_embedding(
                    model_name, base_url, api_key, provider, dimension
                )
            else:
                # 兼容旧接口
                return await self._test_remote_embedding(
                    model_name, base_url, api_key, provider, dimension
                )

        except httpx.TimeoutException:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.warning("embedding_test_timeout", latency_ms=latency_ms)
            return ModelTestResponse(
                available=False,
                message=f"连接超时 (超过 {latency_ms}ms)",
                dimension=0,
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("embedding_test_error", error=str(e), latency_ms=latency_ms)
            return ModelTestResponse(
                available=False,
                message=f"测试失败: {str(e)}",
                dimension=0,
            )

    async def _test_local_embedding(self, model_name: str) -> ModelTestResponse:
        """测试本地 Embedding 模型

        Args:
            model_name: 模型名称

        Returns:
            测试结果
        """
        # 检查 Ollama 服务
        import os

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 检查 Ollama 服务
                response = await client.get(f"{base_url}/api/tags")
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])

                # 检查模型是否存在
                model_exists = any(
                    m.get("name", "").startswith(model_name) for m in models
                )

                if not model_exists:
                    return ModelTestResponse(
                        available=False,
                        message=f"Ollama 服务正常，但模型 {model_name} 未安装",
                        dimension=0,
                    )

                return ModelTestResponse(
                    available=True,
                    message=f"Ollama 模型 {model_name} 可用",
                    dimension=0,  # Ollama embedding 维度需要查询
                )

        except httpx.HTTPError as e:
            return ModelTestResponse(
                available=False,
                message=f"Ollama 服务不可用: {str(e)}",
                dimension=0,
            )

    async def _test_remote_embedding(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        provider: str,
        expected_dimension: int,
    ) -> ModelTestResponse:
        """测试远程 Embedding 模型

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥
            provider: 服务商标识
            expected_dimension: 期望的向量维度

        Returns:
            测试结果
        """
        if not base_url:
            return ModelTestResponse(
                available=False,
                message="远程模型需要提供 base_url",
                dimension=0,
            )

        # 根据提供商调整配置
        embeddings = self._create_embeddings(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )

        if not embeddings:
            return ModelTestResponse(
                available=False,
                message=f"不支持的提供商: {provider}",
                dimension=0,
            )

        # 执行测试嵌入
        try:
            test_text = "这是一个测试文本，用于验证 Embedding 模型是否正常工作。"
            embedding = await embeddings.aembed_query(test_text)

            actual_dimension = len(embedding)

            # 验证维度
            if expected_dimension > 0 and actual_dimension != expected_dimension:
                return ModelTestResponse(
                    available=True,
                    message=f"模型可用，但向量维度不匹配: 期望 {expected_dimension}，实际 {actual_dimension}",
                    dimension=actual_dimension,
                )

            logger.info(
                "embedding_test_success",
                model_name=model_name,
                dimension=actual_dimension,
            )

            return ModelTestResponse(
                available=True,
                message=f"测试成功，向量维度: {actual_dimension}",
                dimension=actual_dimension,
            )

        except Exception as e:
            logger.error("remote_embedding_test_failed", error=str(e))
            return ModelTestResponse(
                available=False,
                message=f"远程调用失败: {str(e)}",
                dimension=0,
            )

    def _create_embeddings(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        provider: str,
    ) -> Embeddings | None:
        """创建 Embeddings 实例

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥
            provider: 服务商标识

        Returns:
            Embeddings 实例
        """
        # 根据提供商创建对应的 Embeddings
        if provider == "aliyun":
            # 阿里云 DashScope
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key or "",
                base_url=base_url,
            )
        elif provider == "zhipu":
            # 智谱
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key or "",
                base_url=base_url,
            )
        elif provider == "jina":
            # Jina AI
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key or "",
                base_url=base_url,
            )
        elif provider == "siliconflow":
            # 硅基流动
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key or "",
                base_url=base_url,
            )
        else:
            # 通用 OpenAI 兼容接口
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key or "",
                base_url=base_url,
            )

    # ============== Rerank 测试 ==============

    async def test_rerank(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        provider: str = "",
    ) -> ModelTestResponse:
        """测试 Rerank 模型

        对齐 WeKnora99 POST /initialization/rerank/check

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥
            provider: 服务商标识

        Returns:
            测试结果
        """
        start_time = time.time()

        logger.info(
            "rerank_test_start",
            model_name=model_name,
            provider=provider,
        )

        try:
            if not model_name or not base_url:
                return ModelTestResponse(
                    available=False,
                    message="模型名称和 Base URL 不能为空",
                )

            # 根据提供商测试不同的 Rerank API
            if provider == "jina" or model_name.startswith("jina"):
                return await self._test_jina_rerank(model_name, base_url, api_key)
            elif provider == "aliyun" or "qwen" in model_name.lower():
                return await self._test_aliyun_rerank(model_name, base_url, api_key)
            elif provider == "zhipu" or model_name.startswith("glm"):
                return await self._test_zhipu_rerank(model_name, base_url, api_key)
            else:
                # 通用 OpenAI 兼容测试
                return await self._test_generic_rerank(model_name, base_url, api_key)

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("rerank_test_error", error=str(e), latency_ms=latency_ms)
            return ModelTestResponse(
                available=False,
                message=f"测试失败: {str(e)}",
            )

    async def _test_jina_rerank(
        self, model_name: str, base_url: str, api_key: str
    ) -> ModelTestResponse:
        """测试 Jina Reranker

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥

        Returns:
            测试结果
        """
        url = base_url.rstrip("/") + "/rerank"

        request_data = {
            "model": model_name,
            "query": "测试查询",
            "documents": ["文档1", "文档2", "文档3"],
            "top_n": 3,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=request_data, headers=headers)
                response.raise_for_status()

                result_data = response.json()
                results = result_data.get("results", [])

                logger.info(
                    "jina_rerank_test_success",
                    model_name=model_name,
                    result_count=len(results),
                )

                return ModelTestResponse(
                    available=True,
                    message=f"Jina Reranker 可用，返回 {len(results)} 个结果",
                )

        except httpx.HTTPStatusError as e:
            logger.error("jina_rerank_http_error", status_code=e.response.status_code)
            return ModelTestResponse(
                available=False,
                message=f"Jina Reranker 请求失败: {e.response.status_code}",
            )

    async def _test_aliyun_rerank(
        self, model_name: str, base_url: str, api_key: str
    ) -> ModelTestResponse:
        """测试阿里云 DashScope Rerank

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥

        Returns:
            测试结果
        """
        # 阿里云 Rerank API 路径
        url = base_url.rstrip("/") + "/services/rerank/text-rerank/text-rerank"

        request_data = {
            "model": model_name,
            "query": "测试查询",
            "documents": ["文档1", "文档2", "文档3"],
            "top_n": 3,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=request_data, headers=headers)
                response.raise_for_status()

                result_data = response.json()
                output = result_data.get("output", {})
                results = output.get("results", [])

                logger.info(
                    "aliyun_rerank_test_success",
                    model_name=model_name,
                    result_count=len(results),
                )

                return ModelTestResponse(
                    available=True,
                    message=f"阿里云 Reranker 可用，返回 {len(results)} 个结果",
                )

        except httpx.HTTPStatusError as e:
            logger.error("aliyun_rerank_http_error", status_code=e.response.status_code)
            return ModelTestResponse(
                available=False,
                message=f"阿里云 Reranker 请求失败: {e.response.status_code}",
            )

    async def _test_zhipu_rerank(
        self, model_name: str, base_url: str, api_key: str
    ) -> ModelTestResponse:
        """测试智谱 Rerank

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥

        Returns:
            测试结果
        """
        url = base_url.rstrip("/") + "/rerank"

        request_data = {
            "model": model_name,
            "query": "测试查询",
            "documents": ["文档1", "文档2", "文档3"],
            "top_n": 3,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=request_data, headers=headers)
                response.raise_for_status()

                result_data = response.json()
                results = result_data.get("results", [])

                logger.info(
                    "zhipu_rerank_test_success",
                    model_name=model_name,
                    result_count=len(results),
                )

                return ModelTestResponse(
                    available=True,
                    message=f"智谱 Reranker 可用，返回 {len(results)} 个结果",
                )

        except httpx.HTTPStatusError as e:
            logger.error("zhipu_rerank_http_error", status_code=e.response.status_code)
            return ModelTestResponse(
                available=False,
                message=f"智谱 Reranker 请求失败: {e.response.status_code}",
            )

    async def _test_generic_rerank(
        self, model_name: str, base_url: str, api_key: str
    ) -> ModelTestResponse:
        """测试通用 Rerank API

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥

        Returns:
            测试结果
        """
        url = base_url.rstrip("/") + "/rerank"

        request_data = {
            "model": model_name,
            "query": "测试查询",
            "documents": ["文档1", "文档2", "文档3"],
            "top_n": 3,
        }

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=request_data, headers=headers)

                if response.status_code == 200:
                    result_data = response.json()
                    results = result_data.get("results", [])

                    logger.info(
                        "generic_rerank_test_success",
                        model_name=model_name,
                        result_count=len(results),
                    )

                    return ModelTestResponse(
                        available=True,
                        message=f"Reranker 可用，返回 {len(results)} 个结果",
                    )
                else:
                    return ModelTestResponse(
                        available=False,
                        message=f"请求失败: {response.status_code}",
                    )

        except Exception as e:
            logger.error("generic_rerank_test_error", error=str(e))
            return ModelTestResponse(
                available=False,
                message=f"测试失败: {str(e)}",
            )

    # ============== LLM 测试 ==============

    async def test_llm(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        provider: str = "",
    ) -> TestResult:
        """测试 LLM 模型

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥
            provider: 服务商标识

        Returns:
            测试结果
        """
        start_time = time.time()

        logger.info(
            "llm_test_start",
            model_name=model_name,
            provider=provider,
        )

        try:
            if not model_name:
                return TestResult(
                    status=TestStatus.FAILED,
                    message="模型名称不能为空",
                )

            if not base_url:
                return TestResult(
                    status=TestStatus.FAILED,
                    message="API 地址不能为空",
                )

            # 创建 LLM 实例
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key or "sk-test",  # 有些实现允许空 key
                base_url=base_url,
                timeout=30.0,
                max_retries=1,
            )

            # 执行简单测试
            from langchain_core.messages import HumanMessage

            messages = [HumanMessage(content="Hi")]
            response = await llm.ainvoke(messages)

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "llm_test_success",
                model_name=model_name,
                latency_ms=latency_ms,
            )

            return TestResult(
                status=TestStatus.SUCCESS,
                message=f"测试成功，响应: {str(response.content)[:50]}...",
                latency_ms=latency_ms,
                details={"response_length": len(str(response.content))},
            )

        except httpx.TimeoutException:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.warning("llm_test_timeout", latency_ms=latency_ms)
            return TestResult(
                status=TestStatus.TIMEOUT,
                message=f"连接超时 (超过 {latency_ms}ms)",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("llm_test_error", error=str(e), latency_ms=latency_ms)
            return TestResult(
                status=TestStatus.ERROR,
                message=f"测试失败: {str(e)}",
                latency_ms=latency_ms,
            )

    # ============== 多模态测试 ==============

    async def test_multimodal(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        image_base64: str | None = None,
    ) -> TestResult:
        """测试多模态模型

        Args:
            model_name: 模型名称
            base_url: API 地址
            api_key: API 密钥
            image_base64: 测试图片的 base64 编码

        Returns:
            测试结果
        """
        start_time = time.time()

        logger.info(
            "multimodal_test_start",
            model_name=model_name,
        )

        try:
            if not model_name:
                return TestResult(
                    status=TestStatus.FAILED,
                    message="模型名称不能为空",
                )

            if not base_url:
                return TestResult(
                    status=TestStatus.FAILED,
                    message="API 地址不能为空",
                )

            # 创建多模态 LLM 实例
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key or "sk-test",
                base_url=base_url,
                timeout=30.0,
                max_retries=1,
            )

            # 准备测试消息（文本 + 图片）
            from langchain_core.messages import HumanMessage

            # 如果没有提供图片，使用简单的文本测试
            if image_base64:
                content = [
                    {"type": "text", "text": "描述这张图片"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ]
            else:
                # 仅使用文本测试（多模态模型通常也支持纯文本）
                content = [
                    {"type": "text", "text": "你好"},
                ]

            messages = [HumanMessage(content=content)]
            response = await llm.ainvoke(messages)

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "multimodal_test_success",
                model_name=model_name,
                latency_ms=latency_ms,
            )

            return TestResult(
                status=TestStatus.SUCCESS,
                message="测试成功",
                latency_ms=latency_ms,
                details={
                    "response_length": len(str(response.content)),
                    "has_image": image_base64 is not None,
                },
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("multimodal_test_error", error=str(e), latency_ms=latency_ms)
            return TestResult(
                status=TestStatus.ERROR,
                message=f"测试失败: {str(e)}",
                latency_ms=latency_ms,
            )

    # ============== 批量测试 ==============

    async def test_model_by_id(
        self,
        model_id: str,
        tenant_id: int,
    ) -> TestResult:
        """根据模型 ID 测试模型

        Args:
            model_id: 模型 ID
            tenant_id: 租户 ID

        Returns:
            测试结果
        """
        model = await self.model_repo.get_by_tenant(model_id, tenant_id)

        if not model:
            return TestResult(
                status=TestStatus.FAILED,
                message=f"模型 {model_id} 不存在",
            )

        parameters = model.parameters or {}
        model_type = model.type
        source = model.source

        if model_type == "Embedding":
            response = await self.test_embedding(
                source=source,
                model_name=model.name,
                base_url=parameters.get("base_url", ""),
                api_key=parameters.get("api_key", ""),
                dimension=parameters.get("embedding_parameters", {}).get("dimension", 0),
                provider=parameters.get("provider", ""),
            )
            return TestResult(
                status=TestStatus.SUCCESS if response.available else TestStatus.FAILED,
                message=response.message,
                details={"dimension": response.dimension},
            )

        elif model_type == "Rerank":
            response = await self.test_rerank(
                model_name=model.name,
                base_url=parameters.get("base_url", ""),
                api_key=parameters.get("api_key", ""),
                provider=parameters.get("provider", ""),
            )
            return TestResult(
                status=TestStatus.SUCCESS if response.available else TestStatus.FAILED,
                message=response.message,
            )

        elif model_type in ("Chat", "KnowledgeQA", "VLLM"):
            return await self.test_llm(
                model_name=model.name,
                base_url=parameters.get("base_url", ""),
                api_key=parameters.get("api_key", ""),
                provider=parameters.get("provider", ""),
            )

        else:
            return TestResult(
                status=TestStatus.FAILED,
                message=f"不支持的模型类型: {model_type}",
            )


__all__ = [
    "TestStatus",
    "TestResult",
    "ModelTestService",
]
