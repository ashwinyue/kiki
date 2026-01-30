"""LangSmith 可观测性集成

提供 LLM 调用追踪、评估和数据集管理。
"""

from typing import Any, Callable
from contextlib import contextmanager
import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langsmith import Client

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)

settings = get_settings()


class LangSmithCallbackHandler(BaseCallbackHandler):
    """LangSmith 追踪回调处理器

    自动记录 LLM 调用、工具调用和 Agent 步骤。
    """

    def __init__(self, project_name: str | None = None):
        """初始化回调处理器

        Args:
            project_name: LangSmith 项目名称
        """
        super().__init__()
        self.project_name = project_name or settings.app_name
        self._enabled = self._check_enabled()

    def _check_enabled(self) -> bool:
        """检查 LangSmith 是否启用

        Returns:
            是否启用
        """
        # 检查环境变量
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            return True

        # 检查 API Key
        if os.getenv("LANGCHAIN_API_KEY"):
            return True

        return False

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """LLM 调用开始"""
        if not self._enabled:
            return

        logger.debug(
            "llm_start",
            model=serialized.get("name", "unknown"),
            prompt_count=len(prompts),
        )

    def on_llm_end(
        self,
        response: BaseMessage,
        **kwargs: Any,
    ) -> None:
        """LLM 调用结束"""
        if not self._enabled:
            return

        logger.debug(
            "llm_end",
            response_type=type(response).__name__,
        )

        # 记录 Token 使用
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            logger.info(
                "llm_tokens_used",
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """工具调用开始"""
        if not self._enabled:
            return

        logger.debug(
            "tool_start",
            tool=serialized.get("name", "unknown"),
        )

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """工具调用结束"""
        if not self._enabled:
            return

        logger.debug(
            "tool_end",
            output_length=len(output),
        )


class LangSmithClient:
    """LangSmith 客户端包装类

    提供数据集、运行和评估功能。
    """

    _client: Client | None = None

    @classmethod
    def get_client(cls) -> Client | None:
        """获取 LangSmith 客户端

        Returns:
            Client 实例或 None
        """
        if cls._client is None:
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if not api_key:
                logger.warning("langsmith_api_key_not_set")
                return None

            cls._client = Client(
                api_key=api_key,
                api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            )
            logger.info("langsmith_client_initialized")

        return cls._client

    @classmethod
    def create_dataset(
        cls,
        name: str,
        description: str | None = None,
    ) -> str | None:
        """创建数据集

        Args:
            name: 数据集名称
            description: 数据集描述

        Returns:
            数据集 ID
        """
        client = cls.get_client()
        if not client:
            return None

        try:
            dataset = client.create_dataset(
                dataset_name=name,
                description=description,
            )
            logger.info("dataset_created", name=name, dataset_id=dataset.id)
            return dataset.id
        except Exception as e:
            logger.error("dataset_creation_failed", name=name, error=str(e))
            return None

    @classmethod
    def create_example(
        cls,
        dataset_id: str,
        inputs: dict,
        outputs: dict | None = None,
    ) -> bool:
        """创建示例（添加到数据集）

        Args:
            dataset_id: 数据集 ID
            inputs: 输入数据
            outputs: 期望输出（可选）

        Returns:
            是否成功
        """
        client = cls.get_client()
        if not client:
            return False

        try:
            client.create_example(
                dataset_id=dataset_id,
                inputs=inputs,
                outputs=outputs,
            )
            logger.debug("example_created", dataset_id=dataset_id)
            return True
        except Exception as e:
            logger.error("example_creation_failed", dataset_id=dataset_id, error=str(e))
            return False

    @classmethod
    @contextmanager
    def trace_run(
        cls,
        project_name: str | None = None,
        run_name: str | None = None,
        metadata: dict | None = None,
    ):
        """追踪运行

        Args:
            project_name: 项目名称
            run_name: 运行名称
            metadata: 元数据

        Yields:
            追踪上下文

        Examples:
            ```python
            with LangSmithClient.trace_run(project_name="MyProject"):
                result = await agent.ainvoke(input_data)
            ```
        """
        from langsmith import traceable

        client = cls.get_client()
        if not client:
            # 如果客户端未初始化，返回空的上下文管理器
            @contextmanager
            def noop():
                yield
            return noop()

        project = project_name or settings.app_name

        @traceable(
            project_name=project,
            name=run_name,
            metadata=metadata or {},
        )
        async def _traced_wrapper(func, *args, **kwargs):
            return await func(*args, **kwargs)

        yield lambda f: _traced_wrapper(f)


# ============== 便捷函数 ==============

def get_langsmith_callbacks() -> list[BaseCallbackHandler]:
    """获取 LangSmith 回调处理器列表

    Returns:
        回调处理器列表
    """
    return [LangSmithCallbackHandler()]


def get_run_config(
    run_name: str | None = None,
    metadata: dict | None = None,
) -> RunnableConfig:
    """获取带追踪的运行配置

    Args:
        run_name: 运行名称
        metadata: 元数据

    Returns:
        RunnableConfig
    """
    config_metadata = {
        "project_name": settings.app_name,
    }
    if metadata:
        config_metadata.update(metadata)

    return RunnableConfig(
        metadata=config_metadata,
        run_name=run_name,
    )


# ============== 数据集管理器 ==============

class DatasetManager:
    """数据集管理器

    管理测试数据集和评估用例。
    """

    def __init__(self, dataset_name: str):
        """初始化数据集管理器

        Args:
            dataset_name: 数据集名称
        """
        self.dataset_name = dataset_name
        self.dataset_id: str | None = None

    def ensure_dataset(self) -> str:
        """确保数据集存在

        Returns:
            数据集 ID
        """
        if self.dataset_id:
            return self.dataset_id

        client = LangSmithClient.get_client()
        if not client:
            raise RuntimeError("LangSmith 客户端未初始化")

        # 尝试获取现有数据集
        try:
            datasets = client.list_datasets(dataset_name=self.dataset_name)
            if datasets:
                self.dataset_id = datasets[0].id
                return self.dataset_id
        except Exception:
            pass

        # 创建新数据集
        self.dataset_id = LangSmithClient.create_dataset(
            name=self.dataset_name,
            description=f"{self.dataset_name} 测试数据集",
        )

        if not self.dataset_id:
            raise RuntimeError("数据集创建失败")

        return self.dataset_id

    def add_example(
        self,
        input_text: str,
        expected_output: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """添加示例到数据集

        Args:
            input_text: 输入文本
            expected_output: 期望输出
            metadata: 元数据
        """
        dataset_id = self.ensure_dataset()

        inputs = {"input": input_text}
        outputs = None
        if expected_output:
            outputs = {"output": expected_output}

        LangSmithClient.create_example(
            dataset_id=dataset_id,
            inputs=inputs,
            outputs=outputs,
        )

        logger.info("example_added", dataset_id=dataset_id, input=input_text[:50])

    def add_examples_from_list(
        self,
        examples: list[dict[str, str]],
    ) -> int:
        """从列表批量添加示例

        Args:
            examples: 示例列表 [{"input": "...", "output": "..."}]

        Returns:
            添加的示例数量
        """
        count = 0
        for example in examples:
            self.add_example(
                input_text=example["input"],
                expected_output=example.get("output"),
                metadata=example.get("metadata"),
            )
            count += 1

        logger.info("examples_added", count=count, dataset=self.dataset_name)
        return count


# ============== 预定义数据集 ==============

def create_chatbot_dataset() -> DatasetManager:
    """创建聊天机器人测试数据集

    Returns:
        DatasetManager 实例
    """
    manager = DatasetManager("chatbot_eval")

    examples = [
        {"input": "你好", "output": "你好！有什么我可以帮助你的吗？"},
        {"input": "你会做什么？", "output": "我可以帮助你解答问题、提供信息和建议。"},
        {"input": "今天天气怎么样？", "output": None},  # 开放式答案
        {"input": "2+2等于几？", "output": "2+2等于4。"},
        {"input": "讲个笑话", "output": None},
    ]

    manager.add_examples_from_list(examples)
    return manager
