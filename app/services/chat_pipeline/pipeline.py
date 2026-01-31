"""聊天管道主流程

对齐 WeKnora99 聊天管道架构，提供完整的 RAG 处理流程。
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

from app.observability.logging import get_logger
from app.services.chat_pipeline.stages import (
    ContextBuildStage,
    QueryRewriteStage,
    RerankStage,
    SearchStage,
)
from app.services.chat_pipeline.types import (
    ChatContext,
    ChatPipelineConfig,
    ChatPipelineResult,
    SearchResult,
)

logger = get_logger(__name__)


class PipelineError(Exception):
    """管道错误

    对齐 WeKnora99 PluginError
    """

    def __init__(self, error_type: str, description: str, original_error: Exception | None = None):
        self.error_type = error_type
        self.description = description
        self.original_error = original_error
        super().__init__(description)


class ChatPipeline:
    """聊天管道

    对齐 WeKnora99 ChatManage 完整流程。

    流程阶段:
    1. query_rewrite: 查询重写
    2. search: 知识库搜索
    3. rerank: 重排序
    4. build_context: 构建上下文
    5. chat_completion: LLM 生成

    使用示例:
        ```python
        from app.services.chat_pipeline import ChatPipeline, ChatPipelineConfig

        config = ChatPipelineConfig(
            tenant_id=1,
            session_id="session-123",
            query="什么是 Python?",
            knowledge_base_ids=["kb-1"],
            enable_search=True,
        )

        pipeline = ChatPipeline(
            config=config,
            llm_service=llm_service,
            knowledge_service=knowledge_service,
        )

        result = await pipeline.run()
        print(result.answer)
        ```
    """

    # 预定义错误
    ERR_SEARCH_NOTHING = PipelineError("search_nothing", "没有找到相关内容")
    ERR_RERANK_FAILED = PipelineError("rerank_failed", "重排序失败")
    ERR_MODEL_CALL = PipelineError("model_call_failed", "模型调用失败")

    def __init__(
        self,
        config: ChatPipelineConfig,
        llm_service: Any,  # LLMService
        knowledge_service: Any,  # KnowledgeService
        message_service: Any | None = None,  # MessageService
        web_search_service: Any | None = None,  # WebSearchService
        model_service: Any | None = None,  # ModelService
        callbacks: list[Any] | None = None,  # LangChain CallbackHandlers
    ):
        """初始化聊天管道

        Args:
            config: 管道配置
            llm_service: LLM 服务
            knowledge_service: 知识库服务
            message_service: 消息服务（用于查询重写）
            web_search_service: 网络搜索服务
            model_service: 模型服务（用于 Rerank）
            callbacks: LangChain CallbackHandlers（用于事件追踪）
        """
        self._config = config
        self._llm_service = llm_service
        self._knowledge_service = knowledge_service
        self._message_service = message_service
        self._web_search_service = web_search_service
        self._model_service = model_service
        self._callbacks = callbacks

        # 构建 RunnableConfig（用于传递 callbacks）
        self._runnable_config: dict[str, Any] = {}
        if callbacks:
            self._runnable_config["callbacks"] = callbacks

        # 初始化处理阶段
        self._rewrite_stage: QueryRewriteStage | None = None
        self._search_stage: SearchStage | None = None
        self._rerank_stage: RerankStage | None = None
        self._context_stage = ContextBuildStage()

        # 延迟初始化阶段（需要依赖时）
        if message_service:
            self._rewrite_stage = QueryRewriteStage(llm_service, message_service)

        if knowledge_service or web_search_service:
            self._search_stage = SearchStage(knowledge_service, web_search_service)

        if model_service:
            self._rerank_stage = RerankStage(model_service)

        logger.info(
            "chat_pipeline_initialized",
            tenant_id=config.tenant_id,
            session_id=config.session_id,
            enable_search=config.enable_search,
            enable_rewrite=config.enable_rewrite,
        )

    async def run(self, query: str) -> ChatPipelineResult:
        """运行聊天管道

        Args:
            query: 用户查询

        Returns:
            ChatPipelineResult 处理结果

        Raises:
            PipelineError: 管道处理错误
        """
        ctx = ChatContext(config=self._config, query=query)

        try:
            # 1. 查询重写
            if self._rewrite_stage:
                await self._rewrite_stage.process(ctx)

            # 2. 搜索
            if self._search_stage and self._config.enable_search:
                await self._search_stage.process(ctx)

            # 3. 重排序
            if self._rerank_stage:
                await self._rerank_stage.process(ctx)

            # 4. 构建上下文
            await self._context_stage.process(ctx)

            # 5. LLM 生成
            answer = await self._chat_completion(ctx)

            sources = ctx.rerank_results or ctx.search_results

            return ChatPipelineResult(
                answer=answer,
                sources=sources,
                context=ctx,
                metadata={
                    "query": query,
                    "rewrite_query": ctx.rewrite_query,
                    "search_result_count": len(ctx.search_results),
                    "rerank_result_count": len(ctx.rerank_results),
                },
            )

        except PipelineError:
            raise
        except Exception as e:
            logger.exception(
                "chat_pipeline_failed",
                session_id=self._config.session_id,
                error=str(e),
            )
            ctx.error = str(e)
            raise PipelineError("pipeline_failed", "聊天管道处理失败", e) from e

    async def run_stream(self, query: str) -> AsyncGenerator[str, None]:
        """运行聊天管道（流式）

        Args:
            query: 用户查询

        Yields:
            str: LLM 生成的文本片段
        """
        ctx = ChatContext(config=self._config, query=query)

        try:
            # 1-4. 非流式阶段
            if self._rewrite_stage:
                await self._rewrite_stage.process(ctx)

            if self._search_stage and self._config.enable_search:
                await self._search_stage.process(ctx)

            if self._rerank_stage:
                await self._rerank_stage.process(ctx)

            await self._context_stage.process(ctx)

            # 5. LLM 生成（流式）
            sources = ctx.rerank_results or ctx.search_results

            async for chunk in self._chat_completion_stream(ctx):
                yield chunk

        except Exception as e:
            logger.exception(
                "chat_pipeline_stream_failed",
                session_id=self._config.session_id,
                error=str(e),
            )
            yield f"\n\n[错误: {str(e)}]"

    async def _chat_completion(self, ctx: ChatContext) -> str:
        """LLM 生成（非流式）"""
        messages = self._build_messages(ctx)

        try:
            response = await self._llm_service.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                config=self._runnable_config if self._runnable_config else None,
            )

            answer = response.get("content", "")

            logger.info(
                "chat_completion_output",
                session_id=self._config.session_id,
                answer_length=len(answer),
            )

            return answer

        except Exception as e:
            logger.error(
                "chat_completion_failed",
                session_id=self._config.session_id,
                error=str(e),
            )
            raise PipelineError("model_call_failed", "LLM 调用失败", e) from e

    async def _chat_completion_stream(self, ctx: ChatContext) -> AsyncGenerator[str, None]:
        """LLM 生成（流式）"""
        messages = self._build_messages(ctx)

        try:
            async for chunk in self._llm_service.chat_stream(
                messages=messages,
                config=self._runnable_config if self._runnable_config else None,
            ):
                yield chunk

        except Exception as e:
            logger.error(
                "chat_completion_stream_failed",
                session_id=self._config.session_id,
                error=str(e),
            )
            raise

    def _build_messages(self, ctx: ChatContext) -> list[dict[str, str]]:
        """构建 LLM 消息"""
        messages = [
            {
                "role": "system",
                "content": self._config.system_prompt,
            }
        ]

        # 添加对话历史
        from app.services.chat_pipeline.stages import remove_thinking_tags

        for h in ctx.history:
            messages.append({"role": "user", "content": h.query})
            # 移除思考标签
            answer = remove_thinking_tags(h.answer)
            messages.append({"role": "assistant", "content": answer})

        # 添加当前轮次
        user_message = ctx.query

        # 如果有上下文，添加到用户消息
        if ctx.context_str:
            user_message = f"""参考信息：
{ctx.context_str}

请根据以上参考信息回答问题：{ctx.query}"""

        messages.append({"role": "user", "content": user_message})

        return messages


class ChatPipelineBuilder:
    """聊天管道构建器

    简化管道创建的辅助类。
    """

    def __init__(self):
        self._config: ChatPipelineConfig | None = None
        self._llm_service: Any | None = None
        self._knowledge_service: Any | None = None
        self._message_service: Any | None = None
        self._web_search_service: Any | None = None
        self._model_service: Any | None = None
        self._callbacks: list[Any] | None = None

    def config(self, config: ChatPipelineConfig) -> "ChatPipelineBuilder":
        """设置配置"""
        self._config = config
        return self

    def llm_service(self, service: Any) -> "ChatPipelineBuilder":
        """设置 LLM 服务"""
        self._llm_service = service
        return self

    def knowledge_service(self, service: Any) -> "ChatPipelineBuilder":
        """设置知识库服务"""
        self._knowledge_service = service
        return self

    def message_service(self, service: Any) -> "ChatPipelineBuilder":
        """设置消息服务"""
        self._message_service = service
        return self

    def web_search_service(self, service: Any) -> "ChatPipelineBuilder":
        """设置网络搜索服务"""
        self._web_search_service = service
        return self

    def model_service(self, service: Any) -> "ChatPipelineBuilder":
        """设置模型服务"""
        self._model_service = service
        return self

    def callbacks(self, callbacks: list[Any]) -> "ChatPipelineBuilder":
        """设置 CallbackHandlers"""
        self._callbacks = callbacks
        return self

    def build(self) -> ChatPipeline:
        """构建管道"""
        if not self._config:
            raise ValueError("配置未设置")
        if not self._llm_service:
            raise ValueError("LLM 服务未设置")

        return ChatPipeline(
            config=self._config,
            llm_service=self._llm_service,
            knowledge_service=self._knowledge_service,
            message_service=self._message_service,
            web_search_service=self._web_search_service,
            model_service=self._model_service,
            callbacks=self._callbacks,
        )


def create_pipeline(
    config: ChatPipelineConfig,
    llm_service: Any,
    knowledge_service: Any,
    **services: Any,
) -> ChatPipeline:
    """创建聊天管道的便捷函数

    Args:
        config: 管道配置
        llm_service: LLM 服务
        knowledge_service: 知识库服务
        **services: 其他可选服务
            - message_service: 消息服务
            - web_search_service: 网络搜索服务
            - model_service: 模型服务
            - callbacks: CallbackHandlers 列表

    Returns:
        ChatPipeline 实例

    Examples:
        ```python
        from app.agent.callbacks import KikiCallbackHandler

        handler = KikiCallbackHandler(session_id="session-123")
        pipeline = create_pipeline(
            config=config,
            llm_service=llm_service,
            knowledge_service=knowledge_service,
            callbacks=[handler],
        )
        ```
    """
    return ChatPipeline(
        config=config,
        llm_service=llm_service,
        knowledge_service=knowledge_service,
        message_service=services.get("message_service"),
        web_search_service=services.get("web_search_service"),
        model_service=services.get("model_service"),
        callbacks=services.get("callbacks"),
    )


__all__ = [
    "ChatPipeline",
    "ChatPipelineBuilder",
    "PipelineError",
    "create_pipeline",
]
