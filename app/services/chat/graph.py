"""基于 LangGraph 的聊天管道

使用 LangGraph StateGraph 构建聊天管道，提供：
- 可视化工作流
- 内置检查点和状态管理
- 更灵活的条件分支
- 与现有 Pipeline API 兼容
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.observability.logging import get_logger
from app.services.chat.stages import (
    ContextBuildStage,
    QueryRewriteStage,
    RerankStage,
    SearchStage,
)
from app.services.chat.types import (
    ChatContext,
    ChatPipelineConfig,
    ChatPipelineResult,
)

logger = get_logger(__name__)


class PipelineState(dict):
    """管道状态

    兼容 LangGraph State，使用 TypedDict 方式定义。

    Attributes:
        config: 管道配置
        query: 原始查询
        rewrite_query: 重写后的查询
        history: 对话历史
        search_results: 搜索结果
        rerank_results: 重排序结果
        context_str: 构建的上下文
        messages: LLM 消息
        error: 错误信息
        stage: 当前阶段
    """

    config: ChatPipelineConfig
    query: str
    rewrite_query: str | None
    history: list[Any]
    search_results: list[Any]
    rerank_results: list[Any]
    context_str: str
    messages: list[Any]
    error: str | None
    stage: str


class ChatPipelineGraph:
    """基于 LangGraph 的聊天管道

    使用 StateGraph 构建可观测的聊天流程。

    流程图:
        START -> query_rewrite -> search -> rerank -> context_build -> generate -> END

    特性:
    - 可视化工作流（通过 mermaid）
    - 内置状态管理
    - 支持中间结果检查
    - 可扩展的阶段添加
    """

    def __init__(
        self,
        llm_service: Any,
        knowledge_service: Any,
        message_service: Any | None = None,
        web_search_service: Any | None = None,
        model_service: Any | None = None,
    ):
        """初始化聊天管道图

        Args:
            llm_service: LLM 服务
            knowledge_service: 知识库服务
            message_service: 消息服务（用于查询重写）
            web_search_service: 网络搜索服务
            model_service: 模型服务（用于 Rerank）
        """
        self._llm_service = llm_service
        self._knowledge_service = knowledge_service
        self._message_service = message_service
        self._web_search_service = web_search_service
        self._model_service = model_service

        # 初始化阶段（复用现有实现）
        self._rewrite_stage: QueryRewriteStage | None = None
        self._search_stage: SearchStage | None = None
        self._rerank_stage: RerankStage | None = None
        self._context_stage = ContextBuildStage()

        if message_service:
            self._rewrite_stage = QueryRewriteStage(llm_service, message_service)

        if knowledge_service or web_search_service:
            self._search_stage = SearchStage(knowledge_service, web_search_service)

        if model_service:
            self._rerank_stage = RerankStage(model_service)

        # 编译图
        self._graph = self._build_graph()

        logger.info("chat_pipeline_graph_initialized")

    def _build_graph(self) -> CompiledStateGraph:
        """构建 StateGraph

        Returns:
            编译后的图
        """
        # 创建 StateGraph
        builder = StateGraph(dict)  # 使用 dict 作为状态类型

        # 添加节点
        builder.add_node("query_rewrite", self._query_rewrite_node)
        builder.add_node("search", self._search_node)
        builder.add_node("rerank", self._rerank_node)
        builder.add_node("context_build", self._context_build_node)
        builder.add_node("generate", self._generate_node)

        # 设置入口点
        builder.set_entry_point("query_rewrite")

        # 添加边（线性流程）
        builder.add_edge("query_rewrite", "search")
        builder.add_edge("search", "rerank")
        builder.add_edge("rerank", "context_build")
        builder.add_edge("context_build", "generate")
        builder.add_edge("generate", END)

        # 编译（使用 MemorySaver 作为默认检查点）
        return builder.compile(checkpointer=MemorySaver())

    async def _query_rewrite_node(self, state: PipelineState) -> dict:
        """查询重写节点"""
        config = state["config"]
        query = state["query"]

        # 创建 ChatContext
        ctx = ChatContext(config=config, query=query)

        if self._rewrite_stage:
            await self._rewrite_stage.process(ctx)

        return {"rewrite_query": ctx.rewrite_query, "history": ctx.history, "stage": "query_rewrite"}

    async def _search_node(self, state: PipelineState) -> dict:
        """搜索节点"""
        config = state["config"]
        query = state["rewrite_query"] or state["query"]

        # 创建 ChatContext
        ctx = ChatContext(config=config, query=query)
        ctx.rewrite_query = state.get("rewrite_query")
        ctx.history = state.get("history", [])

        if self._search_stage and config.enable_search:
            await self._search_stage.process(ctx)

        return {"search_results": ctx.search_results, "stage": "search"}

    async def _rerank_node(self, state: PipelineState) -> dict:
        """重排序节点"""
        config = state["config"]

        # 创建 ChatContext
        ctx = ChatContext(config=config, query=state["query"])
        ctx.rewrite_query = state.get("rewrite_query")
        ctx.search_results = state.get("search_results", [])

        if self._rerank_stage:
            await self._rerank_stage.process(ctx)

        return {"rerank_results": ctx.rerank_results, "stage": "rerank"}

    async def _context_build_node(self, state: PipelineState) -> dict:
        """上下文构建节点"""
        config = state["config"]

        # 创建 ChatContext
        ctx = ChatContext(config=config, query=state["query"])
        ctx.search_results = state.get("search_results", [])
        ctx.rerank_results = state.get("rerank_results", [])

        await self._context_stage.process(ctx)

        return {"context_str": ctx.context_str, "stage": "context_build"}

    async def _generate_node(self, state: PipelineState) -> dict:
        """LLM 生成节点"""
        config = state["config"]

        # 构建消息
        messages = [
            SystemMessage(content=config.system_prompt),
        ]

        # 添加对话历史
        for h in state.get("history", []):
            messages.append(HumanMessage(content=h.query))
            messages.append(AIMessage(content=h.answer))

        # 添加当前轮次
        user_message = state["query"]
        context_str = state.get("context_str", "")
        if context_str:
            user_message = f"""参考信息：
{context_str}

请根据以上参考信息回答问题：{state["query"]}"""

        messages.append(HumanMessage(content=user_message))

        # 调用 LLM
        try:
            response = await self._llm_service.chat(
                messages=[{"role": m.type, "content": m.content} for m in messages],
                temperature=0.7,
                max_tokens=2000,
            )
            answer = response.get("content", "")
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            answer = f"抱歉，处理您的请求时出错：{str(e)}"

        return {"messages": messages, "answer": answer, "stage": "generate"}

    async def run(
        self,
        config: ChatPipelineConfig,
        query: str,
    ) -> ChatPipelineResult:
        """运行管道

        Args:
            config: 管道配置
            query: 用户查询

        Returns:
            ChatPipelineResult
        """
        # 初始状态
        initial_state: PipelineState = {
            "config": config,
            "query": query,
            "rewrite_query": None,
            "history": [],
            "search_results": [],
            "rerank_results": [],
            "context_str": "",
            "messages": [],
            "error": None,
            "stage": "start",
        }

        # 运行图
        result = await self._graph.ainvoke(
            initial_state,
            {"configurable": {"thread_id": config.session_id}},
        )

        # 构建结果
        sources = result.get("rerank_results") or result.get("search_results", [])

        return ChatPipelineResult(
            answer=result.get("answer", ""),
            sources=sources,
            context=None,  # 可选：创建 ChatContext
            metadata={
                "query": query,
                "rewrite_query": result.get("rewrite_query"),
                "search_result_count": len(result.get("search_results", [])),
                "rerank_result_count": len(result.get("rerank_results", [])),
                "stages": ["query_rewrite", "search", "rerank", "context_build", "generate"],
            },
        )

    async def run_stream(
        self,
        config: ChatPipelineConfig,
        query: str,
    ) -> AsyncIterator[str]:
        """运行管道（流式）

        Args:
            config: 管道配置
            query: 用户查询

        Yields:
            LLM 生成的文本片段
        """
        # 初始状态
        initial_state: PipelineState = {
            "config": config,
            "query": query,
            "rewrite_query": None,
            "history": [],
            "search_results": [],
            "rerank_results": [],
            "context_str": "",
            "messages": [],
            "error": None,
            "stage": "start",
        }

        # 流式运行图（仅在 generate 节点流式输出）
        async for chunk in self._graph.astream(
            initial_state,
            {"configurable": {"thread_id": config.session_id}},
            stream_mode="updates",
        ):
            # 检查是否是 generate 节点的输出
            if isinstance(chunk, dict) and "answer" in chunk:
                yield chunk["answer"]

    def get_graph_mermaid(self) -> str:
        """获取工作流的 Mermaid 图

        Returns:
            Mermaid 格式的图表示
        """
        return self._graph.get_graph().print_mermaid()

    def print_graph(self) -> None:
        """打印工作流结构"""
        print(self.get_graph_mermaid())


def create_pipeline_graph(
    llm_service: Any,
    knowledge_service: Any,
    **services: Any,
) -> ChatPipelineGraph:
    """创建聊天管道图

    Args:
        llm_service: LLM 服务
        knowledge_service: 知识库服务
        **services: 其他可选服务
            - message_service: 消息服务
            - web_search_service: 网络搜索服务
            - model_service: 模型服务

    Returns:
        ChatPipelineGraph 实例

    Examples:
        ```python
        graph = create_pipeline_graph(
            llm_service=llm_service,
            knowledge_service=knowledge_service,
            message_service=message_service,
        )

        result = await graph.run(config, "什么是 Python?")
        ```
    """
    return ChatPipelineGraph(
        llm_service=llm_service,
        knowledge_service=knowledge_service,
        message_service=services.get("message_service"),
        web_search_service=services.get("web_search_service"),
        model_service=services.get("model_service"),
    )


__all__ = [
    "ChatPipelineGraph",
    "PipelineState",
    "create_pipeline_graph",
]
