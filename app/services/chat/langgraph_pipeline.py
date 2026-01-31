"""ChatPipeline LangGraph 实现

对齐 WeKnora99 聊天管道架构，使用 LangGraph StateGraph 实现。

流程阶段:
1. query_rewrite: 查询重写
2. search: 知识库搜索
3. rerank: 重排序
4. context_build: 构建上下文
5. chat_completion: LLM 生成

依赖安装:
    uv add langgraph
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from app.observability.logging import get_logger
from app.services.chat.types import (
    ChatPipelineResult,
    History,
    SearchResult,
    SearchResultType,
)

logger = get_logger(__name__)


@dataclass
class ChatPipelineState:
    """聊天管道状态

    使用 TypedDict 风格的类型定义，符合 LangGraph 要求。
    """

    # 输入
    query: str = ""
    session_id: str = ""
    user_id: str | None = None
    tenant_id: int | None = None

    # 中间状态
    history: list[History] = field(default_factory=list)
    rewrite_query: str = ""
    search_results: list[SearchResult] = field(default_factory=list)
    rerank_results: list[SearchResult] = field(default_factory=list)
    context_str: str = ""
    answer: str = ""
    error: str | None = None

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LangGraphChatPipelineConfig:
    """LangGraph ChatPipeline 配置

    Attributes:
        system_prompt: 系统提示词
        enable_rewrite: 是否启用查询重写
        enable_search: 是否启用搜索
        enable_rerank: 是否启用重排序
        enable_web_search: 是否启用网络搜索
        max_history_rounds: 最大历史轮数
        max_context_length: 最大上下文长度
        rerank_top_k: 重排序返回数量
        knowledge_base_ids: 知识库 ID 列表
    """

    system_prompt: str = "你是一个有用的 AI 助手。"
    enable_rewrite: bool = True
    enable_search: bool = True
    enable_rerank: bool = True
    enable_web_search: bool = False
    max_history_rounds: int = 5
    max_context_length: int = 8000
    rerank_top_k: int = 5
    knowledge_base_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)


class LangGraphChatPipeline:
    """LangGraph 聊天管道

    使用 StateGraph 实现聊天管道，支持流式输出。

    使用示例:
        ```python
        from app.services.chat.langgraph_pipeline import (
            LangGraphChatPipeline,
            LangGraphChatPipelineConfig,
        )

        config = LangGraphChatPipelineConfig(
            session_id="session-123",
            system_prompt="你是一个有帮助的助手。",
        )

        pipeline = LangGraphChatPipeline(config=config)

        # 非流式执行
        result = await pipeline.run("什么是 Python?")
        print(result.answer)

        # 流式执行
        async for chunk in pipeline.run_stream("什么是 Python?"):
            print(chunk, end="")
        ```
    """

    def __init__(
        self,
        config: LangGraphChatPipelineConfig,
        llm_service: Any,  # LLMService
        knowledge_service: Any | None = None,  # KnowledgeService
        message_service: Any | None = None,  # MessageService
        web_search_service: Any | None = None,  # WebSearchService
        model_service: Any | None = None,  # ModelService
    ):
        """初始化聊天管道

        Args:
            config: 管道配置
            llm_service: LLM 服务
            knowledge_service: 知识库服务
            message_service: 消息服务
            web_search_service: 网络搜索服务
            model_service: 模型服务
        """
        self._config = config
        self._llm_service = llm_service
        self._knowledge_service = knowledge_service
        self._message_service = message_service
        self._web_search_service = web_search_service
        self._model_service = model_service

        # 构建图
        self._graph = self._build_graph()

        logger.info(
            "langgraph_chat_pipeline_initialized",
            session_id=config.session_id if hasattr(config, "session_id") else None,
        )

    def _build_graph(self) -> StateGraph:
        """构建状态图"""
        builder = StateGraph(ChatPipelineState)

        # 添加节点
        builder.add_node("query_rewrite", self._query_rewrite_node)
        builder.add_node("search", self._search_node)
        builder.add_node("rerank", self._rerank_node)
        builder.add_node("context_build", self._context_build_node)
        builder.add_node("chat_completion", self._chat_completion_node)

        # 设置入口
        builder.set_entry_point("query_rewrite")

        # 添加边和条件路由
        builder.add_conditional_edges(
            "query_rewrite",
            self._should_skip_search,
            {
                "continue": "search",
                "skip": "context_build",
            },
        )

        builder.add_conditional_edges(
            "search",
            self._should_skip_rerank,
            {
                "continue": "rerank",
                "skip": "context_build",
            },
        )

        builder.add_edge("rerank", "context_build")
        builder.add_edge("context_build", "chat_completion")
        builder.add_edge("chat_completion", END)

        return builder.compile()

    def _should_skip_search(self, state: ChatPipelineState) -> Literal["continue", "skip"]:
        """判断是否跳过搜索阶段"""
        if not self._config.enable_search:
            return "skip"
        return "continue"

    def _should_skip_rerank(self, state: ChatPipelineState) -> Literal["continue", "skip"]:
        """判断是否跳过重排序阶段"""
        if not self._config.enable_rerank or not state.search_results:
            return "skip"
        return "continue"

    async def _query_rewrite_node(
        self, state: ChatPipelineState
    ) -> ChatPipelineState:
        """查询重写节点"""
        query_rewrite = state.query

        if not self._config.enable_rewrite or not self._message_service:
            return {"rewrite_query": query_rewrite}

        # 加载对话历史
        history = await self._load_history(state.session_id)
        if not history:
            return {"rewrite_query": query_rewrite, "history": history}

        logger.info(
            "query_rewrite",
            session_id=state.session_id,
            history_rounds=len(history),
        )

        # 调用 LLM 重写
        rewrite_query = await self._call_rewrite_model(state.query, history)
        if rewrite_query:
            query_rewrite = rewrite_query

        return {"rewrite_query": query_rewrite, "history": history}

    async def _search_node(self, state: ChatPipelineState) -> ChatPipelineState:
        """搜索节点"""
        results: list[SearchResult] = []

        if not self._config.knowledge_base_ids and not self._config.enable_web_search:
            return {"search_results": results}

        query = state.rewrite_query or state.query

        # 知识库搜索
        if self._knowledge_service and self._config.knowledge_base_ids:
            for kb_id in self._config.knowledge_base_ids:
                kb_results = await self._knowledge_service.hybrid_search(
                    knowledge_base_id=kb_id,
                    query=query,
                )
                for r in kb_results:
                    results.append(
                        SearchResult(
                            id=r.get("id", ""),
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                            knowledge_id=r.get("knowledge_id"),
                            knowledge_title=r.get("knowledge_title"),
                            match_type=SearchResultType(r.get("match_type", "hybrid")),
                            metadata=r.get("metadata", {}),
                        )
                    )

        # 网络搜索
        if self._web_search_service and self._config.enable_web_search:
            web_results = await self._web_search_service.search(
                query=query,
                max_results=5,
            )
            for r in web_results:
                results.append(
                    SearchResult(
                        id=r.get("id", ""),
                        content=r.get("content", ""),
                        score=0.8,
                        knowledge_source="web_search",
                        match_type=SearchResultType.WEB_SEARCH,
                        metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                    )
                )

        # 去重
        results = self._deduplicate_results(results)

        logger.info(
            "search_completed",
            session_id=state.session_id,
            result_count=len(results),
        )

        return {"search_results": results}

    async def _rerank_node(self, state: ChatPipelineState) -> ChatPipelineState:
        """重排序节点"""
        search_results = state.search_results

        if not self._model_service or not self._config.rerank_model_id:
            return {
                "rerank_results": search_results[: self._config.rerank_top_k],
            }

        logger.info(
            "rerank",
            session_id=state.session_id,
            candidate_count=len(search_results),
        )

        # 调用 Rerank 模型
        passages = [r.content for r in search_results]
        rerank_results = await self._model_service.rerank(
            model_id=self._config.rerank_model_id,
            query=state.rewrite_query or state.query,
            documents=passages,
            top_k=self._config.rerank_top_k,
        )

        # 转换结果
        results: list[SearchResult] = []
        for rr in rerank_results:
            idx = rr.get("index")
            if idx is not None and 0 <= idx < len(search_results):
                result = search_results[idx]
                result.score = rr.get("relevance_score", 0.0)
                results.append(result)

        logger.info(
            "rerank_completed",
            session_id=state.session_id,
            result_count=len(results),
        )

        return {"rerank_results": results}

    async def _context_build_node(self, state: ChatPipelineState) -> ChatPipelineState:
        """上下文构建节点"""
        sources = state.rerank_results or state.search_results

        if not sources:
            return {"context_str": ""}

        context_parts = []
        for i, r in enumerate(sources):
            part = f"[{i + 1}] {r.content}"
            if r.knowledge_title:
                part += f"\n   来源: {r.knowledge_title}"
            context_parts.append(part)

        context_text = "\n\n".join(context_parts)
        max_length = self._config.max_context_length
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "..."

        return {"context_str": context_text}

    async def _chat_completion_node(
        self, state: ChatPipelineState
    ) -> ChatPipelineState:
        """LLM 生成节点"""
        query = state.query
        context = state.context_str
        history = state.history

        # 构建消息
        messages: list[BaseMessage] = [
            SystemMessage(content=self._config.system_prompt),
        ]

        # 添加历史
        for h in history:
            messages.append(HumanMessage(content=h.query))
            messages.append(HumanMessage(content=h.answer))

        # 构建当前用户消息
        if context:
            user_message = f"""参考信息：
{context}

请根据以上参考信息回答问题：{query}"""
        else:
            user_message = query

        messages.append(HumanMessage(content=user_message))

        # 调用 LLM
        llm = self._llm_service.get_llm_with_retry()
        if llm is None:
            return {"answer": "LLM 服务不可用", "error": "llm_not_available"}

        try:
            response = await llm.ainvoke(messages)
            answer = str(response.content)

            logger.info(
                "chat_completion_completed",
                session_id=state.session_id,
                answer_length=len(answer),
            )

            return {"answer": answer}

        except Exception as e:
            logger.error(
                "chat_completion_failed",
                session_id=state.session_id,
                error=str(e),
            )
            return {"answer": f"生成回答时出错: {str(e)}", "error": str(e)}

    async def _load_history(self, session_id: str) -> list[History]:
        """加载对话历史"""
        if not self._message_service:
            return []

        try:
            messages = await self._message_service.get_recent_messages(
                session_id=session_id,
                limit=self._config.max_history_rounds * 2,
            )

            history_map: dict[str, History] = {}
            for msg in messages:
                if msg.request_id not in history_map:
                    history_map[msg.request_id] = History(query="", answer="")

                if msg.role == "user":
                    history_map[msg.request_id].query = msg.content
                else:
                    history_map[msg.request_id].answer = msg.content

            complete_history = [h for h in history_map.values() if h.query and h.answer]
            return complete_history[-self._config.max_history_rounds :]

        except Exception as e:
            logger.warning("history_load_failed", session_id=session_id, error=str(e))
            return []

    async def _call_rewrite_model(
        self, query: str, history: list[History]
    ) -> str | None:
        """调用重写模型"""
        if not self._llm_service:
            return None

        try:
            from datetime import datetime

            conversation = "\n".join(
                f"用户: {h.query}\n助手: {h.answer}" for h in history
            )
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            prompt = f"""对话历史：
{conversation}

当前时间：{current_time}

用户当前的问题是：{query}

请根据对话历史，重写用户的问题，使其成为一个独立、完整的查询。"""

            response = await self._llm_service.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的查询重写助手。请直接输出重写后的查询，不要包含任何其他内容。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )

            content = response.get("content", "").strip()
            return content if content else None

        except Exception as e:
            logger.error("rewrite_model_call_failed", error=str(e))
            return None

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """去重搜索结果"""
        seen: set[str] = set()
        unique: list[SearchResult] = []

        for r in results:
            keys = [r.id]
            if r.parent_chunk_id:
                keys.append(f"parent:{r.parent_chunk_id}")

            if any(k in seen for k in keys):
                continue

            for k in keys:
                seen.add(k)
            unique.append(r)

        return unique

    async def run(
        self, query: str, session_id: str, **kwargs
    ) -> ChatPipelineResult:
        """运行聊天管道

        Args:
            query: 用户查询
            session_id: 会话 ID
            **kwargs: 其他参数

        Returns:
            ChatPipelineResult 处理结果
        """
        initial_state = ChatPipelineState(
            query=query,
            session_id=session_id,
            **kwargs,
        )

        config = RunnableConfig(
            configurable={"thread_id": session_id},
        )

        try:
            final_state = await self._graph.ainvoke(initial_state, config)

            return ChatPipelineResult(
                answer=final_state.get("answer", ""),
                sources=final_state.get("rerank_results") or final_state.get("search_results"),
                context=final_state.get("context_str", ""),
                metadata={
                    "query": query,
                    "rewrite_query": final_state.get("rewrite_query"),
                    "search_result_count": len(final_state.get("search_results", [])),
                    "rerank_result_count": len(final_state.get("rerank_results", [])),
                },
            )

        except Exception as e:
            logger.exception("chat_pipeline_failed", session_id=session_id, error=str(e))
            return ChatPipelineResult(
                answer=f"处理失败: {str(e)}",
                sources=[],
                context="",
                metadata={"error": str(e)},
            )

    async def run_stream(
        self, query: str, session_id: str, **kwargs
    ) -> AsyncGenerator[str]:
        """运行聊天管道（流式）

        Args:
            query: 用户查询
            session_id: 会话 ID
            **kwargs: 其他参数

        Yields:
            文本片段
        """
        initial_state = ChatPipelineState(
            query=query,
            session_id=session_id,
            **kwargs,
        )

        config = RunnableConfig(
            configurable={"thread_id": session_id},
        )

        try:
            async for chunk in self._graph.astream(
                initial_state, config, stream_mode="messages"
            ):
                if hasattr(chunk, "content") and chunk.content:
                    yield str(chunk.content)

        except Exception as e:
            logger.exception("chat_pipeline_stream_failed", session_id=session_id)
            yield f"\n\n[错误: {str(e)}]"


__all__ = [
    "LangGraphChatPipeline",
    "LangGraphChatPipelineConfig",
    "ChatPipelineState",
]
