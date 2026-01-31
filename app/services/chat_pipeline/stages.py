"""聊天管道处理阶段

对齐 WeKnora99 聊天管道各个处理阶段。
"""

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

from app.observability.logging import get_logger
from app.services.chat_pipeline.types import (
    ChatContext,
    ChatPipelineConfig,
    History,
    SearchResult,
    SearchResultType,
)

logger = get_logger(__name__)

# 移除思考过程的正则
THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def remove_thinking_tags(content: str) -> str:
    """移除思考标签

    对齐 WeKnora99 reg 变量
    """
    return THINKING_PATTERN.sub("", content)


def format_conversation_history(history: list[History]) -> str:
    """格式化对话历史为提示词

    对齐 WeKnora99 formatConversationHistory
    """
    if not history:
        return ""

    parts = []
    for h in history:
        answer = remove_thinking_tags(h.answer)
        parts.append(f"------BEGIN------\n用户的问题是：{h.query}\n助手的回答是：{answer}\n------END------")
    return "\n".join(parts)


# 默认提示词
DEFAULT_REWRITE_PROMPT_SYSTEM = """你是一个专业的查询重写助手。你的任务是根据对话历史，将用户当前的查询重写为一个更完整、更清晰的独立查询。

重写规则：
1. 如果当前查询已经足够清晰和完整，可以保持原样
2. 如果当前查询包含代词（如"它"、"这个"），需要根据历史对话替换为具体指代
3. 如果当前查询不完整，需要根据历史对话补充缺失信息
4. 保持查询简洁，不要添加解释性文字

请直接输出重写后的查询，不要包含任何其他内容。"""

DEFAULT_REWRITE_PROMPT_USER = """对话历史：
{{conversation}}

当前时间：{{current_time}}

用户当前的问题是：{{query}}

请根据对话历史，重写用户的问题，使其成为一个独立、完整的查询。"""


class QueryRewriteStage:
    """查询重写阶段

    对齐 WeKnora99 PluginRewrite

    使用 LLM 基于对话历史重写用户查询，提高搜索准确度。
    """

    def __init__(
        self,
        llm_service: Any,  # LLMService
        message_service: Any,  # MessageService
    ):
        self._llm_service = llm_service
        self._message_service = message_service

    async def process(self, ctx: ChatContext) -> None:
        """处理查询重写"""
        # 初始化重写查询为原始查询
        ctx.rewrite_query = ctx.query

        if not ctx.config.enable_rewrite:
            logger.info(
                "query_rewrite_skipped",
                session_id=ctx.config.session_id,
                reason="rewrite_disabled",
            )
            return

        logger.info(
            "query_rewrite_input",
            session_id=ctx.config.session_id,
            query=ctx.query,
        )

        # 获取对话历史
        history = await self._load_history(ctx)
        ctx.history = history

        if not history:
            logger.info(
                "query_rewrite_skipped",
                session_id=ctx.config.session_id,
                reason="empty_history",
            )
            return

        logger.info(
            "query_rewrite_history_ready",
            session_id=ctx.config.session_id,
            history_rounds=len(history),
        )

        # 调用 LLM 重写
        rewrite_query = await self._call_rewrite_model(ctx, history)
        if rewrite_query:
            ctx.rewrite_query = rewrite_query

        logger.info(
            "query_rewrite_output",
            session_id=ctx.config.session_id,
            original_query=ctx.query,
            rewrite_query=ctx.rewrite_query,
        )

    async def _load_history(self, ctx: ChatContext) -> list[History]:
        """加载对话历史"""
        try:
            messages = await self._message_service.get_recent_messages(
                session_id=ctx.config.session_id,
                limit=ctx.config.max_history_rounds * 2,
            )

            # 按请求 ID 分组
            history_map: dict[str, History] = {}
            for msg in messages:
                if msg.request_id not in history_map:
                    history_map[msg.request_id] = History(query="", answer="")

                if msg.role == "user":
                    history_map[msg.request_id].query = msg.content
                    history_map[msg.request_id].created_at = (
                        msg.created_at.isoformat() if msg.created_at else None
                    )
                else:
                    history_map[msg.request_id].answer = msg.content
                    if msg.knowledge_references:
                        history_map[msg.request_id].knowledge_references = [
                            SearchResult(
                                id=r.get("id", ""),
                                content=r.get("content", ""),
                                score=r.get("score", 0.0),
                                match_type=SearchResultType(r.get("match_type", "hybrid")),
                                metadata=r.get("metadata", {}),
                            )
                            for r in msg.knowledge_references
                        ]

            # 过滤完整对话并排序
            complete_history = [
                h for h in history_map.values() if h.query and h.answer
            ]
            complete_history.sort(
                key=lambda h: h.created_at or "", reverse=False
            )

            return complete_history[-ctx.config.max_history_rounds :]

        except Exception as e:
            logger.warning(
                "query_rewrite_history_load_failed",
                session_id=ctx.config.session_id,
                error=str(e),
            )
            return []

    async def _call_rewrite_model(
        self, ctx: ChatContext, history: list[History]
    ) -> str | None:
        """调用重写模型"""
        try:
            from datetime import datetime, timedelta

            # 准备提示词
            system_prompt = (
                ctx.config.rewrite_prompt_system or DEFAULT_REWRITE_PROMPT_SYSTEM
            )
            user_prompt_template = (
                ctx.config.rewrite_prompt_user or DEFAULT_REWRITE_PROMPT_USER
            )

            # 格式化对话历史
            conversation_text = format_conversation_history(history)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 替换占位符
            placeholders = {
                "{{conversation}}": conversation_text,
                "{{query}}": ctx.query,
                "{{current_time}}": current_time,
                "{{yesterday}}": (datetime.now() - timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
            }

            user_prompt = user_prompt_template
            for key, value in placeholders.items():
                user_prompt = user_prompt.replace(key, value)

            # 调用 LLM
            response = await self._llm_service.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )

            content = response.get("content", "").strip()
            if content:
                return content

        except Exception as e:
            logger.error(
                "query_rewrite_model_call_failed",
                session_id=ctx.config.session_id,
                error=str(e),
            )

        return None


class SearchStage:
    """搜索阶段

    对齐 WeKnora99 PluginSearch

    执行混合搜索：知识库 + 网络搜索
    """

    def __init__(
        self,
        knowledge_service: Any,  # KnowledgeService
        web_search_service: Any,  # WebSearchService
    ):
        self._knowledge_service = knowledge_service
        self._web_search_service = web_search_service

    async def process(self, ctx: ChatContext) -> None:
        """处理搜索"""
        has_targets = (
            ctx.config.knowledge_base_ids
            or ctx.config.knowledge_ids
            or ctx.config.tag_ids
        )

        if not has_targets and not ctx.config.enable_web_search:
            logger.info(
                "search_skipped",
                session_id=ctx.config.session_id,
                reason="no_targets_and_web_disabled",
            )
            return

        logger.info(
            "search_input",
            session_id=ctx.config.session_id,
            query=ctx.rewrite_query,
            kb_count=len(ctx.config.knowledge_base_ids),
            web_enabled=ctx.config.enable_web_search,
        )

        # 并行执行知识库搜索和网络搜索
        results: list[SearchResult] = []
        tasks: list[Callable] = []

        # 知识库搜索
        if ctx.config.knowledge_base_ids:
            tasks.append(self._search_knowledge_bases(ctx))

        # 网络搜索
        if ctx.config.enable_web_search:
            tasks.append(self._search_web(ctx))

        # 执行所有搜索
        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(
                        "search_failed",
                        session_id=ctx.config.session_id,
                        error=str(result),
                    )
                elif result:
                    results.extend(result)

        # 去重
        results = self._deduplicate_results(results)
        ctx.search_results = results

        logger.info(
            "search_output",
            session_id=ctx.config.session_id,
            result_count=len(results),
        )

    async def _search_knowledge_bases(
        self, ctx: ChatContext
    ) -> list[SearchResult]:
        """搜索知识库"""
        results: list[SearchResult] = []

        for kb_id in ctx.config.knowledge_base_ids:
            try:
                # 调用混合搜索
                kb_results = await self._knowledge_service.hybrid_search(
                    knowledge_base_id=kb_id,
                    query=ctx.rewrite_query,
                    vector_threshold=ctx.config.vector_threshold,
                    keyword_threshold=ctx.config.keyword_threshold,
                    top_k=ctx.config.embedding_top_k,
                    knowledge_ids=ctx.config.knowledge_ids,
                )

                for r in kb_results:
                    results.append(
                        SearchResult(
                            id=r.get("id", ""),
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                            knowledge_id=r.get("knowledge_id"),
                            knowledge_title=r.get("knowledge_title"),
                            knowledge_filename=r.get("knowledge_filename"),
                            knowledge_source=r.get("knowledge_source"),
                            chunk_index=r.get("chunk_index"),
                            match_type=SearchResultType(
                                r.get("match_type", "hybrid")
                            ),
                            chunk_type=r.get("chunk_type"),
                            parent_chunk_id=r.get("parent_chunk_id"),
                            start_at=r.get("start_at"),
                            end_at=r.get("end_at"),
                            metadata=r.get("metadata", {}),
                        )
                    )

            except Exception as e:
                logger.warning(
                    "kb_search_failed",
                    session_id=ctx.config.session_id,
                    kb_id=kb_id,
                    error=str(e),
                )

        return results

    async def _search_web(self, ctx: ChatContext) -> list[SearchResult]:
        """网络搜索"""
        try:
            web_results = await self._web_search_service.search(
                query=ctx.rewrite_query,
                max_results=5,
            )

            results = []
            for r in web_results:
                results.append(
                    SearchResult(
                        id=r.get("id", ""),
                        content=r.get("content", ""),
                        score=r.get("score", 0.8),
                        knowledge_source="web_search",
                        match_type=SearchResultType.WEB_SEARCH,
                        metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                    )
                )

            return results

        except Exception as e:
            logger.warning(
                "web_search_failed",
                session_id=ctx.config.session_id,
                error=str(e),
            )
            return []

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """去重搜索结果

        对齐 WeKnora99 removeDuplicateResults
        """
        seen: set[str] = set()
        content_sigs: dict[str, str] = {}
        unique: list[SearchResult] = []

        for r in results:
            # 检查 ID 去重
            keys = [r.id]
            if r.parent_chunk_id:
                keys.append(f"parent:{r.parent_chunk_id}")

            if any(k in seen for k in keys):
                continue

            # 内容签名去重
            sig = self._build_content_signature(r.content)
            if sig and sig in content_sigs:
                continue

            for k in keys:
                seen.add(k)
            if sig:
                content_sigs[sig] = r.id

            unique.append(r)

        return unique

    def _build_content_signature(self, content: str) -> str:
        """构建内容签名"""
        # 简单的签名策略：取前100个字符的哈希
        if len(content) < 50:
            return content[:50]
        # 取首尾各50字符
        return content[:50] + content[-50:]


class RerankStage:
    """重排序阶段

    对齐 WeKnora99 PluginRerank

    使用 Rerank 模型重新排序，并应用 MMR 增加多样性
    """

    def __init__(self, model_service: Any):  # ModelService
        self._model_service = model_service

    async def process(self, ctx: ChatContext) -> None:
        """处理重排序"""
        if not ctx.search_results:
            logger.info(
                "rerank_skipped",
                session_id=ctx.config.session_id,
                reason="empty_search_results",
            )
            return

        if not ctx.config.rerank_model_id:
            logger.info(
                "rerank_skipped",
                session_id=ctx.config.session_id,
                reason="no_rerank_model",
            )
            ctx.rerank_results = ctx.search_results[: ctx.config.rerank_top_k]
            return

        logger.info(
            "rerank_input",
            session_id=ctx.config.session_id,
            candidate_count=len(ctx.search_results),
            model=ctx.config.rerank_model_id,
        )

        # 调用 Rerank 模型
        reranked = await self._rerank(ctx)
        if reranked:
            ctx.rerank_results = reranked
        else:
            ctx.rerank_results = ctx.search_results[: ctx.config.rerank_top_k]

        logger.info(
            "rerank_output",
            session_id=ctx.config.session_id,
            result_count=len(ctx.rerank_results),
        )

    async def _rerank(self, ctx: ChatContext) -> list[SearchResult] | None:
        """执行重排序"""
        try:
            # 准备 passages
            passages = [r.content for r in ctx.search_results]

            # 调用 Rerank
            rerank_results = await self._model_service.rerank(
                model_id=ctx.config.rerank_model_id,
                query=ctx.rewrite_query,
                documents=passages,
                top_k=ctx.config.rerank_top_k,
            )

            # 更新分数
            reranked: list[SearchResult] = []
            for rr in rerank_results:
                idx = rr.get("index")
                if idx is not None and 0 <= idx < len(ctx.search_results):
                    result = ctx.search_results[idx]
                    # 计算组合分数
                    model_score = rr.get("relevance_score", 0.0)
                    base_score = result.score
                    result.score = 0.6 * model_score + 0.4 * base_score
                    reranked.append(result)

            # 应用 MMR
            if reranked:
                reranked = self._apply_mmr(
                    reranked,
                    ctx.config.rerank_top_k,
                    lambda_param=0.7,
                )

            return reranked

        except Exception as e:
            logger.error(
                "rerank_failed",
                session_id=ctx.config.session_id,
                error=str(e),
            )
            return None

    def _apply_mmr(
        self, results: list[SearchResult], k: int, lambda_param: float = 0.7
    ) -> list[SearchResult]:
        """应用 MMR 算法增加多样性

        对齐 WeKnora99 applyMMR
        """
        if k <= 0 or not results:
            return []

        selected: list[SearchResult] = []
        selected_indices: set[int] = set()

        # 预计算 token sets
        token_sets = [self._tokenize(r.content) for r in results]

        while len(selected) < k and len(selected_indices) < len(results):
            best_idx = -1
            best_score = -1.0

            for i, r in enumerate(results):
                if i in selected_indices:
                    continue

                relevance = r.score
                redundancy = 0.0

                # 计算与已选结果的最大相似度
                for sel_idx in selected_indices:
                    sim = self._jaccard(token_sets[i], token_sets[sel_idx])
                    redundancy = max(redundancy, sim)

                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx < 0:
                break

            selected.append(results[best_idx])
            selected_indices.add(best_idx)

        return selected

    def _tokenize(self, text: str) -> set[str]:
        """简单的分词"""
        # 简单的按空格和标点分词
        words = re.findall(r"\w+", text.lower())
        return set(words)

    def _jaccard(self, set1: set[str], set2: set[str]) -> float:
        """计算 Jaccard 相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class ContextBuildStage:
    """上下文构建阶段

    对齐 WeKnora99 模板渲染阶段

    将搜索结果格式化为 LLM 上下文
    """

    async def process(self, ctx: ChatContext) -> None:
        """处理上下文构建"""
        sources = ctx.rerank_results or ctx.search_results

        if not sources:
            ctx.context_str = ""
            logger.info(
                "context_build_empty",
                session_id=ctx.config.session_id,
            )
            return

        # 格式化上下文
        context_parts = []
        for i, r in enumerate(sources):
            part = f"[{i + 1}] {r.content}"
            if r.knowledge_title:
                part += f"\n   来源: {r.knowledge_title}"
            if r.knowledge_filename:
                part += f" ({r.knowledge_filename})"
            context_parts.append(part)

        # 限制长度
        context_text = "\n\n".join(context_parts)
        max_length = ctx.config.max_context_length
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "..."

        ctx.context_str = context_text

        logger.info(
            "context_build_output",
            session_id=ctx.config.session_id,
            context_length=len(ctx.context_str),
            source_count=len(sources),
        )


__all__ = [
    "QueryRewriteStage",
    "SearchStage",
    "RerankStage",
    "ContextBuildStage",
    "format_conversation_history",
    "remove_thinking_tags",
]
