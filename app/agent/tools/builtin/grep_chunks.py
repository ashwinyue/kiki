"""全文搜索工具

对知识库内的 chunks 进行全文关键词搜索。
使用 PostgreSQL ILIKE 进行精确/模糊匹配。
对齐 WeKnora99 ToolGrepChunks。
"""

import re
from collections import defaultdict
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Chunk, Knowledge, KnowledgeBase
from app.observability.log_sanitizer import sanitize_log_input
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认配置
DEFAULT_MAX_RESULTS = 50
DEFAULT_MIN_SCORE = 0.0


class GrepChunksError(Exception):
    """全文搜索错误"""
    pass


async def _get_db_session() -> AsyncSession:
    """获取数据库会话

    Returns:
        AsyncSession 实例
    """
    from app.infra.database import async_session_factory
    return async_session_factory()


async def _search_chunks(
    session: AsyncSession,
    tenant_id: int,
    patterns: list[str],
    knowledge_base_ids: list[str] | None = None,
    knowledge_ids: list[str] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """搜索 chunks

    Args:
        session: 数据库会话
        tenant_id: 租户 ID
        patterns: 搜索关键词模式列表（OR 关系）
        knowledge_base_ids: 知识库 ID 列表
        knowledge_ids: 知识 ID 列表
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    # 构建 WHERE 条件
    conditions = [
        Chunk.tenant_id == tenant_id,
        Chunk.deleted_at.is_(None),
        Chunk.is_enabled == True,
    ]

    if knowledge_base_ids:
        conditions.append(Chunk.knowledge_base_id.in_(knowledge_base_ids))

    if knowledge_ids:
        conditions.append(Chunk.knowledge_id.in_(knowledge_ids))

    # 构建 ILIKE 模式（支持多个模式 OR 查询）
    if patterns:
        pattern_conditions = []
        for p in patterns:
            # 转义特殊字符
            escaped = p.replace("%", "\\%").replace("_", "\\_")
            pattern_conditions.append(
                Chunk.content.ilike(f"%{escaped}%")
            )
        conditions.append(func.coalesce(
            or_(*pattern_conditions),
            False
        ))

    # 构建查询
    stmt = (
        select(Chunk, Knowledge, KnowledgeBase)
        .join(Knowledge, Chunk.knowledge_id == Knowledge.id, isouter=True)
        .join(KnowledgeBase, Chunk.knowledge_base_id == KnowledgeBase.id, isouter=True)
        .where(*conditions)
        .order_by(Chunk.created_at.desc())
        .limit(max_results)
    )

    try:
        result = await session.execute(stmt)
        rows = result.all()
    except Exception as e:
        logger.error(
            "grep_chunks_query_failed",
            tenant_id=tenant_id,
            patterns=patterns,
            error=str(e),
        )
        raise GrepChunksError(f"搜索失败: {str(e)}")

    chunks = []
    for row, knowledge, kb in rows:
        # 计算匹配分数（简单计数）
        score = _calculate_match_score(row.content, patterns)

        # 找到匹配的片段
        matches = _find_matches(row.content, patterns)

        chunk_info = {
            "chunk_id": row.id,
            "content": row.content,
            "score": score,
            "chunk_index": row.chunk_index,
            "chunk_type": row.chunk_type,
            "knowledge_id": row.knowledge_id,
            "knowledge_title": knowledge.title if knowledge else None,
            "knowledge_base_id": row.knowledge_base_id,
            "knowledge_base_name": kb.name if kb else None,
            "file_name": knowledge.file_name if knowledge else None,
            "start_at": row.start_at,
            "end_at": row.end_at,
            "matches": matches,
        }
        chunks.append(chunk_info)

    return chunks


def _calculate_match_score(content: str, patterns: list[str]) -> float:
    """计算匹配分数

    基于匹配的关键词数量和位置计算分数。

    Args:
        content: chunk 内容
        patterns: 搜索模式

    Returns:
        匹配分数 (0.0 - 1.0)
    """
    if not patterns:
        return 0.0

    content_lower = content.lower()
    matched_count = 0

    for p in patterns:
        if p.lower() in content_lower:
            matched_count += 1
            # 匹配位置越靠前，分数越高
            pos = content_lower.find(p.lower())
            if pos == 0:
                matched_count += 0.5  # 开头匹配加分

    # 归一化分数
    score = min(1.0, matched_count / (len(patterns) * 1.5))
    return round(score, 3)


def _find_matches(content: str, patterns: list[str]) -> list[dict[str, Any]]:
    """找到匹配的片段

    Args:
        content: chunk 内容
        patterns: 搜索模式

    Returns:
        匹配片段列表
    """
    matches = []
    content_lower = content.lower()

    for p in patterns:
        pattern_lower = p.lower()
        start = 0
        while True:
            pos = content_lower.find(pattern_lower, start)
            if pos == -1:
                break

            # 提取周围上下文
            context_start = max(0, pos - 50)
            context_end = min(len(content), pos + len(p) + 50)
            context = content[context_start:context_end]

            matches.append({
                "pattern": p,
                "position": pos,
                "context": context,
            })

            start = pos + 1

    return matches


def _format_results(results: list[dict[str, Any]], query: str) -> str:
    """格式化搜索结果

    Args:
        results: 搜索结果列表
        query: 原始查询

    Returns:
        格式化的结果字符串
    """
    if not results:
        return f"未找到与 '{query}' 相关的内容"

    parts = [f"## 搜索结果 (共 {len(results)} 条)"]

    for i, r in enumerate(results, 1):
        parts.append(f"\n### [{i}] {r.get('knowledge_title', '未知')}")
        parts.append(f"**知识库**: {r.get('knowledge_base_name', '未知')}")

        if r.get('file_name'):
            parts.append(f"**文件**: {r['file_name']}")

        parts.append(f"**相关性**: {r['score']:.2%}")

        # 显示匹配片段
        for m in r.get('matches', [])[:3]:
            context = m.get('context', '')
            parts.append(f"...{context}...")

        # 显示内容摘要
        content = r.get('content', '')
        if len(content) > 300:
            content = content[:300] + "..."
        parts.append(f"\n内容预览: {content}")

    return "\n".join(parts)


def _escape_pattern(pattern: str) -> str:
    """转义搜索模式

    Args:
        pattern: 原始模式

    Returns:
        转义后的模式
    """
    # 移除潜在的 SQL 注入字符
    escaped = re.sub(r"[%_\\]", r"\\\1", pattern)
    return escaped


@tool
async def grep_chunks(
    patterns: list[str],
    knowledge_base_ids: list[str] | None = None,
    knowledge_ids: list[str] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    tenant_id: int | None = None,
) -> str:
    """全文搜索知识库 chunks

    在知识库内搜索包含指定关键词的文本块。
   支持多个关键词（OR 关系）和知识库/知识过滤。

    Args:
        patterns: 搜索关键词列表，如 ["Python", "LangGraph"]
        knowledge_base_ids: 限制搜索的知识库 ID 列表
        knowledge_ids: 限制搜索的知识 ID 列表
        max_results: 最大返回结果数 (默认 50)
        tenant_id: 租户 ID（自动从上下文获取）

    Returns:
        格式化的搜索结果

    Examples:
        ```python
        # 搜索包含 "Python" 或 "AI" 的 chunks
        result = await grep_chunks(["Python", "AI"])

        # 限制在特定知识库搜索
        result = await grep_chunks(
            ["API"],
            knowledge_base_ids=["kb-123"]
        )
        ```
    """
    # 参数验证
    if not patterns:
        return "错误: 请至少提供一个搜索关键词"

    if max_results < 1:
        max_results = DEFAULT_MAX_RESULTS
    elif max_results > 200:
        max_results = 200  # 限制最大结果数

    # 获取租户 ID（如果没有提供）
    if tenant_id is None:
        from app.middleware import get_tenant_id
        tenant_id = get_tenant_id()
        if tenant_id is None:
            return "错误: 无法获取租户 ID"

    logger.info(
        "grep_chunks_start",
        tenant_id=tenant_id,
        patterns=patterns,
        kb_count=len(knowledge_base_ids) if knowledge_base_ids else 0,
        max_results=max_results,
    )

    try:
        # 获取数据库会话
        session = await _get_db_session()

        try:
            # 执行搜索
            results = await _search_chunks(
                session=session,
                tenant_id=tenant_id,
                patterns=patterns,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_ids=knowledge_ids,
                max_results=max_results,
            )

            # 格式化结果
            query = " OR ".join(patterns)
            formatted = _format_results(results, query)

            logger.info(
                "grep_chunks_complete",
                tenant_id=tenant_id,
                query=query,
                result_count=len(results),
            )

            return formatted

        finally:
            await session.close()

    except GrepChunksError as e:
        logger.error("grep_chunks_failed", tenant_id=tenant_id, error=str(e))
        return f"搜索失败: {str(e)}"

    except Exception as e:
        logger.exception("grep_chunks_error", tenant_id=tenant_id, error=str(e))
        return f"搜索出错: {str(e)}"


__all__ = ["grep_chunks"]
