"""列出知识库 Chunks 工具

对 WeKnora99 ToolListKnowledgeChunks 的对齐实现。

列出指定知识库或知识条目下的所有 chunks。
"""

from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认配置
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class ListChunksError(Exception):
    """列出 chunks 错误"""
    pass


async def _get_db_session():
    """获取数据库会话

    Returns:
        AsyncSession 实例
    """
    from app.infra.database import async_session_factory
    return async_session_factory()


async def _list_chunks(
    tenant_id: int,
    knowledge_id: str | None = None,
    knowledge_base_id: str | None = None,
    page: int = DEFAULT_PAGE,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict[str, Any]]:
    """列出 chunks

    Args:
        tenant_id: 租户 ID
        knowledge_id: 知识条目 ID
        knowledge_base_id: 知识库 ID
        page: 页码
        page_size: 每页数量

    Returns:
        chunks 列表
    """
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models.knowledge import Chunk, Knowledge, KnowledgeBase

    session = await _get_db_session()

    try:
        # 构建查询
        conditions = [
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
            Chunk.is_enabled == True,
        ]

        if knowledge_id:
            conditions.append(Chunk.knowledge_id == knowledge_id)

        if knowledge_base_id:
            conditions.append(Chunk.knowledge_base_id == knowledge_base_id)

        # 计算偏移
        offset = (page - 1) * page_size

        # 查询 chunks
        stmt = (
            select(Chunk, Knowledge, KnowledgeBase)
            .join(Knowledge, Chunk.knowledge_id == Knowledge.id, isouter=True)
            .join(KnowledgeBase, Chunk.knowledge_base_id == KnowledgeBase.id, isouter=True)
            .where(*conditions)
            .order_by(Chunk.chunk_index)
            .offset(offset)
            .limit(page_size)
        )

        result = await session.execute(stmt)
        rows = result.all()

        chunks = []
        for chunk, knowledge, kb in rows:
            chunks.append({
                "id": chunk.id,
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "content_length": len(chunk.content),
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "knowledge_id": chunk.knowledge_id,
                "knowledge_title": knowledge.title if knowledge else None,
                "knowledge_base_id": chunk.knowledge_base_id,
                "knowledge_base_name": kb.name if kb else None,
                "start_at": chunk.start_at,
                "end_at": chunk.end_at,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            })

        return chunks

    finally:
        await session.close()


def _format_chunks_list(
    chunks: list[dict[str, Any]],
    query: str,
    page: int,
    page_size: int,
    total: int,
) -> str:
    """格式化 chunks 列表

    Args:
        chunks: chunks 列表
        query: 查询描述
        page: 页码
        page_size: 每页数量
        total: 总数

    Returns:
        格式化的结果字符串
    """
    if not chunks:
        return f"未找到 chunks\n\n**查询**: {query}"

    parts = [
        f"## Chunks 列表",
        f"**查询**: {query}",
        f"**页码**: {page}/{ (total + page_size - 1) // page_size }",
        f"**总数**: {total}",
        f"**当前页**: {len(chunks)} 条",
        "",
    ]

    for i, chunk in enumerate(chunks, 1):
        parts.append(f"### {i}. Chunk [{chunk['id'][:8]}...]")
        parts.append(f"**知识库**: {chunk.get('knowledge_base_name', 'Unknown')}")
        parts.append(f"**知识**: {chunk.get('knowledge_title', 'Unknown')}")
        parts.append(f"**类型**: {chunk.get('chunk_type', 'text')}")
        parts.append(f"**索引**: {chunk.get('chunk_index', 0)}")
        parts.append(f"**长度**: {chunk.get('content_length', 0)} 字符")
        parts.append(f"**位置**: {chunk.get('start_at', 0)}-{chunk.get('end_at', 0)}")
        parts.append("")
        parts.append(f"**内容预览**")
        parts.append("```")
        parts.append(chunk.get('content', ''))
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


@tool
async def list_knowledge_chunks(
    knowledge_id: str | None = None,
    knowledge_base_id: str | None = None,
    page: int = DEFAULT_PAGE,
    page_size: int = DEFAULT_PAGE_SIZE,
    tenant_id: int | None = None,
) -> str:
    """列出知识库或知识条目下的 chunks

    对齐 WeKnora99 ToolListKnowledgeChunks

    Args:
        knowledge_id: 知识条目 ID
        knowledge_base_id: 知识库 ID
        page: 页码 (默认 1)
        page_size: 每页数量 (默认 20, 最大 100)
        tenant_id: 租户 ID（自动从上下文获取）

    Returns:
        格式化的 chunks 列表

    Examples:
        ```python
        # 列出指定知识库的所有 chunks
        result = await list_knowledge_chunks(
            knowledge_base_id="kb-123"
        )

        # 列出指定知识条目的 chunks（分页）
        result = await list_knowledge_chunks(
            knowledge_id="k-456",
            page=2,
            page_size=10
        )
        ```
    """
    # 参数验证
    if not knowledge_id and not knowledge_base_id:
        return "错误: 请提供 knowledge_id 或 knowledge_base_id"

    if page < 1:
        page = DEFAULT_PAGE

    if page_size < 1:
        page_size = DEFAULT_PAGE_SIZE
    elif page_size > MAX_PAGE_SIZE:
        page_size = MAX_PAGE_SIZE

    # 获取租户 ID
    if tenant_id is None:
        from app.middleware import get_tenant_id
        tenant_id = get_tenant_id()
        if tenant_id is None:
            return "错误: 无法获取租户 ID"

    # 构建查询描述
    if knowledge_id:
        query = f"知识条目: {knowledge_id}"
    else:
        query = f"知识库: {knowledge_base_id}"

    logger.info(
        "list_chunks_start",
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        page=page,
        page_size=page_size,
    )

    try:
        # 获取 chunks
        chunks = await _list_chunks(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            page=page,
            page_size=page_size,
        )

        # 计算总数（简化：使用实际数量）
        total = len(chunks) + (page - 1) * page_size

        # 格式化结果
        result = _format_chunks_list(chunks, query, page, page_size, total)

        logger.info(
            "list_chunks_complete",
            tenant_id=tenant_id,
            count=len(chunks),
        )

        return result

    except Exception as e:
        logger.exception("list_chunks_failed", error=str(e))
        return f"获取 chunks 失败: {str(e)}"


__all__ = ["list_knowledge_chunks"]
