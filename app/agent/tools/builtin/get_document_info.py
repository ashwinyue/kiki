"""获取文档信息工具

对 WeKnora99 ToolGetDocumentInfo 的对齐实现。

获取知识条目的详细信息。
"""

from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)


async def _get_document_info(
    tenant_id: int,
    knowledge_id: str,
) -> dict[str, Any] | None:
    """获取文档信息

    Args:
        tenant_id: 租户 ID
        knowledge_id: 知识条目 ID

    Returns:
        文档信息字典
    """
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models.knowledge import Knowledge, KnowledgeBase, Chunk

    session_factory = None
    try:
        from app.infra.database import async_session_factory
        session_factory = async_session_factory
    except ImportError:
        return None

    if not session_factory:
        return None

    session = session_factory()

    try:
        # 获取知识条目
        stmt = (
            select(Knowledge, KnowledgeBase)
            .join(KnowledgeBase, Knowledge.knowledge_base_id == KnowledgeBase.id, isouter=True)
            .where(
                Knowledge.id == knowledge_id,
                Knowledge.tenant_id == tenant_id,
                Knowledge.deleted_at.is_(None),
            )
        )
        result = await session.execute(stmt)
        row = result.first()

        if not row:
            return None

        knowledge, kb = row

        # 统计 chunks 数量
        chunk_stmt = select(func.count()).where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        chunk_result = await session.execute(chunk_stmt)
        chunk_count = chunk_result.scalar() or 0

        return {
            "id": knowledge.id,
            "title": knowledge.title,
            "type": knowledge.type,
            "source": knowledge.source,
            "description": knowledge.description,
            "file_name": knowledge.file_name,
            "file_type": knowledge.file_type,
            "file_size": knowledge.file_size,
            "knowledge_base_id": knowledge.knowledge_base_id,
            "knowledge_base_name": kb.name if kb else None,
            "parse_status": knowledge.parse_status,
            "enable_status": knowledge.enable_status,
            "chunk_count": chunk_count,
            "tag_id": knowledge.tag_id,
            "created_at": knowledge.created_at.isoformat() if knowledge.created_at else None,
            "updated_at": knowledge.updated_at.isoformat() if knowledge.updated_at else None,
            "processed_at": knowledge.processed_at.isoformat() if knowledge.processed_at else None,
        }

    finally:
        await session.close()


def _format_document_info(info: dict[str, Any]) -> str:
    """格式化文档信息

    Args:
        info: 文档信息字典

    Returns:
        格式化的结果字符串
    """
    parts = [
        "## 文档信息",
        f"**ID**: {info.get('id', 'N/A')}",
        f"**标题**: {info.get('title', 'N/A')}",
        "",
        "### 基本信息",
        f"**类型**: {info.get('type', 'N/A')}",
        f"**来源**: {info.get('source', 'N/A')}",
        f"**状态**: {info.get('parse_status', 'N/A')}",
        f"**启用状态**: {info.get('enable_status', 'N/A')}",
        "",
    ]

    if info.get('file_name'):
        parts.extend([
            "### 文件信息",
            f"**文件名**: {info['file_name']}",
            f"**文件类型**: {info.get('file_type', 'N/A')}",
            f"**文件大小**: {info.get('file_size', 0) or 0} bytes",
            "",
        ])

    parts.extend([
        "### 知识库",
        f"**知识库 ID**: {info.get('knowledge_base_id', 'N/A')}",
        f"**知识库名称**: {info.get('knowledge_base_name', 'N/A')}",
        "",
        "### 处理统计",
        f"**Chunks 数量**: {info.get('chunk_count', 0)}",
        f"**标签 ID**: {info.get('tag_id', 'N/A') or 'N/A'}",
        "",
        "### 时间信息",
        f"**创建时间**: {info.get('created_at', 'N/A')}",
        f"**更新时间**: {info.get('updated_at', 'N/A')}",
        f"**处理完成**: {info.get('processed_at', 'N/A')}",
    ])

    return "\n".join(parts)


@tool
async def get_document_info(
    knowledge_id: str,
    tenant_id: int | None = None,
) -> str:
    """获取知识条目详细信息

    对齐 WeKnora99 ToolGetDocumentInfo

    Args:
        knowledge_id: 知识条目 ID
        tenant_id: 租户 ID（自动从上下文获取）

    Returns:
        格式化的文档信息

    Examples:
        ```python
        # 获取文档信息
        result = await get_document_info("knowledge-id")

        # 指定租户
        result = await get_document_info(
            "knowledge-id",
            tenant_id=1
        )
        ```
    """
    # 参数验证
    if not knowledge_id or not knowledge_id.strip():
        return "错误: 请提供有效的 knowledge_id"

    # 获取租户 ID
    if tenant_id is None:
        from app.middleware import get_tenant_id
        tenant_id = get_tenant_id()
        if tenant_id is None:
            return "错误: 无法获取租户 ID"

    logger.info(
        "get_document_info_start",
        knowledge_id=knowledge_id,
        tenant_id=tenant_id,
    )

    try:
        # 获取文档信息
        info = await _get_document_info(tenant_id, knowledge_id)

        if not info:
            return f"未找到知识条目: {knowledge_id}"

        # 格式化结果
        result = _format_document_info(info)

        logger.info(
            "get_document_info_complete",
            knowledge_id=knowledge_id,
            title=info.get('title', 'N/A')[:50],
        )

        return result

    except Exception as e:
        logger.exception("get_document_info_failed", error=str(e))
        return f"获取文档信息失败: {str(e)}"


__all__ = ["get_document_info"]
