"""知识图谱实体搜索工具

对 WeKnora99 ToolSearchEntity 的对齐实现。

在知识图谱中搜索与给定实体相关的实体。
"""

from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认配置
DEFAULT_MAX_RESULTS = 10
DEFAULT_MAX_DEPTH = 2


class EntitySearchError(Exception):
    """实体搜索错误"""
    pass


async def _search_entities_in_graph(
    entity_name: str,
    knowledge_base_ids: list[str] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[dict[str, Any]]:
    """在知识图谱中搜索相关实体

    Args:
        entity_name: 实体名称
        knowledge_base_ids: 知识库 ID 列表
        max_results: 最大结果数
        max_depth: 搜索深度

    Returns:
        相关实体列表
    """
    # TODO: 实现 Neo4j 图数据库查询
    # 目前返回示例数据

    # 示例返回数据
    entities = [
        {
            "name": f"Related Entity 1",
            "type": "concept",
            "distance": 1,
            "relationship": "related_to",
            "description": f"Entity related to {entity_name}",
        },
        {
            "name": f"Related Entity 2",
            "type": "technology",
            "distance": 1,
            "relationship": "part_of",
            "description": f"Part of {entity_name}",
        },
    ]

    return entities[:max_results]


def _format_entity_results(entities: list[dict[str, Any]], query: str) -> str:
    """格式化实体搜索结果

    Args:
        entities: 实体列表
        query: 原始查询

    Returns:
        格式化的结果字符串
    """
    if not entities:
        return f"未找到与 '{query}' 相关的实体"

    parts = [f"## 相关实体搜索结果 (共 {len(entities)} 条)"]
    parts.append(f"**查询**: {query}")
    parts.append("")

    for i, entity in enumerate(entities, 1):
        parts.append(f"### {i}. {entity.get('name', 'Unknown')}")
        parts.append(f"**类型**: {entity.get('type', 'unknown')}")
        parts.append(f"**关系**: {entity.get('relationship', 'unknown')}")
        parts.append(f"**距离**: {entity.get('distance', 0)}")
        if entity.get("description"):
            parts.append(f"**描述**: {entity['description']}")
        parts.append("")

    return "\n".join(parts)


@tool
async def search_entity(
    entity_name: str,
    knowledge_base_ids: list[str] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    """知识图谱实体搜索

    在知识图谱中搜索与给定实体相关的实体。
    用于增强 RAG 搜索的实体关系理解。

    Args:
        entity_name: 要搜索的实体名称，如 "Python"、"Docker"
        knowledge_base_ids: 限制搜索的知识库 ID 列表
        max_results: 最大返回结果数 (默认 10)
        max_depth: 搜索深度，最大跳数 (默认 2)

    Returns:
        格式化的相关实体列表

    Examples:
        ```python
        # 搜索 Python 相关的实体
        result = await search_entity("Python")

        # 限制知识库和结果数
        result = await search_entity(
            "Docker",
            knowledge_base_ids=["kb-123"],
            max_results=5
        )
        ```
    """
    # 参数验证
    if not entity_name or not entity_name.strip():
        return "错误: 请提供有效的实体名称"

    if max_results < 1:
        max_results = DEFAULT_MAX_RESULTS
    elif max_results > 50:
        max_results = 50

    if max_depth < 1:
        max_depth = 1
    elif max_depth > 5:
        max_depth = 5

    logger.info(
        "search_entity_start",
        entity=entity_name,
        max_results=max_results,
        max_depth=max_depth,
    )

    try:
        # 执行搜索
        entities = await _search_entities_in_graph(
            entity_name=entity_name,
            knowledge_base_ids=knowledge_base_ids,
            max_results=max_results,
            max_depth=max_depth,
        )

        # 格式化结果
        result = _format_entity_results(entities, entity_name)

        logger.info(
            "search_entity_complete",
            entity=entity_name,
            result_count=len(entities),
        )

        return result

    except Exception as e:
        logger.exception("search_entity_failed", error=str(e))
        return f"搜索失败: {str(e)}"


__all__ = ["search_entity"]
