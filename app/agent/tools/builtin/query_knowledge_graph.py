"""知识图谱查询工具

基于实体名称查询知识图谱，返回相关的实体和关系。
对齐 WeKnora99 ToolQueryKnowledgeGraph。

功能：
- 基于实体名称查询知识图谱
- 返回相关的实体和关系
- 支持 Neo4j 图数据库

依赖：
- neo4j 驱动程序（可选依赖）
"""

from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认配置
DEFAULT_MAX_RESULTS = 10
DEFAULT_MAX_DEPTH = 2


class QueryKnowledgeGraphError(Exception):
    """知识图谱查询错误"""
    pass


def _get_neo4j_driver() -> Any:
    """获取 Neo4j 驱动程序

    Returns:
        Neo4j 异步驱动实例

    Raises:
        QueryKnowledgeGraphError: Neo4j 未安装或配置不完整
    """
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        raise QueryKnowledgeGraphError("neo4j 驱动程序未安装，请运行: uv add neo4j")

    from app.config.settings import get_settings

    settings = get_settings()
    neo4j_url = getattr(settings, "neo4j_url", None)
    neo4j_user = getattr(settings, "neo4j_user", "neo4j")
    neo4j_password = getattr(settings, "neo4j_password", None)

    if not neo4j_url or not neo4j_password:
        raise QueryKnowledgeGraphError("Neo4j 配置不完整，请检查 KIKI_NEO4J_URL 和 KIKI_NEO4J_PASSWORD")

    return AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))


async def _query_entities_and_relations(
    driver: Any,
    entity_name: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[dict[str, Any]]:
    """查询实体和关系

    Args:
        driver: Neo4j 驱动
        entity_name: 实体名称
        max_results: 最大结果数
        max_depth: 最大查询深度

    Returns:
        实体和关系列表
    """
    # Cypher 查询：查找与给定实体相关的节点和关系
    cypher_query = """
    MATCH path = (start:Entity {name: $entity_name})-[*1..$max_depth]-(related:Entity)
    WITH related, relationships(path) as rels, length(path) as dist
    ORDER BY dist ASC, related.name ASC
    LIMIT $max_results
    UNWIND rels as rel
    RETURN
        related.name as entity_name,
        related.type as entity_type,
        collect(DISTINCT {
            type: type(rel),
            from: start(rel).name,
            to: end(rel).name,
            properties: properties(rel)
        }) as relations,
        dist as distance,
        [(related)-[r:MENTIONS]->(d:Document) | {
            title: d.title,
            source: d.source,
            chunk_id: d.chunk_id
        }] as documents
    """

    async with driver.session() as session:
        try:
            result = await session.run(
                cypher_query,
                entity_name=entity_name,
                max_depth=max_depth,
                max_results=max_results,
            )
            records = await result.data()
            return records
        except Exception as e:
            logger.error(
                "neo4j_query_failed",
                entity_name=entity_name,
                error=str(e),
            )
            raise QueryKnowledgeGraphError(f"图数据库查询失败: {str(e)}")


async def _get_entity_info(
    driver: Any,
    entity_name: str,
) -> dict[str, Any] | None:
    """获取实体详细信息

    Args:
        driver: Neo4j 驱动
        entity_name: 实体名称

    Returns:
        实体信息字典
    """
    cypher_query = """
    MATCH (e:Entity {name: $entity_name})
    OPTIONAL MATCH (e)-[r:MENTIONS]->(d:Document)
    RETURN
        e.name as name,
        e.type as type,
        e.description as description,
        collect(DISTINCT {
            type: type(r),
            title: d.title,
            source: d.source
        }) as documents
    LIMIT 1
    """

    async with driver.session() as session:
        result = await session.run(cypher_query, entity_name=entity_name)
        record = await result.data()
        return record[0] if record else None


async def _get_relation_types(driver: Any) -> list[dict[str, str]]:
    """获取所有关系类型

    Args:
        driver: Neo4j 驱动

    Returns:
        关系类型列表
    """
    cypher_query = """
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) as relation_type, count(*) as count
    ORDER BY count DESC
    """

    async with driver.session() as session:
        result = await session.run(cypher_query)
        records = await result.data()
        return records


async def _get_entity_types(driver: Any) -> list[dict[str, str]]:
    """获取所有实体类型

    Args:
        driver: Neo4j 驱动

    Returns:
        实体类型列表
    """
    cypher_query = """
    MATCH (e:Entity)
    RETURN DISTINCT e.type as entity_type, count(*) as count
    ORDER BY count DESC
    """

    async with driver.session() as session:
        result = await session.run(cypher_query)
        records = await result.data()
        return records


def _format_results(
    results: list[dict[str, Any]],
    entity_name: str,
    has_graph_config: bool,
) -> str:
    """格式化查询结果

    Args:
        results: 查询结果列表
        entity_name: 查询的实体名称
        has_graph_config: 是否有图谱配置

    Returns:
        格式化的结果字符串
    """
    if not results:
        if has_graph_config:
            return f"未找到与 '{entity_name}' 相关的实体"
        return "知识图谱未配置或无法连接"

    parts = [f"## 知识图谱查询结果\n"]
    parts.append(f"**查询实体**: {entity_name}\n")
    parts.append(f"**相关结果数**: {len(results)}\n\n")

    for i, record in enumerate(results, 1):
        entity_name_result = record.get("entity_name", "未知")
        entity_type = record.get("entity_type", "Unknown")
        distance = record.get("distance", 0)
        relations = record.get("relations", [])
        documents = record.get("documents", [])

        parts.append(f"### {i}. {entity_name_result}\n")
        parts.append(f"- **类型**: {entity_type}\n")
        parts.append(f"- **距离**: {distance} 跳\n")

        if relations:
            parts.append(f"- **关系数**: {len(relations)}\n")
            parts.append("\n**关系详情**:\n")
            for rel in relations[:5]:  # 最多显示 5 个关系
                rel_type = rel.get("type", "UNKNOWN")
                from_node = rel.get("from", "")
                to_node = rel.get("to", "")
                parts.append(f"  - `{from_node}` --[{rel_type}]--> `{to_node}`\n")

        if documents:
            parts.append("\n**相关文档**:\n")
            for doc in documents[:3]:  # 最多显示 3 个文档
                title = doc.get("title", "未知")
                source = doc.get("source", "")
                parts.append(f"  - {title}")
                if source:
                    parts.append(f" ({source})")
                parts.append("\n")

        parts.append("\n")

    return "".join(parts)


def _build_graph_data(results: list[dict[str, Any]], query_entity: str) -> dict[str, Any]:
    """构建用于前端可视化的图数据

    Args:
        results: 查询结果
        query_entity: 查询的实体名称

    Returns:
        图数据字典
    """
    nodes = []
    edges = []
    seen_nodes = set()

    # 添加查询实体
    nodes.append({
        "id": query_entity,
        "label": query_entity,
        "type": "query_entity",
        "size": 20,
    })
    seen_nodes.add(query_entity)

    for record in results:
        entity_name = record.get("entity_name", "")
        entity_type = record.get("entity_type", "Unknown")
        relations = record.get("relations", [])

        # 添加节点
        if entity_name and entity_name not in seen_nodes:
            nodes.append({
                "id": entity_name,
                "label": entity_name,
                "type": entity_type,
                "size": 12,
            })
            seen_nodes.add(entity_name)

        # 添加边
        for rel in relations:
            from_node = rel.get("from", "")
            to_node = rel.get("to", "")
            rel_type = rel.get("type", "RELATED")

            if from_node and to_node:
                edge_id = f"{from_node}-{rel_type}-{to_node}"
                edges.append({
                    "id": edge_id,
                    "source": from_node,
                    "target": to_node,
                    "type": rel_type,
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


@tool
async def query_knowledge_graph(
    entity_name: str,
    knowledge_base_ids: list[str] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    tenant_id: int | None = None,
) -> str:
    """查询知识图谱，探索实体关系和知识网络

    在配置了图谱抽取的知识库中探索实体之间的关系。
    支持 Neo4j 图数据库查询。

    Args:
        entity_name: 要查询的实体名称，如 "Docker"、"Kubernetes"、"Python"
        knowledge_base_ids: 知识库 ID 列表（可选，用于过滤特定知识库）
        max_results: 最大返回结果数 (默认 10)
        max_depth: 查询深度，最大跳数 (默认 2)
        tenant_id: 租户 ID（自动从上下文获取）

    Returns:
        格式化的知识图谱查询结果，包含实体、关系和相关信息

    Examples:
        ```python
        # 查询 "Docker" 相关的实体和关系
        result = await query_knowledge_graph("Docker")

        # 查询 "Python" 并限制结果数量
        result = await query_knowledge_graph("Python", max_results=20)

        # 深度查询 "AI" 的关系网络
        result = await query_knowledge_graph("AI", max_depth=3)
        ```
    """
    # 参数验证
    if not entity_name:
        return "错误: 请提供要查询的实体名称"

    if max_results < 1:
        max_results = DEFAULT_MAX_RESULTS
    elif max_results > 50:
        max_results = 50  # 限制最大结果数

    if max_depth < 1:
        max_depth = DEFAULT_MAX_DEPTH
    elif max_depth > 5:
        max_depth = 5  # 限制最大深度

    # 获取租户 ID（如果没有提供）
    if tenant_id is None:
        from app.middleware import get_tenant_id
        tenant_id = get_tenant_id()
        if tenant_id is None:
            tenant_id = 0  # 使用默认租户

    logger.info(
        "query_knowledge_graph_start",
        tenant_id=tenant_id,
        entity_name=entity_name,
        max_results=max_results,
        max_depth=max_depth,
        kb_filter=knowledge_base_ids,
    )

    try:
        # 尝试获取 Neo4j 驱动
        try:
            driver = _get_neo4j_driver()
        except QueryKnowledgeGraphError:
            # 如果没有配置 Neo4j，返回提示信息
            return (
                "知识图谱查询工具暂未配置。\n"
                "请在环境变量中配置以下参数：\n"
                "- KIKI_NEO4J_URL: Neo4j 连接 URL (如: bolt://localhost:7687)\n"
                "- KIKI_NEO4J_USER: Neo4j 用户名\n"
                "- KIKI_NEO4J_PASSWORD: Neo4j 密码\n\n"
                f"当前查询: {entity_name}\n"
                "提示: 请先确保知识库已配置图谱抽取，并等待图谱数据构建完成。"
            )
        except Exception as e:
            logger.error("neo4j_connection_failed", error=str(e))
            return f"知识图谱连接失败: {str(e)}\n请检查 Neo4j 服务是否正常运行。"

        try:
            # 并行查询实体信息
            import asyncio
            results, entity_info = await asyncio.gather([
                _query_entities_and_relations(
                    driver=driver,
                    entity_name=entity_name,
                    max_results=max_results,
                    max_depth=max_depth,
                ),
                _get_entity_info(driver, entity_name),
            ])

            # 检查是否有结果
            has_graph_config = len(results) > 0

            # 格式化输出
            formatted_output = _format_results(results, entity_name, has_graph_config)

            # 添加使用提示
            if not has_graph_config:
                formatted_output += "\n---\n"
                formatted_output += "**使用提示**:\n"
                formatted_output = (
                    "- 知识图谱未找到相关实体关系\n"
                    "- 请确认：\n"
                    "  1. 知识库已配置图谱抽取\n"
                    "  2. 文档已处理完成\n"
                    "  3. 实体名称拼写正确\n"
                    "- 完整的图查询语言（Cypher）支持开发中\n"
                )

            logger.info(
                "query_knowledge_graph_complete",
                tenant_id=tenant_id,
                entity_name=entity_name,
                result_count=len(results),
            )

            return formatted_output

        finally:
            await driver.close()

    except QueryKnowledgeGraphError as e:
        logger.error("query_knowledge_graph_failed", tenant_id=tenant_id, error=str(e))
        return f"知识图谱查询失败: {str(e)}"

    except Exception as e:
        logger.exception("query_knowledge_graph_error", tenant_id=tenant_id, error=str(e))
        return f"知识图谱查询出错: {str(e)}"


__all__ = ["query_knowledge_graph", "QueryKnowledgeGraphError"]
