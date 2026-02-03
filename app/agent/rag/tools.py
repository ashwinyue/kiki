"""RAG 工具工厂

为 Agent 动态创建专属的 RAG 工具，支持租户级别/Agent 级别的知识库隔离。

设计理念：
- RAG 是服务层，不是工具层
- 工具按 Agent 创建，不注册到全局表
- 支持多租户/多 Agent 隔离

使用示例：
    ```python
    from app.agent.rag import create_rag_tool_for_agent

    # 为研究员 Agent 创建专属工具
    researcher_tool = create_rag_tool_for_agent(
        agent_name="researcher",
        tenant_id=123,
        knowledge_base="research_docs",
    )

    # 创建 Agent
    from app.agent.graph import create_react_agent
    agent = create_react_agent(
        agent_name="researcher",
        tools=[search_web, researcher_tool],
    )
    ```
"""

from typing import Any

from langchain_core.tools import BaseTool

from app.agent.rag.config import KnowledgeBaseConfig
from app.agent.rag.retrievers.base import RetrievalOptions
from app.agent.rag.service import RAGService, get_rag_service, RetrievalContext
from app.observability.logging import get_logger

logger = get_logger(__name__)


def create_rag_tool_for_agent(
    agent_name: str,
    tenant_id: int | None = None,
    knowledge_base: str = "default",
    service: RAGService | None = None,
    tool_name: str | None = None,
) -> BaseTool:
    """为 Agent 创建专属的 RAG 工具

    这是推荐的方式，为每个 Agent 创建专属的知识库检索工具。

    Args:
        agent_name: Agent 名称
        tenant_id: 租户 ID（None 表示全局租户）
        knowledge_base: 知识库名称
        service: RAG 服务实例（可选，默认使用全局实例）
        tool_name: 工具名称（默认: search_{agent_name}_knowledge）

    Returns:
        LangChain 工具实例

    Example:
        ```python
        from app.agent.rag import create_rag_tool_for_agent
        from app.agent.tools import search_web
        from app.agent.graph import create_react_agent

        # 为研究员创建专属工具
        researcher_tool = create_rag_tool_for_agent(
            agent_name="researcher",
            tenant_id=123,
            knowledge_base="research_docs",
        )

        # 创建 Agent
        researcher = create_react_agent(
            agent_name="researcher",
            tools=[search_web, researcher_tool],
        )
        ```
    """
    service = service or get_rag_service()
    return service.create_tool_for_agent(
        agent_name=agent_name,
        tenant_id=tenant_id,
        knowledge_base=knowledge_base,
        tool_name=tool_name,
    )


def create_multi_rag_tools(
    agent_configs: list[dict[str, Any]],
    service: RAGService | None = None,
) -> dict[str, BaseTool]:
    """批量创建多个 Agent 的 RAG 工具

    Args:
        agent_configs: Agent 配置列表，每个配置包含：
            - agent_name: Agent 名称
            - tenant_id: 租户 ID（可选）
            - knowledge_base: 知识库名称（可选）
            - tool_name: 工具名称（可选）
        service: RAG 服务实例（可选）

    Returns:
        工具字典，key 为 agent_name

    Example:
        ```python
        tools = create_multi_rag_tools([
            {"agent_name": "researcher", "knowledge_base": "research_docs"},
            {"agent_name": "analyst", "knowledge_base": "analysis_docs"},
            {"agent_name": "coder", "knowledge_base": "code_docs"},
        ])

        researcher = create_react_agent(
            agent_name="researcher",
            tools=[search_web, tools["researcher"]],
        )
        ```
    """
    service = service or get_rag_service()
    result = {}

    for config in agent_configs:
        agent_name = config["agent_name"]
        tool = service.create_tool_for_agent(
            agent_name=agent_name,
            tenant_id=config.get("tenant_id"),
            knowledge_base=config.get("knowledge_base", "default"),
            tool_name=config.get("tool_name"),
        )
        result[agent_name] = tool

    logger.info(
        "multi_rag_tools_created",
        count=len(result),
    )

    return result


def setup_agent_knowledge_base(
    agent_name: str,
    knowledge_base: str,
    backend: str = "faiss",
    backend_config: dict[str, Any] | None = None,
    tenant_id: int | None = None,
    service: RAGService | None = None,
) -> tuple[KnowledgeBaseConfig, BaseTool]:
    """设置 Agent 的知识库并创建专属工具

    这是一个便捷函数，一次性完成知识库注册和工具创建。

    Args:
        agent_name: Agent 名称
        knowledge_base: 知识库名称
        backend: RAG 后端类型
        backend_config: 后端配置
        tenant_id: 租户 ID
        service: RAG 服务实例

    Returns:
        (知识库配置, RAG 工具) 元组

    Example:
        ```python
        # 为研究员设置 RAGFlow 知识库
        kb_config, researcher_tool = setup_agent_knowledge_base(
            agent_name="researcher",
            knowledge_base="research_docs",
            backend="ragflow",
            backend_config={
                "api_url": "http://localhost:9388",
                "api_key": "ragflow-xxx",
                "dataset_id": "dataset-123",
            },
            tenant_id=123,
        )
        ```
    """
    service = service or get_rag_service()

    # 注册知识库配置
    kb_config = KnowledgeBaseConfig(
        name=knowledge_base,
        backend=backend,  # type: ignore
        backend_config=backend_config or {},
    )
    service.register_knowledge_base(kb_config, tenant_id)

    # 创建工具
    tool = service.create_tool_for_agent(
        agent_name=agent_name,
        tenant_id=tenant_id,
        knowledge_base=knowledge_base,
    )

    logger.info(
        "agent_knowledge_base_setup",
        agent_name=agent_name,
        knowledge_base=knowledge_base,
        backend=backend,
    )

    return kb_config, tool


__all__ = [
    "create_rag_tool_for_agent",
    "create_multi_rag_tools",
    "setup_agent_knowledge_base",
]
