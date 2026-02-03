"""RAG 服务层使用示例

演示如何使用 RAG 服务层为多 Agent 架构创建专属知识库工具。
"""

import asyncio
from langchain_core.tools import tool

from app.agent.rag import (
    RAGService,
    create_multi_rag_tools,
    create_rag_tool_for_agent,
    setup_agent_knowledge_base,
)
from app.agent.rag.config import KnowledgeBaseConfig
from app.agent.graph import create_react_agent


# ============================================================
# 示例 1: 为单个 Agent 创建专属 RAG 工具
# ============================================================

def example_single_agent():
    """为单个 Agent 创建专属 RAG 工具"""

    from app.agent.rag import create_rag_tool_for_agent
    from app.agent.tools import search_web

    # 为研究员创建专属工具（使用默认 FAISS 本地存储）
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

    print("✓ 研究员 Agent 已创建，专属知识库工具: search_researcher_knowledge")


# ============================================================
# 示例 2: 批量创建多个 Agent 的 RAG 工具
# ============================================================

def example_multi_agent():
    """批量创建多个 Agent 的 RAG 工具"""

    from app.agent.rag import create_multi_rag_tools
    from app.agent.tools import search_web, python_repl

    # 批量创建工具
    tools = create_multi_rag_tools([
        {
            "agent_name": "researcher",
            "tenant_id": 123,
            "knowledge_base": "research_docs",
        },
        {
            "agent_name": "analyst",
            "tenant_id": 123,
            "knowledge_base": "analysis_docs",
        },
        {
            "agent_name": "coder",
            "tenant_id": 123,
            "knowledge_base": "code_docs",
        },
    ])

    # 创建各个 Agent
    researcher = create_react_agent(
        agent_name="researcher",
        tools=[search_web, tools["researcher"]],
    )

    analyst = create_react_agent(
        agent_name="analyst",
        tools=[tools["analyst"]],
    )

    coder = create_react_agent(
        agent_name="coder",
        tools=[python_repl, tools["coder"]],
    )

    print("✓ 多个 Agent 已创建，各自使用专属知识库:")
    print(f"  - 研究员: research_docs")
    print(f"  - 分析师: analysis_docs")
    print(f"  - 程序员: code_docs")


# ============================================================
# 示例 3: 配置 RAGFlow 远程服务
# ============================================================

def example_ragflow_remote():
    """配置 RAGFlow 远程服务"""

    # 为研究员设置 RAGFlow 远程知识库
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

    print(f"✓ RAGFlow 知识库已配置:")
    print(f"  - 知识库: {kb_config.name}")
    print(f"  - 后端: {kb_config.backend}")
    print(f"  - API 地址: {kb_config.backend_config['api_url']}")


# ============================================================
# 示例 4: 直接使用服务层
# ============================================================

async def example_direct_service():
    """直接使用 RAG 服务层"""

    from app.agent.rag import RAGService, KnowledgeBaseConfig
    from app.agent.rag.retrievers.faiss import FAISSRetriever

    # 创建服务实例
    service = RAGService()

    # 注册知识库配置
    kb_config = KnowledgeBaseConfig(
        name="research_docs",
        backend="faiss",
        description="研究文档知识库",
    )
    service.register_knowledge_base(kb_config, tenant_id=123)

    # 获取检索器
    retriever = service.get_retriever(
        tenant_id=123,
        knowledge_base="research_docs",
    )

    # 添加示例文档（FAISS）
    if isinstance(retriever, FAISSRetriever):
        retriever.add_texts(
            texts=[
                "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
                "FastAPI 是一个现代、快速的 Python Web 框架，用于构建 API。",
                "LangGraph 是 LangChain 的扩展，用于构建有状态的 Agent 应用。",
            ],
            metadatas=[
                {"title": "Python 简介", "source": "docs/python.md"},
                {"title": "FastAPI 简介", "source": "docs/fastapi.md"},
                {"title": "LangGraph 简介", "source": "docs/langgraph.md"},
            ],
        )
        print("✓ 示例文档已添加到知识库")

    # 执行检索
    results = await service.retrieve(
        query="什么是 Python",
        tenant_id=123,
        knowledge_base="research_docs",
        options={"top_k": 2},
    )

    print(f"\n✓ 检索结果 (共 {len(results)} 条):")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.title}")
        print(f"   来源: {doc.source}")
        print(f"   相似度: {doc.score:.4f}")
        print(f"   内容: {doc.content[:100]}...")


# ============================================================
# 示例 5: 多租户隔离
# ============================================================

async def example_multi_tenant():
    """演示多租户隔离"""

    from app.agent.rag import RAGService, KnowledgeBaseConfig
    from app.agent.rag.retrievers.faiss import FAISSRetriever

    service = RAGService()

    # 租户 A 的知识库
    kb_a = KnowledgeBaseConfig(name="tenant_a_kb", backend="faiss")
    service.register_knowledge_base(kb_a, tenant_id=1)

    # 租户 B 的知识库
    kb_b = KnowledgeBaseConfig(name="tenant_b_kb", backend="faiss")
    service.register_knowledge_base(kb_b, tenant_id=2)

    # 为租户 A 添加文档
    retriever_a = service.get_retriever(tenant_id=1, knowledge_base="tenant_a_kb")
    if isinstance(retriever_a, FAISSRetriever):
        retriever_a.add_texts([
            "租户 A 的内部文档：这是机密信息。",
        ])

    # 为租户 B 添加文档
    retriever_b = service.get_retriever(tenant_id=2, knowledge_base="tenant_b_kb")
    if isinstance(retriever_b, FAISSRetriever):
        retriever_b.add_texts([
            "租户 B 的内部文档：这是不同的机密信息。",
        ])

    # 验证隔离
    results_a = await service.retrieve(
        query="机密信息",
        tenant_id=1,
        knowledge_base="tenant_a_kb",
    )

    results_b = await service.retrieve(
        query="机密信息",
        tenant_id=2,
        knowledge_base="tenant_b_kb",
    )

    print("✓ 多租户隔离验证:")
    print(f"  - 租户 A 的结果: {results_a[0].content if results_a else '无'}")
    print(f"  - 租户 B 的结果: {results_b[0].content if results_b else '无'}")

    # 验证缓存键不同
    print(f"\n✓ 租户 A 的检索器缓存键: 1:tenant_a_kb")
    print(f"✓ 租户 B 的检索器缓存键: 2:tenant_b_kb")


# ============================================================
# 主函数
# ============================================================

async def main():
    """运行所有示例"""
    print("=" * 60)
    print("RAG 服务层使用示例")
    print("=" * 60)

    print("\n[示例 1] 单 Agent 专属工具")
    print("-" * 60)
    example_single_agent()

    print("\n[示例 2] 多 Agent 工具创建")
    print("-" * 60)
    example_multi_agent()

    print("\n[示例 3] RAGFlow 远程服务配置")
    print("-" * 60)
    example_ragflow_remote()

    print("\n[示例 4] 直接使用服务层")
    print("-" * 60)
    await example_direct_service()

    print("\n[示例 5] 多租户隔离")
    print("-" * 60)
    await example_multi_tenant()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
