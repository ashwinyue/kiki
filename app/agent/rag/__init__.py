"""RAG (Retrieval-Augmented Generation) 服务层

提供租户级别/Agent 级别的知识库隔离和管理。
参考 DeerFlow 的服务层设计理念，RAG 作为独立服务而非全局工具。

## 核心特性

- **服务层独立**：RAG 作为服务层，不注册到全局工具表
- **多租户隔离**：支持租户级别的知识库隔离
- **Agent 级别分离**：每个 Agent 可以有专属的知识库
- **配置驱动**：支持多种 RAG 后端（FAISS、RAGFlow、Qdrant、Dify）

## 使用示例

### 1. 为单个 Agent 创建专属工具

```python
from app.agent.rag import create_rag_tool_for_agent
from app.agent.graph import create_react_agent
from app.agent.tools import search_web

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

### 2. 批量创建多个 Agent 的工具

```python
from app.agent.rag import create_multi_rag_tools

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

### 3. 配置 RAGFlow 远程服务

```python
from app.agent.rag import setup_agent_knowledge_base

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

### 4. 直接使用服务层

```python
from app.agent.rag import RAGService

service = RAGService()

# 注册知识库
from app.agent.rag import KnowledgeBaseConfig
kb_config = KnowledgeBaseConfig(
    name="research_docs",
    backend="ragflow",
    backend_config={"api_url": "...", "api_key": "..."},
)
service.register_knowledge_base(kb_config, tenant_id=123)

# 检索
results = await service.retrieve(
    query="Python 异步编程",
    tenant_id=123,
    knowledge_base="research_docs",
)
```

## 环境变量配置

```bash
# 选择默认后端
export RAG_DEFAULT_BACKEND=faiss  # faiss, ragflow, qdrant, dify

# FAISS（本地，无需额外配置）
export RAG_RETRIEVAL_SIZE=5

# RAGFlow（可选）
export RAGFLOW_API_URL=http://localhost:9388
export RAGFLOW_API_KEY=ragflow-xxx
export RAGFLOW_DATASET_ID=dataset-123

# Qdrant（待实现）
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=your-key

# Dify（待实现）
export DIFY_API_URL=http://localhost:5001
export DIFY_API_KEY=dify-xxx
```

## 架构设计

```
服务层（独立于工具）
├── RAGService
│   ├── 租户隔离
│   ├── Agent 级别知识库
│   └── 检索器生命周期管理
├── 支持的后端
│   ├── FAISS（本地）
│   ├── RAGFlow（远程）
│   ├── Qdrant（待实现）
│   └── Dify（待实现）
└── 工具工厂
    └── 动态创建 Agent 专属工具

Agent 创建
├── Researcher ──→ [search, crawl, *专属 RAG 工具]
├── Analyst ────→ [data_analysis, *专属 RAG 工具]
└── Coder ──────→ [python_repl, *专属 RAG 工具]
```
"""

# 导出配置相关
from app.agent.rag.config import (
    KnowledgeBaseConfig,
    RAGBackend,
    RAGConfig,
    create_retriever,
    load_rag_config_from_env,
    load_rag_config_from_yaml,
)

# 导出检索器相关
from app.agent.rag.retrievers.base import (
    BaseRetriever,
    RetrievedDocument,
    RetrievalError,
    RetrievalOptions,
)
from app.agent.rag.retrievers.faiss import FAISSRetriever
from app.agent.rag.retrievers.ragflow import (
    RAGFlowConfig,
    RAGFlowRetriever,
)

# 导出服务层
from app.agent.rag.service import (
    RetrievalContext,
    RAGService,
    get_rag_service,
)

# 导出工具工厂
from app.agent.rag.tools import (
    create_multi_rag_tools,
    create_rag_tool_for_agent,
    setup_agent_knowledge_base,
)

__all__ = [
    # 配置
    "RAGBackend",
    "RAGConfig",
    "KnowledgeBaseConfig",
    "create_retriever",
    "load_rag_config_from_env",
    "load_rag_config_from_yaml",
    # 检索器
    "BaseRetriever",
    "RetrievedDocument",
    "RetrievalError",
    "RetrievalOptions",
    "FAISSRetriever",
    "RAGFlowConfig",
    "RAGFlowRetriever",
    # 服务层
    "RAGService",
    "get_rag_service",
    "RetrievalContext",
    # 工具工厂
    "create_rag_tool_for_agent",
    "create_multi_rag_tools",
    "setup_agent_knowledge_base",
]

# 版本信息
__version__ = "0.2.0"
