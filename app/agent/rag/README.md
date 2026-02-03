# RAG 服务层架构

## 概述

RAG (Retrieval-Augmented Generation) **服务层**架构，参考 DeerFlow 设计理念，提供租户级别/Agent 级别的知识库隔离和管理。

### 核心特性

| 特性 | 说明 |
|------|------|
| **服务层独立** | RAG 作为服务层，不注册到全局工具表 |
| **多租户隔离** | 不同租户使用独立的检索器缓存 |
| **Agent 级别分离** | 每个 Agent 可以有专属的知识库 |
| **配置驱动** | 支持多种后端（FAISS、RAGFlow、Qdrant、Dify） |

---

## 架构对比

### 旧架构：全局工具注册

```
全局工具注册表
├── search_web
├── python_repl
└── search_knowledge_base  ← 所有 Agent 共享

Agent A ──┐
Agent B ──┼──→ 同一个 RAG 检索器（无隔离）
Agent C ──┘
```

**问题**：缺乏租户/Agent 隔离，无法按需配置

### 新架构：服务层 + 动态工具

```
服务层
├── RAGService
│   ├── 租户 123
│   │   ├── researcher ──→ research_docs 检索器
│   │   ├── analyst ──────→ analysis_docs 检索器
│   │   └── coder ────────→ code_docs 检索器
│   └── 租户 456
│       └── researcher ──→ 不同的 research_docs 检索器
└── 工具工厂
    └── 动态创建 Agent 专属工具

Agent 创建
├── Researcher ──→ [search, *专属 RAG 工具]
├── Analyst ────→ [data_analysis, *专属 RAG 工具]
└── Coder ──────→ [python_repl, *专属 RAG 工具]
```

---

## 快速开始

### 1. 为 Agent 创建专属工具

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

# Agent 可以使用专属知识库
result = await researcher.ainvoke({
    "messages": [("user", "搜索研究文档：Python 异步编程")]
})
```

### 2. 批量创建多个 Agent 工具

```python
from app.agent.rag import create_multi_rag_tools

tools = create_multi_rag_tools([
    {"agent_name": "researcher", "knowledge_base": "research_docs"},
    {"agent_name": "analyst", "knowledge_base": "analysis_docs"},
    {"agent_name": "coder", "knowledge_base": "code_docs"},
])

# 使用工具
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

---

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

---

## API 参考

### 服务层

#### `RAGService`

RAG 服务管理器。

```python
service = RAGService(config=None)

# 注册知识库
service.register_knowledge_base(kb_config, tenant_id=None)

# 获取检索器
retriever = service.get_retriever(tenant_id=None, knowledge_base="default")

# 执行检索
results = await service.retrieve(query, tenant_id=None, knowledge_base="default")

# 为 Agent 创建工具
tool = service.create_tool_for_agent(agent_name, tenant_id=None, knowledge_base="default")
```

#### `get_rag_service()`

获取全局 RAG 服务实例。

```python
service = get_rag_service()
```

### 工具工厂

#### `create_rag_tool_for_agent()`

为单个 Agent 创建专属 RAG 工具。

```python
tool = create_rag_tool_for_agent(
    agent_name="researcher",
    tenant_id=123,
    knowledge_base="research_docs",
    service=None,  # 可选，默认使用全局服务
    tool_name=None,  # 可选，默认: search_{agent_name}_knowledge
)
```

#### `create_multi_rag_tools()`

批量创建多个 Agent 的 RAG 工具。

```python
tools = create_multi_rag_tools([
    {"agent_name": "researcher", "tenant_id": 123, "knowledge_base": "research_docs"},
    {"agent_name": "analyst", "tenant_id": 123, "knowledge_base": "analysis_docs"},
])
# 返回: {"researcher": tool1, "analyst": tool2}
```

#### `setup_agent_knowledge_base()`

设置 Agent 的知识库并创建专属工具（便捷函数）。

```python
kb_config, tool = setup_agent_knowledge_base(
    agent_name="researcher",
    knowledge_base="research_docs",
    backend="faiss",
    backend_config={},
    tenant_id=123,
)
```

### 配置

#### `KnowledgeBaseConfig`

知识库配置。

```python
kb_config = KnowledgeBaseConfig(
    name="research_docs",
    backend="faiss",
    backend_config={},
    description="研究文档知识库",
)
```

---

## 支持的后端

| 后端 | 类型 | 状态 | 适用场景 |
|------|------|------|----------|
| **FAISS** | 本地向量存储 | ✅ 已实现 | 开发、演示、离线场景 |
| **RAGFlow** | 远程服务 | ✅ 已实现 | 生产环境、完整 RAG 引擎 |
| **Qdrant** | 向量数据库 | 🔜 待实现 | 高性能向量检索 |
| **Dify** | LLM 平台 | 🔜 待实现 | 一站式 Agent 开发 |

---

## 项目结构

```
app/agent/rag/
├── __init__.py         # 统一导出接口
├── config.py           # 配置管理
├── service.py          # 服务管理器（核心）
├── tools.py            # 工具工厂
├── examples.py         # 使用示例
├── README.md           # 本文档
└── retrievers/
    ├── __init__.py
    ├── base.py         # 抽象基类
    ├── faiss.py        # FAISS 本地检索器
    └── ragflow.py      # RAGFlow 远程检索器
```

---

## 最佳实践

1. **开发阶段**：使用 FAISS 本地存储，快速迭代
2. **生产环境**：使用 RAGFlow 远程服务，获得完整功能
3. **多租户**：每个租户使用不同的 `tenant_id`
4. **Agent 隔离**：每个 Agent 使用不同的 `knowledge_base`
5. **配置管理**：使用环境变量或 YAML 配置管理后端

---

## 故障排除

### FAISS 不可用

```bash
uv add langchain-community
```

### RAGFlow 连接失败

1. 检查 API 地址是否正确
2. 确认 RAGFlow 服务已启动
3. 验证 API 密钥是否有效

### 检索结果为空

1. 检查是否已添加文档到向量存储
2. 降低 `score_threshold` 值
3. 增加 `top_k` 数量

---

## 更多示例

完整的使用示例请参考 `examples.py` 文件。
