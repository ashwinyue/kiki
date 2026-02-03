# Kiki 企业级 Agent 脚手架功能补充建议

> 基于 `aold/ai-engineer-training2/` AI 工程师训练营项目分析

**生成日期**: 2026-02-03
**分析范围**: ai-engineer-training2/ 全栈 11 周课程内容

---

## 🔍 版本兼容性分析

| 包 | Kiki 版本 | 训练营版本 | 兼容性 |
|---|-----------|------------|--------|
| `langchain` | >= 1.2.7 | 0.3.27 | ⚠️ **API 差异** |
| `langchain-core` | >= 0.3.0 | 0.3.72 / 1.0.2 | ✅ 兼容 |
| `langgraph` | >= 0.3.0 | 0.6.4 | ✅ 兼容 |
| `langgraph-checkpoint-postgres` | >= 2.0.0 | 2.1.1 | ✅ 兼容 |

### ⚠️ LangChain 1.x 迁移注意事项

**1. 状态定义风格变化**

```python
# 训练营代码（旧风格，仍兼容）
from typing import TypedDict
class GenerationState(TypedDict):
    original_text: str
    chunks: list[str]

# Kiki 推荐风格（LangChain 1.x 最佳实践）
from typing import Annotated
from langgraph.graph import add_messages
class GenerationState(TypedDict):
    messages: Annotated[list, add_messages]
    original_text: str
    chunks: list[str]
```

**2. ChatModel 导入路径**

```python
# 训练营代码
from langchain_community.chat_models.tongyi import ChatTongyi

# Kiki 抽象（推荐）
from app.llm import get_llm_service
llm = get_llm_service().get_chat_model()
```

**3. 工具绑定方式**

```python
# 训练营代码（仍兼容）
model_with_tools = model.bind_tools([tool1, tool2])

# Kiki 推荐方式
from app.agent.tools import alist_tools
llm_with_tools = llm_service.get_llm_with_tools()
```

### ✅ 可直接复用的代码模式

以下模式在两个版本间完全兼容：

| 模式 | 训练营代码 | Kiki 兼容性 |
|------|-----------|-------------|
| `StateGraph` 构建 | `StateGraph(State)` | ✅ 直接兼容 |
| `ToolNode` | `ToolNode(tools)` | ✅ 直接兼容 |
| `START/END` | `from langgraph.graph import START, END` | ✅ 直接兼容 |
| 知识图谱 (NetworkX) | `networkx.MultiDiGraph` | ✅ 纯 Python，无依赖 |
| `@tool` 装饰器 | `@tool def my_tool()` | ✅ 直接兼容 |

### 📝 代码迁移清单

从训练营项目迁移代码时，需要注意：

1. **替换 LLM 初始化** → 使用 Kiki 的 `LLMService`
2. **替换配置获取** → 使用 `config["configurable"]`
3. **状态定义适配** → 遵循 Kiki 的 `app.agent.state` 模式
4. **工具注册** → 使用 Kiki 的工具注册系统

---

## 📊 Kiki 当前功能矩阵

| 模块 | 功能 | 状态 |
|------|------|------|
| **Agent 核心** | BaseAgent、ChatAgent、ReactAgent | ✅ 完整 |
| **多 Agent** | Supervisor、Router 模式 | ✅ 完整 |
| **记忆管理** | 短期记忆、长期记忆（基础版） | ⚠️ 缺知识图谱 |
| **重试机制** | 指数退避、策略配置 | ✅ 完整 |
| **工具系统** | 工具注册、MCP 集成、拦截器 | ✅ 完整 |
| **流式输出** | Token/事件流 | ✅ 完整 |
| **Human-in-the-Loop** | 中断、审批 | ✅ 完整 |
| **RAG** | ❌ **缺失** | 🔴 **核心缺失** |
| **工具监控** | ❌ **缺失** | 🔴 **核心缺失** |
| **缓存层** | ❌ 缺失 | 🟡 可选 |
| **ELK 日志** | ❌ 缺失 | 🟡 可选 |
| **Prometheus 监控** | ⚠️ 基础指标 | 🟡 可扩展 |

---

## 🎯 优先级分级

### P0 - 核心缺失功能（必须补充）

#### 1. 知识图谱记忆模块

**参考来源**: `aold/ai-engineer-training2/week07/p10-KnowledgeTripleMEM.py`

**缺失原因**: 当前长期记忆仅支持向量检索，知识图谱能提供更强的推理能力

**实现建议**:
```python
# app/agent/memory/knowledge_graph.py
from typing import Any

class KnowledgeGraphMemory(BaseLongTermMemory):
    """知识图谱记忆

    支持三元组 (Subject, Predicate, Object) 存储，
    支持实体关系推理和图谱遍历。
    """

    async def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加三元组"""

    async def search_entity(
        self,
        entity: str,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """搜索实体相关三元组"""

    async def get_neighbors(
        self,
        entity: str,
        direction: Literal["in", "out", "both"] = "both",
    ) -> list[str]:
        """获取相邻实体"""
```

**数据库**: 使用 Neo4j 或 PostgreSQL + `age` 扩展

---

#### 2. 完整 RAG 模块

**参考来源**: `aold/ai-engineer-training2/week03/`, `homework_examples/week03-homework-2/`

**缺失原因**: 企业级 Agent 必需知识库能力

**目录结构**:
```
app/rag/
├── __init__.py
├── retriever.py       # 检索器基类
├── vector_retriever.py # 向量检索
├── hybrid_retriever.py # 混合检索 (BM25 + 向量)
├── reranker.py        # 重排序
├── document.py        # 文档处理
├── chunker.py         # 智能切片
├── store.py           # 向量存储抽象
└── graph_rag.py       # GraphRAG 实现
```

**核心功能**:
- 向量数据库支持: Milvus、Qdrant、PGVector
- 混合检索: BM25 + 向量检索融合
- 智能切片: 语义感知的文档分块
- 重排序: Cohere Rerank 或 BGE Reranker

---

#### 3. 工具执行监控

**参考来源**: `aold/ai-engineer-training2/week08/ollama-exporter-main/`

**缺失原因**: 生产环境需要监控工具调用健康度

**实现建议**:
```python
# app/agent/observability/tool_monitor.py
from prometheus_client import Counter, Histogram

tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"]
)

tool_duration = Histogram(
    "tool_duration_seconds",
    "Tool execution duration",
    ["tool_name"]
)

class ToolMonitor:
    """工具执行监控"""

    async def monitor_execution(
        self,
        tool_name: str,
        coro: Coroutine,
    ) -> Any:
        """监控工具执行"""
```

---

### P1 - 高价值增强

#### 4. GraphRAG 实现

**参考来源**: `aold/ai-engineer-training2/homework_examples/week03-homework-2/graph_rag/`

**功能**: 结合知识图谱和向量检索，提供更好的知识推理

```python
# app/rag/graph_rag.py
class GraphRAGRetriever:
    """GraphRAG 检索器

    1. 向量检索获取候选文档
    2. 知识图谱扩展相关实体
    3. 图谱遍历发现隐式关联
    4. 融合排序返回结果
    """
```

---

#### 5. RAG 评估框架

**参考来源**: `aold/ai-engineer-training2/week03/code/P32-ragas.py`

**功能**: 使用 RAGAS 量化评估 RAG 系统质量

```python
# app/rag/evaluation.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

async def evaluate_rag_system(
    dataset: list[dict],
) -> dict[str, float]:
    """评估 RAG 系统性能

    返回指标:
    - faithfulness: 忠实度
    - answer_relevancy: 答案相关性
    - context_precision: 上下文精确度
    - context_recall: 上下文召回率
    """
```

---

#### 6. Redis 缓存层

**参考来源**: `aold/ai-engineer-training2/week09/3/p30缓存策略设计/`

**功能**: 减少 LLM 调用，提升响应速度

```python
# app/infra/cache.py
from redis.asyncio import Redis
from typing import Callable

class CacheManager:
    """缓存管理器"""

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: int = 3600,
    ) -> Any:
        """获取或计算缓存"""
```

---

#### 7. 多进程 + 协程混合架构

**参考来源**: `aold/ai-engineer-training2/week09/p21_多进程与协程混合/`

**功能**: 显著提升并发性能

```python
# app/concurrency/
├── __init__.py
├── scheduler.py      # 任务调度器
├── worker_pool.py    # 工作进程池
└── async_bridge.py   # 异步桥接
```

---

### P2 - 生产级工程化

#### 8. ELK 日志集成

**参考来源**: `aold/ai-engineer-training2/week08/p41elk.py`

```python
# app/observability/elk.py
import structlog
from elasticsearch import AsyncElasticsearch

class ElasticsearchHandler:
    """Elasticsearch 日志处理器"""

    async def emit(self, log_dict: dict) -> None:
        """发送日志到 Elasticsearch"""
```

---

#### 9. Prometheus 指标增强

**参考来源**: `aold/ai-engineer-training2/week08/prometheus-config/`

```python
# app/observability/prometheus.py
from prometheus_client import Counter, Gauge, Histogram, Info

# LLM 调用指标
llm_requests_total = Counter(...)
llm_tokens_total = Counter(...)
llm_latency = Histogram(...)

# Agent 指标
agent_iterations_total = Counter(...)
agent_errors_total = Counter(...)
```

---

#### 10. Celery 异步任务

**参考来源**: `aold/ai-engineer-training2/week08/p17_webLLM/celery_app.py`

```python
# app/tasks/
├── __init__.py
├── celery_app.py
└── handlers/
    ├── document.py
    └── rag_index.py
```

---

### P3 - 高级特性（可选增强）

#### 11. 模型微调支持

**参考来源**: `aold/ai-engineer-training2/projects/project2_2/`

- LoRA 微调: 参数高效微调
- 权重合并: merge_and_unload 优化
- 多维评估: ROUGE、BERTScore

---

#### 12. DSL 工作流引擎

**参考来源**: `aold/ai-engineer-training2/week06/p15-CoffeeDSL/`

- DSL 语法解析: Lark 解析器集成
- 动态规则修改
- SQL 生成 DSL: Vanna 自然语言转 SQL

---

## 📋 实施路线图

```
Phase 1 (P0) - 核心功能补齐
├── 知识图谱记忆模块
│   └── app/agent/memory/knowledge_graph.py
├── 完整 RAG 模块
│   └── app/rag/
└── 工具执行监控
    └── app/agent/observability/tool_monitor.py

Phase 2 (P1) - 能力增强
├── GraphRAG
│   └── app/rag/graph_rag.py
├── RAG 评估
│   └── app/rag/evaluation.py
├── Redis 缓存层
│   └── app/infra/cache.py
└── 多进程 + 协程架构
    └── app/concurrency/

Phase 3 (P2) - 生产级工程化
├── ELK 日志集成
│   └── app/observability/elk.py
├── Prometheus 监控增强
│   └── app/observability/prometheus.py
└── Celery 异步任务
    └── app/tasks/
```

---

## 📁 参考代码路径映射

| 功能 | 参考路径 | 说明 |
|------|----------|------|
| **知识图谱记忆** | `aold/ai-engineer-training2/week07/p10-KnowledgeTripleMEM.py` | 完整的知识图谱管理器 |
| **GraphRAG** | `aold/ai-engineer-training2/homework_examples/week03-homework-2/graph_rag/` | Neo4j + 向量检索融合 |
| **混合检索** | `aold/ai-engineer-training2/week03/code/P35-es混合检索的典型demo.ipynb` | BM25 + 向量检索 |
| **RAG 评估** | `aold/ai-engineer-training2/week03/code/P32-ragas.py` | RAGAS 框架集成 |
| **工具重试** | `aold/ai-engineer-training2/week07/p13-toolRetry.py` | 指数退避重试策略 |
| **多进程架构** | `aold/ai-engineer-training2/week09/p21_多进程与协程混合/` | 混合并发架构 |
| **ELK 日志** | `aold/ai-engineer-training2/week08/p41elk.py` | Logstash + ES 集成 |
| **Prometheus** | `aold/ai-engineer-training2/week08/ollama-exporter-main/ollama_exporter.py` | Ollama 指标导出器 |
| **Celery** | `aold/ai-engineer-training2/week08/p17_webLLM/celery_app.py` | 异步任务队列 |
| **Docker 部署** | `aold/ai-engineer-training2/week08/docker/` | Docker Compose 配置 |
| **Kubernetes** | `aold/ai-engineer-training2/week08/p18_k8s/` | K8s 部署配置 |
| **模型微调** | `aold/ai-engineer-training2/projects/project2_2/` | LoRA 微调完整实现 |

---

## 🔍 训练营项目结构概览

```
ai-engineer-training2/
├── week01/          # LLM 基础与 LangGraph 入门
├── week02/          # 模型微调
├── week03/          # LlamaIndex 与 RAG
├── week04/          # LangChain 学习
├── week05/          # 多 Agent 协作
├── week06/          # DSL 语言设计
├── week07/          # 智能 Agent 高级能力
├── week08/          # 工程化部署与监控
├── week09/          # Python 高性能并发
├── week10/          # 综合实战项目
├── week11-homework/ # 狼人杀游戏系统
├── homework_examples/ # 优秀作业示例
└── projects/        # 综合项目 (project1_1 ~ project5_2)
```

---

## 📝 技术栈对比

| 层级 | Kiki 当前 | 训练营推荐 | 建议 |
|------|-----------|------------|------|
| **Agent 框架** | LangGraph | LangGraph | ✅ 保持 |
| **向量数据库** | ❌ 缺失 | Milvus, FAISS | 🔴 需补充 |
| **知识图谱** | ❌ 缺失 | Neo4j | 🔴 需补充 |
| **日志** | structlog | ELK | 🟡 可增强 |
| **监控** | 基础指标 | Prometheus + Grafana | 🟡 可增强 |
| **异步任务** | ❌ 缺失 | Celery | 🟡 可补充 |
| **并发** | asyncio | 多进程 + 协程 | 🟡 可优化 |
| **部署** | ❌ 缺失 | Docker + K8s | 🟡 可补充 |

---

## 📖 LangChain 1.x 代码适配指南

### 知识图谱记忆模块（适配 Kiki）

```python
# app/agent/memory/knowledge_graph.py
"""基于训练营代码适配的知识图谱记忆模块"""

from __future__ import annotations

import json
import pickle
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.tools import tool

from app.agent.memory.base import BaseLongTermMemory
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeNode:
    """知识节点"""
    label: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KnowledgeTriple:
    """知识三元组 (Subject, Predicate, Object)"""
    subject: str
    predicate: str
    object: str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeGraphMemory(BaseLongTermMemory):
    """知识图谱记忆 - Kiki 适配版

    基于训练营 week07/p10-KnowledgeTripleMEM.py
    适配点:
    - 使用 BaseLongTermMemory 接口
    - 集成 Kiki 日志系统
    - 支持依赖注入
    """

    def __init__(
        self,
        storage_path: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """初始化知识图谱记忆

        Args:
            storage_path: 存储路径
            user_id: 用户 ID（多租户隔离）
        """
        self.storage_path = Path(storage_path or "data/knowledge_graph")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.user_id = user_id or "default"

        # 数据结构
        self.nodes: dict[str, KnowledgeNode] = {}
        self.triples: list[KnowledgeTriple] = []
        self.nx_graph = nx.MultiDiGraph()

        # 索引
        self._node_label_index: defaultdict[str, set[str]] = defaultdict(set)

        self._load_or_create()

    async def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加记忆（三元组形式）

        Args:
            content: 三元组 JSON，格式 {"subject": "...", "predicate": "...", "object": "..."}
            metadata: 额外元数据

        Returns:
            记忆 ID
        """
        try:
            data = json.loads(content) if isinstance(content, str) else content
            triple = KnowledgeTriple(
                subject=data["subject"],
                predicate=data["predicate"],
                object=data["object"],
                properties=metadata or {},
            )
            self.triples.append(triple)
            self._add_triple_to_graph(triple)
            self._save()
            logger.info(
                "knowledge_triple_added",
                subject=triple.subject,
                predicate=triple.predicate,
                object=triple.object,
            )
            return str(uuid.uuid4())

        except (KeyError, json.JSONDecodeError) as e:
            logger.error("invalid_triple_format", error=str(e))
            raise ValueError(f"无效的三元组格式: {e}") from e

    async def search_memories(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """搜索记忆

        Args:
            query: 查询文本（实体名称）
            k: 返回数量
            filter: 元数据过滤

        Returns:
            匹配的三元组列表
        """
        results = []

        # 按主体搜索
        for triple in self.triples:
            if query.lower() in triple.subject.lower():
                results.append({
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "confidence": triple.confidence,
                })
                if len(results) >= k:
                    break

        logger.debug("knowledge_search_completed", query=query, results=len(results))
        return results

    def _add_triple_to_graph(self, triple: KnowledgeTriple) -> None:
        """添加三元组到图"""
        # 确保节点存在
        self._ensure_node(triple.subject)
        self._ensure_node(triple.object)

        # 添加边
        source_id = self._get_node_id(triple.subject)
        target_id = self._get_node_id(triple.object)

        self.nx_graph.add_edge(
            source_id,
            target_id,
            relation=triple.predicate,
            confidence=triple.confidence,
        )

    def _ensure_node(self, label: str) -> None:
        """确保节点存在"""
        if label not in self._node_label_index:
            node = KnowledgeNode(label=label, type="entity")
            self.nodes[node.id] = node
            self._node_label_index[label.lower()].add(node.id)
            self.nx_graph.add_node(node.id, label=label)

    def _get_node_id(self, label: str) -> str | None:
        """获取节点 ID"""
        node_ids = self._node_label_index.get(label.lower(), set())
        return next(iter(node_ids)) if node_ids else None

    def _load_or_create(self) -> None:
        """加载或创建存储"""
        data_file = self.storage_path / "graph_data.json"

        if data_file.exists():
            self._load()
        else:
            logger.info("creating_new_knowledge_graph", path=str(self.storage_path))

    def _load(self) -> None:
        """加载现有数据"""
        data_file = self.storage_path / "graph_data.json"

        try:
            with data_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for triple_data in data.get("triples", []):
                triple = KnowledgeTriple(**triple_data)
                self.triples.append(triple)
                self._add_triple_to_graph(triple)

            logger.info("knowledge_graph_loaded", triples=len(self.triples))

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("knowledge_graph_load_failed", error=str(e))

    def _save(self) -> None:
        """保存数据"""
        data_file = self.storage_path / "graph_data.json"

        data = {
            "triples": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence,
                }
                for t in self.triples
            ],
        }

        with data_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============== 工具函数（兼容 Kiki 工具系统）=============


def create_knowledge_graph_tools(memory: KnowledgeGraphMemory):
    """创建知识图谱工具集

    Args:
        memory: 知识图谱记忆实例

    Returns:
        工具列表
    """

    @tool
    async def add_knowledge_triple(
        subject: str,
        predicate: str,
        obj: str,
    ) -> str:
        """添加知识三元组

        Args:
            subject: 主体
            predicate: 谓词/关系
            obj: 客体

        Returns:
            操作结果
        """
        await memory.add_memory(
            content=json.dumps({"subject": subject, "predicate": predicate, "object": obj})
        )
        return f"✓ 已添加三元组: ({subject}, {predicate}, {obj})"

    @tool
    async def search_knowledge(entity: str) -> str:
        """搜索知识图谱中的实体

        Args:
            entity: 实体名称

        Returns:
            相关三元组
        """
        results = await memory.search_memories(query=entity, k=5)

        if not results:
            return f"未找到关于 '{entity}' 的知识"

        lines = [f"关于 '{entity}' 的知识:"]
        for r in results:
            lines.append(f"  • {r['subject']} --{r['predicate']}--> {r['object']}")

        return "\n".join(lines)

    return [add_knowledge_triple, search_knowledge]
```

### RAG 模块基础框架

```python
# app/rag/__init__.py
"""RAG 模块 - Kiki 适配版"""

from .retriever import BaseRetriever, VectorRetriever, HybridRetriever
from .store import VectorStore, create_vector_store
from .chunker import DocumentChunker, SemanticChunker

__all__ = [
    "BaseRetriever",
    "VectorRetriever",
    "HybridRetriever",
    "VectorStore",
    "create_vector_store",
    "DocumentChunker",
    "SemanticChunker",
]


# app/rag/store.py
"""向量存储抽象"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档"""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """相似度搜索"""


def create_vector_store(
    store_type: str = "faiss",
    embeddings: Embeddings | None = None,
    **kwargs: Any,
) -> VectorStore:
    """创建向量存储实例

    Args:
        store_type: 存储类型 (faiss, milvus, pgvector)
        embeddings: 嵌入模型
        **kwargs: 额外参数

    Returns:
        向量存储实例
    """
    if store_type == "faiss":
        from .stores.faiss_store import FAISSVectorStore

        return FAISSVectorStore(embeddings=embeddings, **kwargs)

    # 其他存储类型...
    raise ValueError(f"Unknown store type: {store_type}")


# app/rag/retriever.py
"""检索器实现"""

from .store import VectorStore


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    async def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """检索文档"""


class VectorRetriever(BaseRetriever):
    """向量检索器"""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    async def retrieve(self, query: str, k: int = 4) -> list[Document]:
        return await self.vector_store.similarity_search(query, k=k)


class HybridRetriever(BaseRetriever):
    """混合检索器 (BM25 + 向量)"""

    def __init__(
        self,
        vector_store: VectorStore,
        alpha: float = 0.5,  # 融合权重
    ) -> None:
        self.vector_store = vector_store
        self.alpha = alpha

    async def retrieve(self, query: str, k: int = 4) -> list[Document]:
        # 先做向量检索
        vector_results = await self.vector_store.similarity_search(query, k=k * 2)

        # TODO: 加入 BM25 分数融合
        return vector_results[:k]
```

---

## 总结

Kiki 项目已具备完整的 Agent 核心框架，主要缺失 **RAG 能力**和**知识图谱记忆**两个核心企业级功能。

### 关键适配要点

1. **版本兼容**: 训练营项目使用 LangChain 0.3.x / 1.x，与 Kiki 基本兼容
2. **LLM 抽象**: 使用 Kiki 的 `LLMService` 而非直接初始化模型
3. **状态定义**: 遵循 Kiki 的 `app.agent.state` 模式
4. **工具注册**: 使用 Kiki 的工具注册系统
5. **日志集成**: 使用 `app.observability.logging.get_logger()`

建议按 P0 → P1 → P2 优先级分阶段实施，优先补齐核心缺失功能，再逐步增强工程化能力。
