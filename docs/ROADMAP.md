# Kiki Agent Framework - 路线图

> 版本: v0.1.0
> 更新日期: 2026-01-31
> 基于 AI 工程师训练营参考实现

---

## 版本规划

| 版本 | 状态 | 预计时间 | 核心功能 |
|------|------|----------|----------|
| v0.1.0 | ✅ 已完成 | - | 基础框架、P0 模块 |
| v0.2.0 | 🚧 规划中 | 1-2 周 | P1 记忆增强、可观测性 |
| v0.3.0 | 📋 待规划 | 2-3 周 | P2 多轮研究、知识库 |
| v0.4.0 | 📋 待规划 | 4-5 周 | P3 高级特性 |

---

## v0.2.0 - 记忆增强与可观测性

### P1-1: Summary Memory（总结记忆）

**参考实现：** `week07/p06-summaryMEM.py`

**功能描述：**
- 使用 `langmem` 自动压缩对话历史
- 当 Token 超过阈值时自动触发总结
- 保留对话的关键信息，丢弃冗余内容

**工作量：** 2-3 天

**文件：** `app/agent/memory/summary.py`

**依赖：**
```bash
uv add langmem
```

**实现要点：**
```python
from langmem import create_memory_store

# 创建总结记忆
store = create_memory_store(
    summarization_model="gpt-4o-mini",
    summarization_threshold=1000,  # Token 阈值
)

# 自动总结
summary = await store.summarize(messages)
```

---

### P1-2: Vector Memory（向量记忆）

**参考实现：** `week07/p08-vectorMEM.py`

**功能描述：**
- 使用 Embedding 将消息转向量化
- 语义搜索历史对话
- 支持多种向量存储（pgvector、pinecone、chroma）

**工作量：** 2-3 天

**文件：** `app/agent/memory/vector.py`

**实现要点：**
```python
from langchain_core.vectorstores import InMemoryVectorStore
from app.llm.embeddings import get_embeddings

# 创建向量记忆
embeddings = get_embeddings()
vector_store = InMemoryVectorStore(embeddings)

# 语义搜索
results = await vector_store.asimilarity_search(
    "用户之前问过什么关于天气的问题？",
    k=3
)
```

---

### P1-3: Prometheus 指标

**参考实现：** `week08/prometheus/`

**功能描述：**
- Agent 执行耗时 (P50/P95 分位数)
- LLM 请求速率和错误率
- 工具调用统计
- 内存使用情况

**工作量：** 2-3 天

**文件：** `app/observability/metrics.py`

**实现要点：**
```python
from prometheus_client import Counter, Histogram

# 定义指标
agent_duration = Histogram(
    'agent_duration_seconds',
    'Agent 执行耗时',
    ['agent_type', 'status']
)

llm_requests = Counter(
    'llm_requests_total',
    'LLM 请求总数',
    ['model', 'status']
)
```

---

## v0.3.0 - 高级 Agent 能力

### P2-1: Multi-Round Research Agent（多轮研究 Agent）

**参考实现：** `week07/p24-multiRoundRESEARCH.py`

**功能描述：**
- 分阶段执行复杂研究任务
- 每个阶段独立思考和验证
- 支持阶段回溯和修正

**工作量：** 3-5 天

**文件：** `app/agent/graphs/research.py`

**状态结构：**
```python
class ResearchState(TypedDict):
    phase: Literal["planning", "research", "synthesis", "review"]
    query: str
    findings: list[dict]
    synthesis: str
    current_iteration: int
```

---

### P2-2: Knowledge Base Manager（知识库管理器）

**参考实现：** `week07/kb_manager.py`

**功能描述：**
- FAQ 管理和向量化索引
- 知识库 CRUD 操作
- 与 RAG 集成

**工作量：** 3-5 天

**文件：** `app/agent/knowledge_base.py`

---

## v0.4.0 - 高级特性

### P3-1: FAISS Memory（高性能向量记忆）

**参考实现：** `week07/p09-faissMEM.py`

**功能描述：**
- 使用 FAISS 进行高性能向量搜索
- 持久化向量存储
- 用户隔离的记忆管理

**工作量：** 3-4 天

**文件：** `app/agent/memory/faiss.py`

**依赖：**
```bash
uv add faiss-cpu
# 或 GPU 版本
uv add faiss-gpu
```

---

### P3-2: Knowledge Triple Memory（知识三元组记忆）

**参考实现：** `week07/p10-KnowledgeTripleMEM.py`

**功能描述：**
- 结构化知识存储（实体-关系-实体）
- 基于 NetworkX 的图查询
- 路径搜索和关系推理

**工作量：** 4-5 天

**文件：** `app/agent/memory/knowledge_triple.py`

**依赖：**
```bash
uv add networkx
```

---

### P3-3: CLIP Image Search（图像搜索）

**参考实现：** `week07/standalone_projects/p25-CLIP/`

**功能描述：**
- 使用 CLIP 模型进行图文嵌入
- 图像相似度搜索
- Milvus 向量数据库集成

**工作量：** 3-4 天

**文件：** `app/agent/multimodal/clip.py`

**依赖：**
```bash
uv add clip-by-openai
uv add pymilvus
```

---

### P3-4: Hybrid Task Scheduler（混合任务调度器）

**参考实现：** `week09/p21_多进程与协程混合/`

**功能描述：**
- 多进程（CPU 密集型）+ 协程（IO 密集型）混合架构
- 智能任务分发
- 进程池管理

**工作量：** 4-5 天

**文件：** `app/core/scheduler.py`

---

### P3-5: Q-Learning 优化

**参考实现：** `week07/qlearn-4.py`

**功能描述：**
- 基于 Q-learning 的 Agent 行为优化
- 奖励机制设计
- 策略迭代和收敛

**工作量：** 5-7 天

**文件：** `app/agent/optimization/q_learning.py`

---

## 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                      v0.2.0 (P1)                            │
├─────────────────────────────────────────────────────────────┤
│  Summary Memory ─────┐                                      │
│                      ├──→ Combined Memory ───→ Enhanced Agent│
│  Vector Memory ──────┘                                      │
│                                                             │
│  Prometheus Metrics ─────────────────────────────────→ 监控   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      v0.3.0 (P2)                            │
├─────────────────────────────────────────────────────────────┤
│  Multi-Round Research ───→ Complex Tasks                    │
│  Knowledge Base Manager ─→ RAG Integration                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      v0.4.0 (P3)                            │
├─────────────────────────────────────────────────────────────┤
│  Q-Learning ───────────────→ Agent Optimization             │
│  Knowledge Graph ──────────→ Semantic Memory                │
└─────────────────────────────────────────────────────────────┘
```

---

## 已完成模块 (v0.1.0)

### P0-1: Window Memory（窗口记忆）✅

**文件：** `app/agent/memory/window.py`

- ✅ Token 级别限制
- ✅ 多种修剪策略
- ✅ pre_model_hook 钩子

---

### P0-2: Tool Retry（工具重试）✅

**文件：** `app/agent/retry.py`

- ✅ 可配置重试策略
- ✅ 指数退避算法
- ✅ 自定义异常类型

---

### 可观测性模块 ✅

| 模块 | 文件 | 状态 |
|------|------|------|
| ELK 日志 | `app/observability/elk_handler.py` | ✅ |
| 令牌桶限流 | `app/core/token_bucket.py` | ✅ |
| Redis 缓存 | `app/infra/cache.py` | ✅ |
| WebSocket 流式 | 已移除（仅保留 LangGraph SSE） | ⛔ |

---

## 技术债务

| 优先级 | 项目 | 预计工作量 |
|--------|------|------------|
| 高 | 单元测试覆盖率达到 80% | 3-5 天 |
| 高 | 集成测试完善 | 2-3 天 |
| 中 | API 文档自动生成 | 1-2 天 |
| 中 | 性能基准测试 | 2-3 天 |
| 低 | Docker 镜像优化 | 1 天 |

---

## 参考资料索引

| 模块 | 参考文件路径 |
|------|-------------|
| Window Memory | `aold/ai-engineer-training2/week07/p07-windowMEM.py` |
| Tool Retry | `aold/ai-engineer-training2/week07/p13-toolRetry.py` |
| Summary Memory | `aold/ai-engineer-training2/week07/p06-summaryMEM.py` |
| Vector Memory | `aold/ai-engineer-training2/week07/p08-vectorMEM.py` |
| FAISS Memory | `aold/ai-engineer-training2/week07/p09-faissMEM.py` |
| Knowledge Triple | `aold/ai-engineer-training2/week07/p10-KnowledgeTripleMEM.py` |
| Redis Memory | `aold/ai-engineer-training2/week07/p11-redisMEM.py` |
| Multi-Round Research | `aold/ai-engineer-training2/week07/p24-multiRoundRESEARCH.py` |
| Q-Learning | `aold/ai-engineer-training2/week07/qlearn-4.py` |
| Knowledge Base | `aold/ai-engineer-training2/week07/kb_manager.py` |
| CLIP Image Search | `aold/ai-engineer-training2/week07/standalone_projects/p25-CLIP/` |
| ELK Handler | `aold/ai-engineer-training2/week08/p41elk.py` |
| Token Bucket | `aold/ai-engineer-training2/week09/3/p29限流中间件.py` |
| Enhanced Cache | `aold/ai-engineer-training2/week09/3/p30缓存策略.py` |
| WebSocket | `aold/ai-engineer-training2/week09/3/p26WebSocket.py` |
| Hybrid Scheduler | `aold/ai-engineer-training2/week09/p21_多进程与协程混合/` |

---

## 贡献指南

1. 选择一个未完成的模块
2. 阅读参考实现文件
3. 创建功能分支 `feature/module-name`
4. 实现并添加测试
5. 更新本路线图

---

最后更新：2026-01-31
