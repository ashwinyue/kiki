# 企业级 Agent 开发脚手架指南

> 基于 **LangGraph** 的企业级 Agent 开发脚手架，参考 **WeKnora99** 业务规范，采用 **deer-flow** 的 Python LangGraph 最佳实践。

---

## 设计原则

### 架构参考
| 参考项目 | 参考内容 | 应用方式 |
|---------|---------|---------|
| **WeKnora99** | 企业级 RAG 业务功能、事件驱动 Pipeline | 将事件映射为 LangGraph 节点，保持业务逻辑对齐 |
| **deer-flow** | Python + LangGraph 最佳实践 | StateGraph + MessagesState + Checkpoint 模式 |

### LangGraph 最佳实践
- **StateGraph** - 使用 `StateGraph` 定义工作流，替代传统的事件总线
- **MessagesState** - 继承 `MessagesState` 获得内置的消息管理能力
- **Reducer 函数** - 使用 `add_messages` reducer 自动管理消息追加
- **Conditional Edges** - 使用条件边实现动态路由，替代 if-else 分支
- **Checkpoint 持久化** - 使用 PostgreSQL Checkpointer 实现状态持久化
- **Command 模式** - 使用 `Command` 更新状态，保证不可变性

### WeKnora 事件到 LangGraph 节点的映射

| WeKnora 事件 | LangGraph 节点 | 说明 |
|-------------|---------------|------|
| `rewrite_query` | `query_rewrite_node` | 问题改写，结合会话历史 |
| `preprocess_query` | `query_preprocess_node` | 查询预处理、分词 |
| `chunk_search` | `vector_search_node` | 向量 + 关键词混合检索 |
| `chunk_rerank` | `rerank_node` | 检索结果重排序 |
| `chunk_merge` | `merge_node` | 相邻区块合并 |
| `filter_top_k` | `filter_node` | Top-K 过滤 |
| `into_chat_message` | `prompt_builder_node` | 组装提示词 |
| `chat_completion_stream` | `generation_node` | LLM 生成回答 |
| `stream_filter` | `stream_filter_node` | 流式输出过滤 |

---

## 项目概述

Kiki 是一个企业级 Agent 开发脚手架，提供：
- **LangGraph 原生工作流** - StateGraph + MessagesState + Conditional Edges
- **事件驱动的 RAG Pipeline** - 参考 WeKnora99 的完整检索生成流程
- **双模式 Agent** - Quick-Answer (RAG) 和 Smart-Reasoning (ReAct)
- **企业级特性** - 限流、认证、指标监控、可观测性
- **工具生态** - MCP 协议集成、Web 搜索、自定义工具
- **Checkpoint 持久化** - PostgreSQL Checkpointer 支持状态恢复

---

## 技术栈

### 核心依赖
```toml
# Web Framework
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.10.0
pydantic-settings>=2.6.0

# LangGraph & LangChain
langgraph>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langgraph-checkpoint-postgres>=2.0.0  # PostgreSQL 持久化

# Database
sqlmodel>=0.0.24
asyncpg>=0.30.0
pgvector>=0.3.0  # 向量搜索支持

# Observability
structlog>=25.1.0
prometheus-client>=0.21.0

# Auth
python-jose[cryptography]>=3.4.0
passlib[bcrypt]>=1.7.4
slowapi>=0.1.9

# Utils
python-dotenv>=1.0.0
tenacity>=9.1.0
```

---

## 项目目录结构

```
kiki/
├── app/                              # 主应用代码
│   ├── api/                          # API 路由层
│   │   ├── middleware.py             # API 中间件 (CORS, 请求上下文)
│   │   ├── dependencies.py           # FastAPI 依赖注入
│   │   └── v1/                       # API v1 路由
│   │       ├── agents.py             # Agent 管理
│   │       ├── api_keys.py           # API 密钥管理
│   │       ├── auth.py               # 认证授权
│   │       ├── chat.py               # 聊天接口
│   │       ├── evaluation.py         # 评估路由
│   │       ├── knowledge.py          # 知识库管理 (含知识条目)
│   │       ├── mcp_services.py       # MCP 服务管理
│   │       ├── messages.py           # 消息管理
│   │       ├── models.py             # 模型管理
│   │       ├── sessions.py           # 会话管理
│   │       ├── tenants.py            # 租户管理
│   │       └── tools.py              # 工具列表
│   ├── agent/                        # Agent 核心模块 (LangGraph)
│   │   ├── graphs/                   # LangGraph 工作流定义
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # 图构建基类
│   │   │   ├── builder.py            # 图构建器 (参考 deer-flow)
│   │   │   ├── state.py              # 状态定义 (MessagesState)
│   │   │   ├── nodes.py              # 节点实现
│   │   │   └── edges.py              # 条件边定义
│   │   ├── nodes/                    # RAG Pipeline 节点 (对齐 WeKnora)
│   │   │   ├── __init__.py
│   │   │   ├── query_rewrite_node.py     # 问题改写
│   │   │   ├── query_preprocess_node.py  # 查询预处理
│   │   │   ├── vector_search_node.py     # 混合检索 (向量+关键词)
│   │   │   ├── rerank_node.py            # 重排序
│   │   │   ├── merge_node.py             # 区块合并
│   │   │   ├── filter_node.py            # Top-K 过滤
│   │   │   ├── prompt_builder_node.py    # 提示词组装
│   │   │   ├── generation_node.py        # LLM 生成
│   │   │   └── stream_filter_node.py     # 流式过滤
│   │   ├── react/                    # ReAct Agent 节点
│   │   │   ├── __init__.py
│   │   │   ├── agent_node.py        # Agent 推理节点
│   │   │   ├── tool_node.py         # 工具执行节点
│   │   │   └── reflection_node.py   # 反思节点
│   │   ├── checkpoint/               # 检查点持久化
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py          # PostgreSQL Checkpointer
│   │   │   └── memory.py            # MemorySaver (开发用)
│   │   ├── state/                    # 状态管理
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # 基础状态 (MessagesState)
│   │   │   ├── rag.py               # RAG 状态
│   │   │   └── react.py             # ReAct 状态
│   │   ├── tools/                    # 工具系统
│   │   │   ├── builtin/              # 内置工具
│   │   │   │   ├── search.py         # 网络搜索
│   │   │   │   ├── tavily_search.py  # Tavily 搜索
│   │   │   │   ├── academic.py       # 学术搜索
│   │   │   │   ├── weather.py        # 天气查询
│   │   │   │   ├── calculation.py    # 计算器
│   │   │   │   ├── database.py       # 数据库查询
│   │   │   │   ├── python_repl.py    # Python REPL
│   │   │   │   └── crawl.py          # 网页爬取
│   │   │   ├── registry.py           # 工具注册表
│   │   │   ├── mcp.py                # MCP 工具适配
│   │   │   ├── interceptor.py        # 工具拦截器
│   │   │   ├── search_postprocessor.py  # 搜索结果后处理
│   │   │   └── decorators.py         # 工具装饰器
│   │   ├── prompts/                  # 提示词模板
│   │   │   ├── __init__.py
│   │   │   ├── chat.py               # 聊天提示词
│   │   │   ├── router.py             # 路由提示词
│   │   │   ├── supervisor.py         # 监督提示词
│   │   │   └── template.py           # 模板工具
│   │   ├── streaming/                # 流式输出
│   │   │   ├── __init__.py
│   │   │   ├── streaming.py          # SSE 流式处理
│   │   │   └── callback.py           # 流式回调
│   │   ├── factory.py                # Agent 工厂
│   │   └── schemas.py                # Agent 数据模式
│   ├── rag/                          # RAG 能力模块 (对齐 WeKnora)
│   │   ├── __init__.py
│   │   ├── retriever.py              # 检索器 (向量+关键词混合)
│   │   ├── reranker.py               # 重排序器
│   │   ├── vector_store.py           # 向量存储抽象
│   │   ├── stores/                   # 向量存储实现
│   │   │   ├── pgvector.py          # PGVector
│   │   │   ├── pinecone.py          # Pinecone
│   │   │   └── chroma.py            # Chroma
│   │   └── merger.py                 # 区块合并器
│   ├── auth/                         # 认证授权模块
│   │   ├── jwt.py                    # JWT 令牌处理
│   │   ├── api_key.py                # API 密钥验证
│   │   ├── tenant_api_key.py         # 租户 API 密钥
│   │   ├── tenant.py                 # 租户管理
│   │   └── middleware.py             # 认证中间件
│   ├── config/                       # 配置管理
│   │   ├── settings.py               # 应用配置 (Pydantic Settings)
│   │   ├── runtime.py                # 运行时配置
│   │   ├── dependencies.py           # 配置依赖
│   │   └── errors.py                 # 配置错误定义
│   ├── evaluation/                   # 评估模块
│   │   ├── runner.py                 # 评估运行器
│   │   ├── evaluators.py             # 评估器
│   │   ├── datasets.py               # 数据集管理
│   │   └── report.py                 # 报告生成
│   ├── infra/                        # 基础设施层
│   │   ├── database.py               # 数据库连接管理
│   │   ├── cache.py                  # 缓存抽象
│   │   ├── redis.py                  # Redis 实现
│   │   ├── storage.py                # 存储抽象
│   │   └── search.py                 # 搜索抽象
│   ├── llm/                          # LLM 服务模块
│   │   ├── service.py                # LLM 服务主类
│   │   ├── registry.py               # 模型注册表
│   │   ├── providers.py              # 多提供商适配
│   │   ├── embeddings.py             # 嵌入模型
│   │   └── cost_tracker.py           # 成本追踪
│   ├── models/                       # 数据模型 (SQLModel)
│   │   ├── database.py               # 数据库表定义
│   │   ├── agent.py                  # Agent 模型
│   │   ├── api_key.py                # API 密钥模型
│   │   └── ...
│   ├── observability/                # 可观测性模块
│   │   ├── logging.py                # 结构化日志 (structlog)
│   │   ├── metrics.py                # Prometheus 指标
│   │   ├── audit.py                  # 审计日志
│   │   ├── log_sanitizer.py          # 日志脱敏
│   │   └── elk_handler.py            # ELK 日志处理器
│   ├── rate_limit/                   # 限流模块
│   │   ├── limiter.py                # 速率限制器
│   │   └── token_bucket.py           # 令牌桶算法
│   ├── repositories/                 # 数据访问层
│   │   ├── base.py                   # 基础仓储 (分页等)
│   │   ├── user.py                   # 用户仓储
│   │   ├── tenant.py                 # 租户仓储
│   │   ├── session.py                # 会话仓储
│   │   ├── thread.py                 # 线程仓储
│   │   ├── message.py                # 消息仓储
│   │   ├── memory.py                 # 记忆仓储
│   │   ├── agent_async.py            # Agent 异步仓储
│   │   ├── mcp_service.py            # MCP 服务仓储
│   │   ├── api_key.py                # API 密钥仓储
│   │   ├── knowledge.py              # 知识仓储
│   │   └── model.py                  # 模型仓储
│   ├── schemas/                      # Pydantic 数据模式
│   │   ├── response.py               # 统一响应格式
│   │   ├── chat.py                   # 聊天模式
│   │   ├── agent.py                  # Agent 模式
│   │   ├── session.py                # 会话模式
│   │   ├── message.py                # 消息模式
│   │   ├── knowledge.py              # 知识库模式
│   │   ├── model.py                  # 模型模式
│   │   ├── evaluation.py             # 评估模式
│   │   └── ...
│   ├── services/                     # 业务服务层
│   │   ├── knowledge_service.py      # 知识库服务
│   │   ├── model_service.py          # 模型服务
│   │   ├── auth.py                   # 认证服务
│   │   ├── tenant.py                 # 租户服务
│   │   └── mcp_service_service.py    # MCP 服务
│   └── main.py                       # 应用入口
├── tests/                            # 测试套件
│   ├── unit/                         # 单元测试
│   │   ├── agent/                    # Agent 测试
│   │   │   ├── nodes/                # 节点测试
│   │   │   └── graphs/               # 图测试
│   │   └── ...
│   └── integration/                  # 集成测试
├── .env.example                      # 环境变量示例
├── AGENTS.md                         # 本文档
├── ENTERPRISE_GUIDE.md               # 企业级功能使用指南
└── pyproject.toml                    # 项目配置
```

---

## Agent 模式（对齐 WeKnora99）

Kiki 支持两种核心 Agent 模式，与 WeKnora99 完全对齐：

### 1. Quick-Answer 模式（RAG 模式）

适用于快速、准确的基于知识库的问答。

**特点：**
- 直接基于知识库检索结果生成回答
- 无复杂推理，响应速度快
- 支持混合检索（向量 + 关键词）
- 支持 FAQ 优先策略

**适用场景：**
- 文档问答
- FAQ 问答
- 知识库检索

```python
# 配置示例
agent_config = {
    "agent_mode": "quick-answer",
    "system_prompt": "你是一个专业的智能信息检索助手...",
    "context_template": "请根据以下参考资料回答用户问题...",
    "kb_selection_mode": "all",  # all | selected | none
    "knowledge_bases": [],  # 关联的知识库 ID
    "faq_priority_enabled": True,
    "faq_direct_answer_threshold": 0.9,
    "web_search_enabled": True,
}
```

### 2. Smart-Reasoning 模式（ReAct 模式）

适用于复杂任务、多步推理、工具调用场景。

**特点：**
- ReAct 推理框架
- 支持多步思考和工具调用
- 支持网络搜索、知识库检索、MCP 工具
- 支持反思（Reflection）机制

**适用场景：**
- 数据分析
- 复杂任务分解
- 多工具协作

```python
# 配置示例
agent_config = {
    "agent_mode": "smart-reasoning",
    "system_prompt": "你是一个专业的助手，可以思考和使用工具...",
    "max_iterations": 10,
    "allowed_tools": ["web_search", "knowledge_search"],  # 允许使用的工具
    "reflection_enabled": False,  # 是否启用反思
    "kb_selection_mode": "all",
    "web_search_enabled": True,
}
```

---

## 内置 Agent

系统默认提供以下内置 Agent（对齐 WeKnora99）：

| ID | 名称 | 描述 | 模式 |
|----|------|------|------|
| `builtin-quick-answer` | 快速问答 | 基于知识库的 RAG 问答，快速准确地回答问题 | quick-answer |
| `builtin-smart-reasoning` | 智能推理 | ReAct 推理框架，支持多步思考和工具调用 | smart-reasoning |
| `builtin-data-analyst` | 数据分析师 | 专业数据分析智能体，支持 CSV/Excel 文件的 SQL 查询与统计分析 | smart-reasoning |

---

## 聊天接口

Kiki 提供两种聊天接口，与 WeKnora99 对齐：

### 1. POST `/knowledge-chat/:session_id` - 知识库问答

基于知识库的快速问答，使用 Quick-Answer 模式。

**请求参数：**
```python
{
    "query": "彗尾的形状",  # 查询文本
    "knowledge_base_ids": ["kb-00000001"],  # 可选：指定知识库
}
```

**响应格式：** Server-Sent Events (SSE)

**响应类型：**
| response_type | 描述 |
|---------------|------|
| `references` | 知识库检索引用 |
| `answer` | 回答内容 |

### 2. POST `/agent-chat/:session_id` - Agent 智能问答

基于 Agent 的智能问答，支持 ReAct 推理和工具调用。

**请求参数：**
```python
{
    "query": "帮我查询今天的天气",  # 查询文本
    "agent_enabled": True,  # 是否启用 Agent 模式
    "agent_id": "builtin-smart-reasoning",  # 可选：指定 Agent ID
    "knowledge_base_ids": ["kb-00000001"],  # 可选：指定知识库
    "knowledge_ids": ["knowledge-001"],  # 可选：指定具体知识文件
    "web_search_enabled": True,  # 可选：启用网络搜索
    "mentioned_items": [  # 可选：@提及的知识库和文件
        {"id": "kb-00000001", "name": "天气知识库", "type": "kb", "kb_type": "document"}
    ],
}
```

**响应格式：** Server-Sent Events (SSE)

**响应类型：**
| response_type | 描述 |
|---------------|------|
| `thinking` | Agent 思考过程 |
| `tool_call` | 工具调用信息 |
| `tool_result` | 工具调用结果 |
| `references` | 知识库检索引用 |
| `answer` | 最终回答内容 |
| `reflection` | Agent 反思内容 |
| `error` | 错误信息 |

---

## 核心架构

### 1. RAG Pipeline (对齐 WeKnora 事件系统)

参考 WeKnora 的事件驱动设计，使用 LangGraph 节点实现完整的 RAG 流程：

```python
# app/agent/graphs/builder.py
from langgraph.graph import StateGraph, START, END
from app.agent.state import RAGState
from app.agent.checkpoint.postgres import create_postgres_checkpointer
from app.agent.nodes import (
    query_rewrite_node,
    query_preprocess_node,
    vector_search_node,
    rerank_node,
    merge_node,
    filter_node,
    prompt_builder_node,
    generation_node,
    stream_filter_node,
)

def build_rag_graph(checkpointer=None):
    """构建 RAG Pipeline 图

    对应 WeKnora 的事件流:
    rewrite_query → preprocess_query → chunk_search →
    chunk_rerank → chunk_merge → filter_top_k →
    into_chat_message → chat_completion_stream → stream_filter
    """
    builder = StateGraph(RAGState)

    # 添加节点
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("query_preprocess", query_preprocess_node)
    builder.add_node("vector_search", vector_search_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("merge", merge_node)
    builder.add_node("filter", filter_node)
    builder.add_node("prompt_builder", prompt_builder_node)
    builder.add_node("generation", generation_node)
    builder.add_node("stream_filter", stream_filter_node)

    # 添加边 (线性 Pipeline)
    builder.add_edge(START, "query_rewrite")
    builder.add_edge("query_rewrite", "query_preprocess")
    builder.add_edge("query_preprocess", "vector_search")
    builder.add_edge("vector_search", "rerank")
    builder.add_edge("rerank", "merge")
    builder.add_edge("merge", "filter")
    builder.add_edge("filter", "prompt_builder")
    builder.add_edge("prompt_builder", "generation")
    builder.add_edge("generation", "stream_filter")
    builder.add_edge("stream_filter", END)

    # 编译图 (带 Checkpoint 持久化)
    return builder.compile(
        checkpointer=checkpointer or create_postgres_checkpointer()
    )
```

### 2. 状态定义 (继承 MessagesState)

参考 deer-flow 的状态定义方式：

```python
# app/agent/state/rag.py
from dataclasses import dataclass, field
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

@dataclass
class RAGState(MessagesState):
    """RAG Pipeline 状态

    继承 MessagesState 获得内置的 messages 管理能力。
    使用 add_messages reducer 自动管理消息追加。
    """
    # MessagesState 自带 messages 字段，使用 add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # 查询相关
    original_query: str = ""          # 原始查询
    rewritten_query: str = ""         # 改写后查询
    preprocessed_query: str = ""      # 预处理后查询

    # 检索相关
    search_results: list[dict] = field(default_factory=list)    # 原始检索结果
    rerank_results: list[dict] = field(default_factory=list)    # 重排序结果
    merged_chunks: list[dict] = field(default_factory=list)     # 合并后的区块
    filtered_chunks: list[dict] = field(default_factory=list)   # 过滤后区块

    # 生成相关
    context: str = ""                 # 组装的上下文
    answer: str = ""                  # 生成的回答
    references: list[dict] = field(default_factory=list)       # 引用信息

    # 元数据
    session_id: str = ""
    user_id: str = ""
    knowledge_base_ids: list[str] = field(default_factory=list)
```

### 3. 节点实现示例

```python
# app/agent/nodes/query_rewrite_node.py
from app.observability.logging import get_logger
from app.llm.service import get_llm_service

log = get_logger(__name__)
llm_service = get_llm_service()

async def query_rewrite_node(state: RAGState) -> dict:
    """问题改写节点

    对应 WeKnora 的 rewrite_query 事件。
    结合会话历史改写用户问题。
    """
    log.info("query_rewrite_start",
             session_id=state.session_id,
             original_query=state.original_query)

    # 获取历史消息
    history = state.messages[-10:] if len(state.messages) > 10 else state.messages

    # 调用 LLM 改写
    rewritten = await llm_service.ainvoke([
        {"role": "system", "content": "你是一个查询改写专家..."},
        *history,
        {"role": "user", "content": f"改写这个问题: {state.original_query}"}
    ])

    log.info("query_rewrite_complete",
             session_id=state.session_id,
             rewritten_query=rewritten)

    return {"rewritten_query": rewritten}
```

```python
# app/agent/nodes/vector_search_node.py
from app.rag.retriever import hybrid_search
from app.observability.logging import get_logger

log = get_logger(__name__)

async def vector_search_node(state: RAGState) -> dict:
    """向量检索节点

    对应 WeKnora 的 chunk_search 事件。
    执行向量 + 关键词混合检索。
    """
    log.info("vector_search_start",
             session_id=state.session_id,
             query=state.preprocessed_query,
             knowledge_base_ids=state.knowledge_base_ids)

    # 混合检索 (向量 + 关键词)
    results = await hybrid_search(
        query=state.preprocessed_query,
        knowledge_base_ids=state.knowledge_base_ids,
        top_k=20,
    )

    log.info("vector_search_complete",
             session_id=state.session_id,
             result_count=len(results))

    return {"search_results": results}
```

### 4. ReAct Agent 图 (Smart-Reasoning 模式)

```python
# app/agent/graphs/react_builder.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from app.agent.state.react import ReactState
from app.agent.react.agent_node import agent_node
from app.agent.react.reflection_node import reflection_node

def build_react_graph(checkpointer=None, tools=None):
    """构建 ReAct Agent 图

    支持多步推理、工具调用、反思机制。
    """
    builder = StateGraph(ReactState)

    # 添加节点
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    if reflection_node:
        builder.add_node("reflect", reflection_node)

    # 入口
    builder.add_edge(START, "agent")

    # 条件边: agent -> tools 或 END
    def route_after_agent(state: ReactState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    builder.add_conditional_edges("agent", route_after_agent)

    # 工具执行后返回 agent
    builder.add_edge("tools", "agent")

    # 反思节点 (可选)
    if reflection_node:
        def route_after_reflect(state: ReactState) -> str:
            if state.get("should_continue", False):
                return "agent"
            return END

        builder.add_conditional_edges("reflect", route_after_reflect)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["reflect"]  # 支持人工干预
    )
```

### 5. Checkpoint 持久化 (PostgreSQL)

```python
# app/agent/checkpoint/postgres.py
from langgraph.checkpoint.postgres import PostgresSaver
from app.config.settings import get_settings

settings = get_settings()

def create_postgres_checkpointer():
    """创建 PostgreSQL Checkpointer

    参考 deer-flow 的 checkpoint.py 实现。
    """
    return PostgresSaver.from_conn_string(
        settings.database_url
    )

# 使用示例
checkpointer = create_postgres_checkpointer()
graph = build_rag_graph(checkpointer=checkpointer)

# 执行时指定 thread_id
config = {"configurable": {"thread_id": "session-123"}}
async for chunk in graph.astream({"messages": [("user", "你好")]}, config):
    print(chunk)
```

### 6. LLM 服务 (重试 + 结构化输出)

```python
# app/llm/service.py
from app.llm.service import get_llm_service, LLMService

llm_service = get_llm_service()

# 普通调用 (内置重试和回退)
response = await llm_service.ainvoke(messages)

# 结构化输出 (用于路由决策等场景)
from pydantic import BaseModel

class RouteDecision(BaseModel):
    agent: str = Field(description="目标 agent 名称")
    reason: str = Field(description="选择原因")
    confidence: float = Field(ge=0.0, le=1.0)

structured_llm = llm_service.with_structured_output(RouteDecision)
decision: RouteDecision = await structured_llm.ainvoke(messages)
```

### 7. 工具定义 (内置工具 + MCP 工具)

```python
# app/agent/tools/builtin/search.py
from langchain_core.tools import tool

@tool
async def search_web(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网络

    Args:
        query: 搜索查询
        max_results: 最大结果数
    """
    # 实现逻辑
    return results

# 绑定到 LLM
llm_with_tools = llm.bind_tools([search_web])
```

```python
# app/agent/tools/mcp.py - MCP 工具集成
from app.agent.tools.mcp import load_mcp_tools

# 加载 MCP 工具
mcp_tools = await load_mcp_tools()
agent.bind_tools(mcp_tools)
```

---

---

## 模型管理

Kiki 支持多种模型类型，与 WeKnora99 完全对齐：

### 模型类型

| 类型 | 说明 | 用途 |
|------|------|------|
| `KnowledgeQA` | 对话模型 | 知识库问答、对话生成 |
| `Embedding` | 嵌入模型 | 文本向量化、知识库检索 |
| `Rerank` | 排序模型 | 检索结果重排序、相关性优化 |
| `VLLM` | 视觉语言模型 | 多模态分析、图文理解 |

### 模型来源

| 来源 | 说明 | 配置要求 |
|------|------|----------|
| `local` | 本地模型 | 需要已安装 Ollama 并拉取模型 |
| `remote` | 远程 API | 需要提供 `base_url` 和 `api_key` |

### 支持的服务商

| 服务商 | 说明 | 支持的模型类型 |
|--------|------|----------------|
| `generic` | 自定义（OpenAI 兼容） | Chat, Embedding, Rerank, VLLM |
| `openai` | OpenAI | Chat, Embedding, Rerank, VLLM |
| `aliyun` | 阿里云 DashScope | Chat, Embedding, Rerank, VLLM |
| `zhipu` | 智谱 BigModel | Chat, Embedding, Rerank, VLLM |
| `deepseek` | DeepSeek | Chat |
| `jina` | Jina AI | Embedding, Rerank |
| `gemini` | Google Gemini | Chat |

### API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/models` | 创建模型 |
| GET | `/models` | 获取模型列表 |
| GET | `/models/:id` | 获取模型详情 |
| PUT | `/models/:id` | 更新模型 |
| DELETE | `/models/:id` | 删除模型 |
| GET | `/models/providers` | 获取模型服务商列表 |

---

## 知识库管理

### 知识库 API

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/knowledge-bases` | 创建知识库 |
| GET | `/knowledge-bases` | 获取知识库列表 |
| GET | `/knowledge-bases/:id` | 获取知识库详情 |
| PUT | `/knowledge-bases/:id` | 更新知识库 |
| DELETE | `/knowledge-bases/:id` | 删除知识库 |
| GET | `/knowledge-bases/:id/hybrid-search` | 混合搜索（向量+关键词） |

### 知识库配置

```python
{
    "name": "知识库名称",
    "description": "知识库描述",
    "chunking_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", "。", "！", "？"],
        "enable_multimodal": True
    },
    "embedding_model_id": "model-uuid",  # 嵌入模型
    "rerank_model_id": "model-uuid",     # 重排序模型
    "summary_model_id": "model-uuid",    # 摘要模型
    "vlm_config": {
        "enabled": True,
        "model_id": "model-uuid"  # 视觉语言模型
    }
}
```

### 知识条目管理 API

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/knowledge-bases/:id/knowledge/file` | 从文件创建知识 |
| POST | `/knowledge-bases/:id/knowledge/url` | 从 URL 创建知识 |
| POST | `/knowledge-bases/:id/knowledge/manual` | 创建手工 Markdown 知识 |
| GET | `/knowledge-bases/:id/knowledge` | 获取知识列表 |
| GET | `/knowledge/:id` | 获取知识详情 |
| PUT | `/knowledge/:id` | 更新知识 |
| DELETE | `/knowledge/:id` | 删除知识 |
| GET | `/knowledge/:id/download` | 下载知识文件 |

### 知识状态

| 状态 | 说明 |
|------|------|
| `pending` | 等待处理 |
| `processing` | 处理中 |
| `completed` | 处理完成 |
| `failed` | 处理失败 |

---

## Agent 配置参数

完整的 Agent 配置参数（对齐 WeKnora99）：

### 基础设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `agent_mode` | string | - | 智能体模式：`quick-answer`（RAG）或 `smart-reasoning`（ReAct） |
| `system_prompt` | string | - | 系统提示词，支持使用占位符 |
| `context_template` | string | - | 上下文模板（仅 quick-answer 模式使用） |

### 模型设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_id` | string | - | 对话模型 ID |
| `rerank_model_id` | string | - | 重排序模型 ID |
| `temperature` | float | 0.7 | 温度参数（0-1） |
| `max_completion_tokens` | int | 2048 | 最大生成 token 数 |

### Agent 模式设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_iterations` | int | 10 | ReAct 最大迭代次数 |
| `allowed_tools` | []string | - | 允许使用的工具列表 |
| `reflection_enabled` | bool | false | 是否启用反思 |
| `mcp_selection_mode` | string | - | MCP 服务选择模式：`all`/`selected`/`none` |
| `mcp_services` | []string | - | 选中的 MCP 服务 ID 列表 |

### 知识库设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kb_selection_mode` | string | - | 知识库选择模式：`all`/`selected`/`none` |
| `knowledge_bases` | []string | - | 关联的知识库 ID 列表 |
| `supported_file_types` | []string | - | 支持的文件类型（如 `["csv", "xlsx"]`） |

### FAQ 策略设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `faq_priority_enabled` | bool | true | FAQ 优先策略开关 |
| `faq_direct_answer_threshold` | float | 0.9 | FAQ 直接回答阈值 |
| `faq_score_boost` | float | 1.2 | FAQ 分数加成系数 |

### 网络搜索设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `web_search_enabled` | bool | true | 是否启用网络搜索 |
| `web_search_max_results` | int | 5 | 网络搜索最大结果数 |

### 多轮对话设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `multi_turn_enabled` | bool | true | 是否启用多轮对话 |
| `history_turns` | int | 5 | 保留的历史轮次数 |

### 检索策略设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_top_k` | int | 10 | 向量检索 TopK |
| `keyword_threshold` | float | 0.3 | 关键词检索阈值 |
| `vector_threshold` | float | 0.5 | 向量检索阈值 |
| `rerank_top_k` | int | 5 | 重排序 TopK |
| `rerank_threshold` | float | 0.5 | 重排序阈值 |

### 高级设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_query_expansion` | bool | true | 是否启用查询扩展 |
| `enable_rewrite` | bool | true | 是否启用多轮对话查询改写 |
| `rewrite_prompt_system` | string | - | 改写系统提示词 |
| `rewrite_prompt_user` | string | - | 改写用户提示词模板 |
| `fallback_strategy` | string | model | 回退策略：`fixed`（固定回复）或 `model`（模型生成） |
| `fallback_response` | string | - | 固定回退回复 |
| `fallback_prompt` | string | - | 回退提示词 |

---

## 多 Agent 协作模式（扩展）

### Router Agent (路由模式)

适用于意图分类、任务分发场景。

```python
from app.agent.multi_agent import create_multi_agent_system
from app.agent.graph import compile_chat_graph

# 创建专业 Agent
sales_agent = compile_chat_graph(llm_service=llm_service, system_prompt="销售专家...")
support_agent = compile_chat_graph(llm_service=llm_service, system_prompt="客服专家...")

# 创建路由系统
router_graph = create_multi_agent_system(
    mode="router",
    llm_service=llm_service,
    agents={
        "Sales": sales_agent,
        "Support": support_agent,
    },
)
```

### Supervisor Agent (监督模式)

适用于复杂任务分解、多步骤协作。

```python
from app.agent.multi_agent import create_multi_agent_system

supervisor_graph = create_multi_agent_system(
    mode="supervisor",
    llm_service=llm_service,
    workers={
        "Researcher": researcher,
        "Writer": writer,
        "Reviewer": reviewer,
    },
)
```

### Handoff Agent (Swarm 模式)

适用于动态协作、Agent 自主切换。

```python
from app.agent.multi_agent import create_multi_agent_system
from app.agent.multi_agent import HandoffAgent

alice = HandoffAgent(
    llm_service=llm_service,
    name="Alice",
    tools=[search_products],
)

bob = HandoffAgent(
    llm_service=llm_service,
    name="Bob",
    tools=[check_specifications],
)

handoff_graph = create_multi_agent_system(
    mode="swarm",
    agents=[alice, bob],
    default_agent="Alice",
)
```

> 详见 [MULTI_AGENT_EXAMPLES.md](./docs/MULTI_AGENT_EXAMPLES.md)

---

## 企业级功能

### 1. 速率限制

```python
# app/rate_limit/limiter.py
from app.rate_limit.limiter import limiter, RateLimit

@router.post("/chat")
@limiter.limit(RateLimit.CHAT)  # 30/min, 500/day
async def chat(message: str):
    return {"response": "..."}
```

### 2. JWT 认证

```python
# app/auth/jwt.py
from app.auth.jwt import create_access_token
from app.auth.dependencies import get_current_user_id

token = create_access_token(data={"sub": "user-123"})

@router.get("/protected")
async def protected(user_id: str = Depends(get_current_user_id)):
    return {"user_id": user_id}
```

### 3. Prometheus 指标

```python
# app/observability/metrics.py
from app.observability.metrics import track_llm_request, record_llm_tokens

async with track_llm_request(model="gpt-4o", provider="openai"):
    response = await llm.ainvoke(messages)

record_llm_tokens("gpt-4o", prompt_tokens=100, completion_tokens=50)
```

### 4. LangSmith 可观测性

```python
# app/agent/callbacks/handler.py
from app.agent.callbacks.handler import get_langsmith_callbacks, get_run_config

callbacks = get_langsmith_callbacks()

response = await graph.ainvoke(
    input_data,
    config=get_run_config(
        run_name="chat_session_123",
        metadata={"user_id": "user-123"},
    ),
    config={"callbacks": callbacks},
)
```

### 5. MCP 工具集成

```python
# app/agent/tools/mcp.py
from app.agent.tools.mcp import MCPRegistry, load_mcp_tools

# 注册 MCP 服务器
MCPRegistry.register(
    name="filesystem",
    command="uvx",
    args=["mcp-server-filesystem", "/allowed/path"],
)

# 加载所有 MCP 工具
mcp_tools = await load_mcp_tools()
agent.bind_tools(mcp_tools)
```

### 6. Web 搜索

```python
# app/agent/tools/builtin/search.py
from app.agent.tools.builtin.search import search_web

results = await search_web.invoke({"query": "最新 AI 新闻", "max_results": 5})
```

> 详见 [ENTERPRISE_GUIDE.md](./ENTERPRISE_GUIDE.md)

---

## 编码规范

### 日志 (structlog)

```python
# app/observability/logging.py
from app.observability.logging import get_logger

log = get_logger(__name__)

# 正确 - 使用键值对
log.info("chat_request_received", session_id=session.id, message_count=len(messages))

# 错误 - 禁止 f-string
log.info(f"chat_request_received {session.id}")

# 异常必须用 exception() 保留堆栈
log.exception("llm_call_failed", error=str(e))
```

### 重试 (tenacity)

```python
# app/agent/retry/retry.py
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def call_llm(messages: list) -> str:
    # LLM 调用逻辑
    ...
```

### 错误处理

```python
async def process_request(request: Request) -> Response:
    # 1. 前置验证（卫语句）
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # 2. 快乐路径
    try:
        result = await agent.process(request.messages)
        log.info("request_processed_successfully")
        return result
    except SpecificError as e:
        log.error("specific_error_occurred", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("unexpected_error")
        raise HTTPException(status_code=500, detail="Internal error")
```

---

## 数据库模型

```python
# app/models/database.py
from app.models.database import User, Session, Thread, Message, Agent, Model, KnowledgeBase, Knowledge

# 用户
user = User(
    email="user@example.com",
    full_name="John Doe",
)
user.set_password("secure_password")

# 会话 (避免与 SQLAlchemy.Session 冲突，命名为 ChatSession)
session = Session(id=str(uuid.uuid4()), user_id=user.id, name="新对话")

# 消息
message = Message(
    session_id=session.id,
    role="user",
    content="你好",
)

# Agent
agent = Agent(
    name="我的智能体",
    description="自定义智能体描述",
    config={
        "agent_mode": "smart-reasoning",
        "temperature": 0.7,
        "max_iterations": 10,
    },
)

# 模型
model = Model(
    name="text-embedding-v3",
    type="Embedding",  # KnowledgeQA | Embedding | Rerank | VLLM
    source="remote",   # local | remote
    parameters={
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-***",
        "provider": "aliyun",
    },
)

# 知识库
kb = KnowledgeBase(
    name="我的知识库",
    description="知识库描述",
    chunking_config={
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", "。", "！", "？"],
        "enable_multimodal": True,
    },
    embedding_model_id="model-uuid",
)

# 知识条目
knowledge = Knowledge(
    knowledge_base_id=kb.id,
    type="file",  # file | url | text | faq
    title="文档名称",
    parse_status="completed",  # pending | processing | completed | failed
    enable_status="enabled",    # enabled | disabled
)
```

---

## RAG 能力（对齐 WeKnora99）

Kiki 的 RAG 能力完全对齐 WeKnora99，支持多种向量存储后端：

### 向量存储支持

```python
# app/agent/capabilities/rag.py
from app.agent.capabilities.rag import (
    BaseVectorStore,
    VectorStoreType,
    create_vector_store,
    retrieve_documents,
    index_documents,
    VectorStoreConfig,
)

# 创建向量存储
pgvector_store = create_vector_store(
    store_type=VectorStoreType.PGVECTOR,
    config=VectorStoreConfig(
        collection_name="knowledge_base",
        embedding_model="text-embedding-v4",
        dimension=1024,
    ),
)

# Pinecone 云存储
pinecone_store = create_vector_store(
    store_type=VectorStoreType.PINECONE,
    api_key="your-api-key",
    environment="gcp-starter",
)

# Chroma 本地存储
chroma_store = create_vector_store(
    store_type=VectorStoreType.CHROMA,
    persist_directory="./chroma_db",
)
```

### 检索与索引

```python
# 索引文档
texts = ["文档1内容", "文档2内容"]
metadatas = [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
ids = await index_documents(texts, metadatas, store=pgvector_store)

# 检索文档
results = await retrieve_documents(
    query="用户查询内容",
    store=pgvector_store,
    k=5,
    score_threshold=0.5,
)

for result in results:
    print(f"分数: {result.score}")
    print(f"内容: {result.content}")
    print(f"来源: {result.metadata}")
```

### 作为 Agent 工具

```python
# 创建 RAG 检索工具
from langchain_core.tools import tool
from app.agent.capabilities.rag import BaseVectorStore

@tool
async def search_knowledge(query: str, k: int = 5) -> str:
    """搜索知识库

    Args:
        query: 搜索查询
        k: 返回结果数量
    """
    results = await vector_store.asimilarity_search(query, k=k)
    return "\n\n".join([r.content for r in results])

# 绑定到 Agent
llm_with_tools = llm.bind_tools([search_knowledge])
```

### 1. 问题改写 (rewrite_query)

结合会话历史，将用户问题改写得更具体。

```python
# 原问题: "入住的房型是什么"
# 改写后: "Liwx本次入住的房型是什么"
```

<think>` 标签）。

---

## 环境变量配置

```bash
# 应用配置
KIKI_APP_NAME=Kiki Agent
KIKI_ENVIRONMENT=development
KIKI_DEBUG=true

# 数据库配置
KIKI_DATABASE_URL=postgresql+asyncpg://localhost:5432/kiki

# LLM 配置
KIKI_LLM__PROVIDER=openai
KIKI_LLM__MODEL=gpt-4o
KIKI_LLM__API_KEY=your-api-key

# 认证配置
KIKI_SECRET_KEY=change-me-in-production-min-32-chars
KIKI_ACCESS_TOKEN_EXPIRE_MINUTES=30

# 可观测性配置
KIKI_LOG_LEVEL=INFO

# LangSmith (替代 Langfuse)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=kiki-agent
```

---

## 十大工程原则

1. **所有路由必须配置速率限制**
2. **所有 LLM 调用必须启用可观测性追踪**
3. **所有异步操作必须有完整错误处理**
4. **所有日志必须遵循结构化格式 (lowercase_underscore)**
5. **所有重试必须使用 tenacity 库**
6. **所有敏感配置必须通过环境变量管理**
7. **所有服务必须通过依赖注入解耦**
8. **所有检查点必须持久化到 PostgreSQL**
9. **Agent 状态必须使用 add_messages reducer**
10. **工具定义必须使用 @tool 装饰器**

---

## 参考资料

### 架构参考

| 项目 | 说明 | 链接 |
|------|------|------|
| **WeKnora99** | 腾讯开源的企业级 RAG 框架 (Go + Python)，提供完整的业务规范 | [WeKnora99](./WeKnora99/) |
| **deer-flow** | 字节跳动的多 Agent 研究框架，Python + LangGraph 最佳实践 | [deer-flow](./aold/deer-flow/) |

### WeKnora99 核心设计

- **事件驱动 Pipeline**: `rewrite_query → preprocess_query → chunk_search → chunk_rerank → chunk_merge → filter_top_k → into_chat_message → chat_completion_stream → stream_filter`
- **混合检索**: 向量检索 + 关键词检索，两次搜索结果合并
- **重排序**: 支持多种 Reranker (Normal / LLM-based / Layerwise)
- **FAQ 优先**: FAQ 匹配优先策略，支持直接回答阈值
- **多模态处理**: 文档解析、OCR、图像描述生成
- **事件总线**: 完整的事件监听和中间件系统

**关键文档**:
- [WeKnora 架构文档](./WeKnora99/docs/WeKnora.md) - Pipeline 流程详解
- [事件系统总结](./WeKnora99/internal/event/SUMMARY.md) - 事件总线设计

### deer-flow 核心设计

- **StateGraph 工作流**: 使用 `StateGraph` + `MessagesState` 定义状态和节点
- **Checkpoint 持久化**: `MemorySaver` / PostgreSQL Checkpointer
- **条件边**: `add_conditional_edges` 实现动态路由
- **工具拦截器**: 统一的工具调用拦截和日志
- **流式输出管理**: `ChatStreamManager` 处理流式消息持久化

**关键代码**:
- [Graph Builder](./aold/deer-flow/src/graph/builder.py) - 图构建模式
- [State 定义](./aold/deer-flow/src/graph/types.py) - MessagesState 继承
- [Checkpoint 管理](./aold/deer-flow/src/graph/checkpoint.py) - PostgreSQL 持久化

### 官方文档

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 官方文档](https://python.langchain.com/docs/)
- [FastAPI 最佳实践](https://fastapi.tiangolo.com/tutorial/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [LangSmith 文档](https://docs.smith.langchain.com/)

---

## 相关文档

- [API 对齐任务清单](./docs/API_ALIGNMENT_TODO.md) - Kiki 与 WeKnora99 API 对齐进度
- [企业级功能指南](./ENTERPRISE_GUIDE.md) - 企业级功能使用指南
- [多 Agent 示例](./docs/MULTI_AGENT_EXAMPLES.md) - 多 Agent 协作模式详解
- [项目结构说明](./CLAUDE.md) - Kiki 项目技术栈和结构说明

---

## 重构任务清单

### 阶段一: 图结构重构
- [ ] 创建 `app/agent/graphs/builder.py` - 参考 deer-flow
- [ ] 创建 `app/agent/state/` 目录 - 定义 RAGState、ReactState
- [ ] 创建 `app/agent/nodes/` 目录 - 实现 RAG Pipeline 各节点
- [ ] 实现 `app/agent/checkpoint/postgres.py` - PostgreSQL Checkpointer

### 阶段二: 节点实现
- [ ] `query_rewrite_node.py` - 问题改写
- [ ] `query_preprocess_node.py` - 查询预处理
- [ ] `vector_search_node.py` - 混合检索
- [ ] `rerank_node.py` - 重排序
- [ ] `merge_node.py` - 区块合并
- [ ] `filter_node.py` - Top-K 过滤
- [ ] `prompt_builder_node.py` - 提示词组装
- [ ] `generation_node.py` - LLM 生成
- [ ] `stream_filter_node.py` - 流式过滤

### 阶段三: ReAct 图实现
- [ ] 创建 `app/agent/react/` 目录
- [ ] 实现 `agent_node.py` - ReAct 推理节点
- [ ] 实现 `tool_node.py` - 工具执行节点
- [ ] 实现 `reflection_node.py` - 反思节点

### 阶段四: 测试和验证
- [ ] 单元测试 - 每个节点独立测试
- [ ] 集成测试 - 完整 Pipeline 测试
- [ ] 性能测试 - 对比 WeKnora 性能指标
