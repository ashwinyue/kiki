# Kiki 与 FastAPI-LangGraph-Chatbot 架构对比分析

> **生成时间**: 2026-02-03
> **对比项目**: [fastapi-langgraph-chatbot-with-vector-store-memory-mcp-tools-and-voice-mode](../a/fastapi-langgraph-chatbot-with-vector-store-memory-mcp-tools-and-voice-mode/)
> **分析目的**: 识别可借鉴的设计模式，为 Kiki 项目提供改进建议

---

## 📊 核心差异概览

| 维度 | 外部项目 (Chatbot) | Kiki 项目 | 评价 |
|------|-------------------|-----------|------|
| **代码规模** | ~4,000 行 | ~18,000 行 (agent 模块) | Kiki 更大型、更模块化 |
| **架构风格** | 简洁单体 | 高度模块化 | 各有优势 |
| **状态管理** | 全局单例 | 依赖注入 + 工厂 | Kiki 更企业化 |
| **多 Agent** | Supervisor 协作 | 单 Agent ReAct | 外部更灵活 |
| **内存系统** | Mem0 + Qdrant 双层 | 统一 MemoryManager | 外部更专业 |
| **依赖注入** | 链式 Depends | 分散在各文件 | 需要统一 |
| **可观测性** | 基础日志 | Langfuse + Prometheus | Kiki 更完善 |

---

## 💡 值得借鉴的设计模式

### 1️⃣ 全局单例 + 懒初始化（资源密集型服务）

**外部项目实现**：

```python
# MultiTenantVectorStore - Qdrant 客户端单例
class MultiTenantVectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MultiTenantVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, collection_name: str = "multi_tenant_chat_history"):
        if self._initialized:
            return  # 防止重复初始化
        # 仅初始化一次
        self.client = QdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self._initialized = True
```

**优点**：
- 确保 Qdrant 客户端只创建一次（节省连接资源）
- 线程安全的单例实现
- 延迟初始化（首次调用时才创建）

**Kiki 改进点**：
```python
# 建议在 app/db/session.py 或 app/agent/memory/store.py 添加
class QdrantClientSingleton:
    """Qdrant 客户端单例"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from qdrant_client import QdrantClient
        from app.config.settings import get_settings

        settings = get_settings()
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._initialized = True
```

---

### 2️⃣ Supervisor-Agent 协作模式（多 Agent 编排）

**外部项目实现**：

```python
async def supervisor_agent(state: AgentState) -> Dict:
    """Supervisor agent that decides which agent to use next."""
    members = ["Researcher", "Scrapper"]
    options = members + ["FINISH"]

    # 分析对话历史，决定路由
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    result = await supervisor_chain.ainvoke(state)

    return {
        "next": result.next,  # "Researcher" | "Scrapper" | "FINISH"
        "task_completed": result.next == "FINISH"
    }

# 图结构：所有 agent → Supervisor → 决定下一步
workflow.add_node("Researcher", research_node)
workflow.add_node("Scrapper", scrapper_node)
workflow.add_node("Supervisor", supervisor_agent)

for member in members:
    workflow.add_edge(member, "Supervisor")

workflow.add_conditional_edges("Supervisor", lambda x: x["next"], {
    "Researcher": "Researcher",
    "Scrapper": "Scrapper",
    "FINISH": END
})
```

**优点**：
- 动态任务路由（Supervisor 根据上下文决定调用哪个 Agent）
- 迭代控制（`max_iterations` 防止无限循环）
- 结构化输出（`RouteResponse` 确保路由决策可解析）

**Kiki 当前状态**：
- 使用 `route_by_tools` 决定是否调用工具节点
- 缺少多 Agent 协作机制

**Kiki 改进建议**：

```python
# app/agent/graph/supervisor.py (新文件)
from typing import Literal
from langchain_core.messages import SystemMessage
from app.agent.state import AgentState

class RouteResponse(BaseModel):
    """Supervisor 路由决策"""
    next: Literal["Researcher", "Scrapper", "Database", "FINISH"]
    reasoning: str

async def supervisor_node(state: AgentState) -> Dict:
    """Supervisor 节点 - 决定调用哪个 Agent"""
    llm = get_llm_service()

    prompt = f"""
    你是任务协调者。根据用户需求，决定调用哪个专家 Agent：

    - Researcher: 网络搜索、学术查询
    - Scrapper: 网页抓取、数据提取
    - Database: 数据库查询
    - FINISH: 任务完成

    当前对话: {state["messages"][-1].content}
    """

    supervisor_chain = (
        SystemMessage(content=prompt)
        | llm.with_structured_output(RouteResponse)
    )

    result = await supervisor_chain.ainvoke(state)

    return {
        "next": result.next,
        "reasoning": result.reasoning
    }
```

---

### 3️⃣ 双层内存架构（Mem0 + Qdrant）

**外部项目实现**：

```python
async def ask(self, question: str, user_id: str, chat_id: str, tenant_id: str):
    # 1. Mem0: 检索长期记忆（用户偏好、实体）
    memories = await self.__search_memory(question, user_id=user_id)

    # 2. Qdrant: 检索当前会话历史
    relevant_docs = self.vector_store.get_chat_by_id(
        chat_id=chat_id, user_id=user_id, tenant_id=tenant_id
    )

    # 3. 组装上下文
    context = "Relevant information from previous conversations:\n"
    for memory in memories['results']:
        context += f" - {memory['memory']}\n"

    if relevant_docs:
        context += "\nRelevant chat history:\n"
        for doc in relevant_docs:
            context += f" - User: {doc['user_message']}\n"
            context += f" - Assistant: {doc['assistant_message']}\n"

    # 4. 调用 LangGraph
    messages = [
        SystemMessage(content=f"CONTEXT AWARENESS:\n{context}"),
        HumanMessage(content=question)
    ]
    response = await self.__graph.ainvoke(messages)

    # 5. 存储新记忆
    await self.__add_memory(question, response_content, user_id=user_id)
    self.vector_store.store_conversation(question, response_content, ...)
```

**架构图**：

```
┌─────────────────────────────────────────────────────────┐
│                     Agent Context                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Mem0       │         │   Qdrant     │             │
│  │ (长期记忆)    │         │ (会话历史)    │             │
│  │              │         │              │             │
│  │ - 用户偏好    │         │ - 语义搜索    │             │
│  │ - 实体提取    │         │ - 向量检索    │             │
│  │ - 事实存储    │         │ - 滑动窗口    │             │
│  └──────────────┘         └──────────────┘             │
│         │                         │                     │
│         └───────────┬─────────────┘                     │
│                     ▼                                   │
│            ┌─────────────────┐                          │
│            │ Context Builder │                          │
│            └─────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

**Kiki 当前状态**：
- `MemoryManager` 统一管理短期和长期记忆
- 缺少实体提取和语义检索能力

**Kiki 改进建议**：

```python
# app/agent/memory/entity_extractor.py (新文件)
from typing import List
from pydantic import BaseModel

class Entity(BaseModel):
    """提取的实体"""
    name: str
    type: str  # "person", "location", "organization", etc.
    confidence: float

class EntityExtractor:
    """实体提取器 - 增强长期记忆"""

    async def extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取实体"""
        llm = get_llm_service()
        prompt = f"""
        从以下文本中提取重要实体（人物、地点、组织）：

        文本: {text}

        返回 JSON 格式的实体列表。
        """

        response = await llm.ainvoke(prompt)
        return self._parse_entities(response)
```

---

### 4️⃣ 链式依赖注入（FastAPI 标准）

**外部项目实现**：

```python
# app/api/deps.py
async def get_vector_store() -> MultiTenantVectorStore:
    return MultiTenantVectorStore()

def get_ai_support(
    vector_store: Annotated[MultiTenantVectorStore, Depends(get_vector_store)]
) -> AISupport:
    return AISupport(vector_store)

def get_streaming_service(
    support_agent: Annotated[AISupport, Depends(get_ai_support)]
) -> StreamingService:
    return StreamingService(support_agent=support_agent)

# 使用
@router.post("/completions")
async def chat_completions(
    streaming_service: StreamingService = Depends(get_streaming_service)
):
    return await streaming_service.streaming_chat(request, current_user)
```

**优点**：
- FastAPI 原生依赖注入
- 显式的依赖链（`vector_store → ai_support → streaming_service`）
- 自动处理生命周期

**Kiki 当前状态**：
- 依赖注入分散在各个 API 文件中
- 缺少统一的依赖注入文件

**Kiki 改进建议**：

```python
# app/api/v1/dependencies.py (新文件)
from typing import Annotated, AsyncIterator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory.manager import MemoryManager, create_memory_manager
from app.agent.graph.builder import compile_chat_graph
from app.db.session import get_db
from app.llm import get_llm_service
from app.middleware import get_current_tenant_id

# 类型别名
DbDep = Annotated[AsyncSession, Depends(get_db)]
TenantIdDep = Annotated[int | None, Depends(get_current_tenant_id)]

async def get_memory_manager_dep(
    session_id: str,
    user_id: str | None = None,
    db: DbDep,
) -> AsyncIterator[MemoryManager]:
    """获取 Memory Manager 实例"""
    manager = create_memory_manager(
        session_id=session_id,
        user_id=user_id,
    )
    try:
        yield manager
    finally:
        await manager.cleanup()

async def get_chat_graph_dep(
    system_prompt: str | None = None,
):
    """获取编译后的聊天图"""
    return await compile_chat_graph(system_prompt=system_prompt)
```

---

### 5️⃣ 多租户向量存储（Payload Partitioning）

**外部项目实现**：

```python
# 单 Collection + Payload 过滤
response = client.scroll(
    collection_name="multi_tenant_chat_history",
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="metadata.tenant_id",
                match=MatchValue(value=tenant_id)
            ),
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=str(user_id))
            ),
            FieldCondition(
                key="metadata.chat_id",
                match=MatchValue(value=chat_id)
            )
        ]),
    with_payload=True,
)
```

**优点**：
- 使用单一 collection，通过 payload 过滤实现租户隔离
- 减少集合数量（相比为每个租户创建 collection）
- 灵活的查询条件组合

**Kiki 当前状态**：
- 使用 PostgreSQL 字段过滤实现多租户
- 向量存储部分可以借鉴此模式

---

### 6️⃣ MCP 工具的 AsyncExitStack 管理

**外部项目实现**：

```python
class MCPClientWrapper:
    def __init__(self, server_url: str, name: str):
        self.server_url = server_url
        self.name = name
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self) -> None:
        (read, write) = await self.exit_stack.enter_async_context(
            sse_client(f"{self.server_url}")
        )
        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        self.session = session
        await self.session.initialize()

    async def close(self):
        await self.exit_stack.aclose()
```

**优点**：
- 使用 `AsyncExitStack` 自动管理资源生命周期
- 清晰的连接/加载/关闭流程

**Kiki 参考**：`app/agent/tools/mcp.py` 可以添加类似资源管理

---

### 7️⃣ OpenAI 兼容的流式响应

**外部项目实现**：

```python
async def generate_stream():
    # 首个 chunk
    first_chunk = await create_streaming_openai_chunk(role="assistant")
    yield f"data: {json.dumps(first_chunk)}\n\n"

    # 内容 chunks
    for i in range(0, len(full_content), chunk_size):
        content_chunk = full_content[i:i+chunk_size]
        chunk_data = await create_streaming_openai_chunk(content=content_chunk)
        yield f"data: {json.dumps(chunk_data)}\n\n"

    # 结束标记
    final_chunk = await create_streaming_openai_chunk(finish_reason="stop")
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

return StreamingResponse(
    generate_stream(),
    media_type="text/event-stream",
)
```

**优点**：
- 兼容 OpenAI API 格式（易于客户端集成）
- 固定 chunk 大小确保流畅性
- 标准的 SSE 格式

**Kiki 当前状态**：
- `app/agent/streaming/service.py` 实现了流式响应
- 需要验证是否完全兼容 OpenAI 格式

---

## 📋 改进优先级建议

| 优先级 | 改进项 | 预估工作量 | 收益 | 文件位置 |
|--------|--------|-----------|------|----------|
| 🔴 高 | 创建 `app/api/v1/dependencies.py` | 2h | 提升可维护性 | 新建文件 |
| 🔴 高 | Qdrant/PostgreSQL 连接池单例 | 3h | 降低资源消耗 | `app/db/session.py` |
| 🟡 中 | Supervisor-Agent 模式集成 | 8h | 支持多 Agent 编排 | `app/agent/graph/supervisor.py` |
| 🟡 中 | 双层内存架构（集成实体提取） | 12h | 提升记忆能力 | `app/agent/memory/entity_extractor.py` |
| 🟡 中 | MCP 工具 AsyncExitStack 管理 | 2h | 改进资源清理 | `app/agent/tools/mcp.py` |
| 🟢 低 | 流式响应 OpenAI 格式验证 | 1h | 提升兼容性 | `app/agent/streaming/service.py` |
| 🟢 低 | 多租户向量存储 Payload 过滤 | 4h | 优化向量查询 | `app/agent/memory/store.py` |

---

## ✅ 已完成的改进

### 2026-02-03

#### 1️⃣ 创建统一的依赖注入文件 ✅

**文件**: `app/api/v1/dependencies.py`

**实现内容**：
- 定义类型别名（`DbDep`, `TenantIdDep`, `AgentDep`, `LlmServiceDep` 等）
- 实现链式依赖注入函数（`get_session_service_dep`, `get_agent_with_memory_dep` 等）
- 添加服务类依赖（`get_knowledge_service_dep`, `get_model_service_dep` 等）
- 添加辅助函数（`validate_session_access_dep`, `resolve_effective_user_id_dep`）

**使用示例**：
```python
from app.api.v1.dependencies import DbDep, TenantIdDep, AgentDep

@router.get("/items/{id}")
async def get_item(
    id: str,
    db: DbDep,              # 简洁的类型别名
    tenant_id: TenantIdDep,
):
    # ...
```

#### 2️⃣ 实现连接池单例模式 ✅

**文件**: `app/infra/database.py`

**实现内容**：
- 添加 `DatabaseConnectionPool` 类（线程安全单例）
- 使用双重检查锁定确保线程安全
- 支持懒初始化 + 自动清理
- 兼容现有代码（保留 `get_async_engine()` 等函数）

**设计模式参考**：
- 外部项目的 `MultiTenantVectorStore` 单例模式
- GoF 单例模式 + Python 线程安全

**代码示例**：
```python
pool = DatabaseConnectionPool()
engine = pool.get_async_engine()  # 全局唯一实例

# 应用关闭时
await pool.close()  # 释放所有连接
```

#### 3️⃣ 实现 Qdrant 客户端单例 ✅

**文件**: `app/vector_stores/qdrant.py`

**实现内容**：
- 添加 `QdrantClientSingleton` 类（线程安全单例）
- 支持多个配置的客户端（通过配置键区分）
- 修改 `QdrantVectorStore.initialize()` 使用单例客户端
- 添加客户端关闭管理

**设计模式参考**：
- 外部项目的 `MultiTenantVectorStore` 单例模式

**代码示例**：
```python
client = QdrantClientSingleton()
qdrant_client = await client.get_client(config)

# 应用关闭时
await client.close_all()
```

---

### 4️⃣ MCP 工具 AsyncExitStack 管理 ✅

**文件**: `app/agent/tools/mcp.py`

**实现内容**：
- 使用 `AsyncExitStack` 管理 MCP 会话生命周期
- 自动清理 stdio/http/sse 连接资源
- 改进错误处理和资源释放

**设计模式参考**：
- 外部项目的 `MCPClientWrapper` 实现

**代码示例**：
```python
class MCPClient:
    def __init__(self):
        self._exit_stack: AsyncExitStack | None = None

    async def initialize(self):
        self._exit_stack = AsyncExitStack()
        # 使用 exit_stack 管理会话
        session = await self._exit_stack.enter_async_context(stdio_client_ctx)

    async def close(self):
        await self._exit_stack.aclose()  # 自动清理所有资源
```

---

### 5️⃣ Supervisor-Agent 多 Agent 编排 ✅

**文件**: `app/agent/graph/supervisor.py`

**实现内容**：
- 创建 `SupervisorState` 状态类型
- 实现 `supervisor_node` 路由决策节点
- 添加专门 Agent（Researcher、Scrapper、Database）
- 实现 `build_supervisor_graph` 图构建函数

**设计模式参考**：
- 外部项目的 `supervisor_agent` 实现

**代码示例**：
```python
from app.agent.graph.supervisor import invoke_supervisor

result = await invoke_supervisor(
    message="帮我搜索最新的 AI 技术趋势",
    session_id="session-123"
)

# Supervisor 自动路由到 Researcher Agent
# 结果包含 agent_results、agent_history 等
```

---

### 6️⃣ 实体提取增强长期记忆 ✅

**文件**: `app/agent/memory/entity_extractor.py`

**实现内容**：
- 创建 `EntityExtractor` 实体提取器
- 定义 `EntityType` 枚举（人物、组织、地点等）
- 实现 `EntityStore` 实体存储管理
- 支持从消息列表提取实体

**设计模式参考**：
- 外部项目的 Mem0 实体提取

**代码示例**：
```python
from app.agent.memory.entity_extractor import get_entity_extractor

extractor = get_entity_extractor()
response = await extractor.extract(
    text="我喜欢用 Python 和 FastAPI 开发 Web 应用",
    user_id="user-123",
)

# 返回实体：Python (skill), FastAPI (product), Web (concept)
```

---

## 🎓 总结

### 外部项目的优势

- ✅ **简洁的全局单例模式** - 减少资源开销
- ✅ **Supervisor-Agent 多 Agent 编排** - 支持复杂任务分解
- ✅ **Mem0 + Qdrant 双层内存** - 分离长期/短期记忆
- ✅ **清晰的依赖注入链** - FastAPI 标准模式
- ✅ **OpenAI 兼容的流式响应** - 易于客户端集成

### Kiki 项目的优势

- ✅ **更模块化的目录结构** - 高内聚低耦合
- ✅ **完善的可观测性** - Langfuse + Prometheus + structlog
- ✅ **MCP 注册表更完善** - 支持多种传输方式
- ✅ **类型注解更完整** - mypy 严格模式
- ✅ **企业级特性** - 多租户、审计日志、工具拦截

### 建议采纳策略

**保持 Kiki 的优势**：
- 继续使用模块化架构
- 保持完善的可观测性
- 维持严格类型检查

**选择性借鉴外部项目**：
- 资源管理使用单例模式（Qdrant、PostgreSQL）
- 集成 Supervisor-Agent 支持多 Agent 编排
- 添加实体提取增强长期记忆
- 创建统一的依赖注入文件

---

## 📚 参考资料

- **外部项目**: `../a/fastapi-langgraph-chatbot-with-vector-store-memory-mcp-tools-and-voice-mode/`
- **Kiki 架构**: `ARCHITECTURE.md`
- **Kiki Agent 模块**: `AGENT.md`
- **LangGraph 官方文档**: https://langchain-ai.github.io/langgraph/
