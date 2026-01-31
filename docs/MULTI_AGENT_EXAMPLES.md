# Kiki 多 Agent 系统示例

> 展示如何使用 Kiki 的多 Agent 协作模式构建复杂应用

## 目录

- [概述](#概述)
- [Router Agent 示例](#router-agent-示例)
- [Supervisor Agent 示例](#supervisor-agent-示例)
- [Handoff/Swarm Agent 示例](#handoffswarm-agent-示例)
- [完整 FastAPI 集成示例](#完整-fastapi-集成示例)

---

## 概述

Kiki 提供三种多 Agent 协作模式：

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| **Router** | 意图识别、任务分发 | 根据用户意图路由到不同的专业 Agent |
| **Supervisor** | 复杂任务分解 | 由监督者协调多个 Worker 协作完成 |
| **Handoff/Swarm** | 动态协作 | Agent 可以主动切换控制权 |

---

## Router Agent 示例

### 场景：客服系统

根据用户咨询类型（销售、售后、技术）路由到不同的专业 Agent。

```python
from app.agent.graph import compile_chat_graph
from app.agent.multi_agent import RouterAgent
from app.llm import LLMService

llm_service = LLMService()

# 创建专业 Agent
sales_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="""你是销售专家，负责：
- 产品推荐
- 价格咨询
- 促销活动介绍

沟通风格：热情、专业"""
)

support_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="""你是客服专家，负责：
- 订单查询
- 退换货处理
- 物流咨询

沟通风格：耐心、细致"""
)

tech_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="""你是技术专家，负责：
- 产品使用指导
- 故障排查
- 技术问题解答

沟通风格：专业、清晰"""
)

# 创建路由 Agent
router = RouterAgent(
    llm_service=llm_service,
    agents={
        "Sales": sales_agent,
        "Support": support_agent,
        "Tech": tech_agent,
    },
    router_prompt="""根据用户咨询内容，选择最合适的专家：
- Sales: 购买咨询、价格、促销
- Support: 订单、物流、退换货
- Tech: 使用指导、故障排查"""
)

graph = router.compile()

# 测试
async def test_router():
    # 销售咨询
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "这款手机多少钱？"}]},
        config={"configurable": {"thread_id": "test-1"}},
    )
    print("销售:", response["messages"][-1].content)

    # 售后咨询
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "我的订单什么时候到？"}]},
        config={"configurable": {"thread_id": "test-2"}},
    )
    print("售后:", response["messages"][-1].content)

    # 技术咨询
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "手机怎么连接蓝牙？"}]},
        config={"configurable": {"thread_id": "test-3"}},
    )
    print("技术:", response["messages"][-1].content)
```

---

## Supervisor Agent 示例

### 场景：内容生产系统

由监督者协调研究员、写手、审核员完成报告撰写。

```python
from app.agent.graph import compile_chat_graph
from app.agent.multi_agent import SupervisorAgent
from app.llm import LLMService

llm_service = LLMService()

# 创建 Worker Agent
researcher = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="""你是研究员，负责：
1. 收集主题相关的最新信息
2. 分析和整理关键数据
3. 为写手提供素材

输出格式：结构化的研究报告"""
)

writer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="""你是写手，负责：
1. 根据研究员的素材撰写报告
2. 确保内容准确、逻辑清晰
3. 使用简洁易懂的语言

输出格式：完整的报告文档"""
)

reviewer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="""你是审核员，负责：
1. 检查报告的准确性和完整性
2. 评估内容的可读性
3. 提出修改建议

输出格式：审核意见和最终结论"""
)

# 创建监督 Agent
supervisor = SupervisorAgent(
    llm_service=llm_service,
    workers={
        "Researcher": researcher,
        "Writer": writer,
        "Reviewer": reviewer,
    },
    supervisor_prompt="""你是内容生产项目经理，负责：
1. 接收用户的报告需求
2. 协调研究员收集素材
3. 指派写手撰写初稿
4. 要求审核员检查质量
5. 决定是否需要修改或可以交付

工作流程：Researcher → Writer → Reviewer → 完成"""
)

graph = supervisor.compile()

# 测试
async def test_supervisor():
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "写一份关于 2024 年 AI 发展趋势的报告"}]},
        config={"configurable": {"thread_id": "report-1"}},
    )

    # 查看协作过程
    for msg in response["messages"]:
        if hasattr(msg, 'name') and msg.name:
            print(f"[{msg.name}]: {msg.content[:100]}...")
```

### 场景：软件开发团队

模拟真实的软件开发流程。

```python
from app.agent.graph import compile_chat_graph
from app.agent.multi_agent import SupervisorAgent

# 创建开发团队成员
product_manager = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是产品经理，负责需求分析和验收标准"
)

designer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是 UI/UX 设计师，负责界面设计和交互流程"
)

developer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是开发工程师，负责代码实现"
)

qa_engineer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是 QA 工程师，负责测试和质量保证"
)

supervisor = SupervisorAgent(
    llm_service=llm_service,
    workers={
        "PM": product_manager,
        "Designer": designer,
        "Dev": developer,
        "QA": qa_engineer,
    },
    supervisor_prompt="协调软件开发流程：PM → Designer → Dev → QA → 交付"
)

graph = supervisor.compile()

# 测试
async def test_development():
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "开发一个用户登录功能"}]},
        config={"configurable": {"thread_id": "dev-1"}},
    )
    print(response["messages"][-1].content)
```

---

## Handoff/Swarm Agent 示例

### 场景：客服协作系统

销售和客服 Agent 可以互相切换，处理复杂咨询。

```python
from app.agent.multi_agent import HandoffAgent, create_swarm
from app.llm import LLMService
from langchain_core.tools import tool

llm_service = LLMService()

# 定义工具
@tool
async def search_products(category: str) -> str:
    """搜索产品信息"""
    return f"在 {category} 类别下找到 5 个产品"

@tool
async def check_inventory(product_id: str) -> str:
    """检查库存"""
    return f"产品 {product_id} 库存: 100 件"

@tool
async def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    return f"订单 {order_id} 状态: 已发货"

@tool
async def process_refund(order_id: str, reason: str) -> str:
    """处理退款"""
    return f"订单 {order_id} 退款已处理"

# 创建可切换的 Agent
alice = HandoffAgent(
    name="Alice",
    llm_service=llm_service,
    tools=[search_products, check_inventory],
    handoff_targets=["Bob"],
    system_prompt="""你是 Alice，销售专家。

职责：
- 产品推荐
- 价格咨询
- 库存查询

当遇到以下情况时转给 Bob：
- 订单查询
- 退换货请求
- 物流问题"""
)

bob = HandoffAgent(
    name="Bob",
    llm_service=llm_service,
    tools=[check_order_status, process_refund],
    handoff_targets=["Alice"],
    system_prompt="""你是 Bob，客服专家。

职责：
- 订单查询
- 退换货处理
- 售后服务

当遇到以下情况时转给 Alice：
- 产品咨询
- 价格询问
- 新购买需求"""
)

# 创建 Swarm
graph = create_swarm(
    agents=[alice, bob],
    default_agent="Alice",
)

# 测试
async def test_swarm():
    # 场景 1：先销售后转售后
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "我要退款，同时再买一个新的"}]},
        config={"configurable": {"thread_id": "swarm-1"}},
    )

    for msg in response["messages"]:
        if hasattr(msg, 'name') and msg.name:
            print(f"[{msg.name}]: {msg.content[:80]}...")
```

---

## 完整 FastAPI 集成示例

### 项目结构

```
app/
├── api/
│   └── v1/
│       └── multi_chat.py      # 多 Agent 聊天 API
├── agent/
│   └── examples.py            # Agent 定义
└── main.py                    # 应用入口
```

### API 实现

```python
# app/api/v1/multi_chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import compile_chat_graph
from app.agent.multi_agent import RouterAgent, SupervisorAgent, HandoffAgent, create_swarm
from app.llm import get_llm_service
from langchain_core.tools import tool

router = APIRouter(prefix="/multi-chat", tags=["multi-agent"])

llm_service = get_llm_service()

# ============== 定义工具 ==============

@tool
async def search_products(query: str) -> str:
    """搜索产品"""
    return f"找到产品: {query}"

@tool
async def check_order(order_id: str) -> str:
    """查询订单"""
    return f"订单 {order_id} 状态: 已发货"

# ============== 定义 Agent ==============

sales_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是销售专家",
)

support_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是客服专家",
)

# ============== 创建多 Agent 系统 ==============

router_system = RouterAgent(
    llm_service=llm_service,
    agents={"Sales": sales_agent, "Support": support_agent},
).compile()

supervisor_system = SupervisorAgent(
    llm_service=llm_service,
    workers={"Worker1": sales_agent, "Worker2": support_agent},
).compile()

alice = HandoffAgent(name="Alice", llm_service=llm_service, tools=[search_products])
bob = HandoffAgent(name="Bob", llm_service=llm_service, tools=[check_order])
swarm_system = create_swarm(agents=[alice, bob], default_agent="Alice")

# ============== 系统存储 ==============

AGENT_SYSTEMS = {
    "router": router_system,
    "supervisor": supervisor_system,
    "swarm": swarm_system,
}

# ============== API 端点 ==============

class ChatRequest(BaseModel):
    system_type: str = Field(..., description="系统类型: router/supervisor/swarm")
    message: str = Field(..., description="用户消息")
    session_id: str = Field(..., description="会话 ID")

@router.post("/chat")
async def multi_chat(request: ChatRequest):
    """多 Agent 聊天接口"""
    system = AGENT_SYSTEMS.get(request.system_type)
    if not system:
        raise HTTPException(status_code=400, detail=f"Unknown system type: {request.system_type}")

    try:
        response = await system.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config={"configurable": {"thread_id": request.session_id}},
        )
        return {
            "content": response["messages"][-1].content,
            "session_id": request.session_id,
            "system_type": request.system_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 主应用注册

```python
# app/main.py

from fastapi import FastAPI
from app.api.v1 import multi_chat

app = FastAPI()

# 注册路由
app.include_router(multi_chat.router, prefix="/api/v1")
```

### 测试 API

```bash
# 测试路由系统
curl -X POST "http://localhost:8000/api/v1/multi-chat/chat?system_type=router&message=我想买手机&session_id=test-123"

# 测试监督系统
curl -X POST "http://localhost:8000/api/v1/multi-chat/chat?system_type=supervisor&message=写一份报告&session_id=test-456"

# 测试 Swarm 系统
curl -X POST "http://localhost:8000/api/v1/multi-chat/chat?system_type=swarm&message=我要退款&session_id=test-789"
```

---

## 高级用法

### 动态工具加载

根据 Agent 角色动态加载工具：

```python
from app.agent.graph import create_react_agent
from app.agent.tools import alist_tools

async def get_tools_for_role(role: str):
    """根据角色获取工具"""
    all_tools = await alist_tools()

    role_tools = {
        "sales": ["search_products", "calculate_discount"],
        "support": ["check_order", "process_refund"],
        "tech": ["search_knowledge", "troubleshoot"],
    }

    tool_names = role_tools.get(role, [])
    return [t for t in all_tools if t.name in tool_names]

# 创建带动态工具的 Agent
async def create_dynamic_agent(role: str):
    tools = await get_tools_for_role(role)
    return create_react_agent(
        llm_service=llm_service,
        system_prompt=f"你是 {role} 专家",
        tools=tools,
    )
```

### 状态持久化

使用 PostgreSQL 持久化多 Agent 对话状态：

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# 创建检查点保存器
pool = AsyncConnectionPool(conninfo="postgresql://localhost:5432/kiki")
checkpointer = AsyncPostgresSaver(pool)
await checkpointer.setup()

# 编译图时使用检查点
graph = router.compile(checkpointer=checkpointer)

# 对话状态自动持久化
response = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config={"configurable": {"thread_id": "user-123"}},
)
```

---

## 参考资源

- [LangGraph 多 Agent 文档](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Agent 系统文档](AGENT.md)
- [API 文档](API.md)
- [源码示例](../app/agent/examples.py)
