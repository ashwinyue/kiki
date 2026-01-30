"""多 Agent 使用示例

演示如何使用不同的多 Agent 模式。
"""

from langchain_core.tools import tool

from app.core.agent.graph import AgentGraph
from app.core.agent.multi_agent import (
    HandoffAgent,
    RouterAgent,
    SupervisorAgent,
    create_swarm,
)
from app.core.llm import LLMService
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============== 定义工具 ==============

@tool
async def search_products(query: str) -> str:
    """搜索产品信息"""
    logger.info("search_products", query=query)
    return f"找到 3 个产品: {query}"


@tool
async def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    logger.info("check_order_status", order_id=order_id)
    return f"订单 {order_id} 状态: 已发货"


@tool
async def process_refund(order_id: str, reason: str) -> str:
    """处理退款"""
    logger.info("process_refund", order_id=order_id, reason=reason)
    return f"订单 {order_id} 退款已处理"


@tool
async def calculate_discount(amount: float, discount_percent: float) -> str:
    """计算折扣后价格"""
    discounted = amount * (1 - discount_percent / 100)
    return f"原价: ¥{amount}, 折扣: {discount_percent}%, 最终价格: ¥{discounted:.2f}"


# ============== 示例 1: Router Agent ==============

async def example_router_agent():
    """Router Agent 示例

    根据用户意图路由到不同的专业 Agent。
    """
    # 初始化 LLM 服务
    llm_service = LLMService(default_model="gpt-4o-mini")

    # 创建专业 Agent
    sales_agent = AgentGraph(
        llm_service=llm_service,
        system_prompt="你是销售专家，负责产品推荐和价格咨询。",
    )

    support_agent = AgentGraph(
        llm_service=llm_service,
        system_prompt="你是客服专家，负责订单查询和售后处理。",
    )

    # 创建路由 Agent
    router = RouterAgent(
        llm_service=llm_service,
        agents={
            "Sales": sales_agent,
            "Support": support_agent,
        },
        router_prompt="根据用户意图选择 Sales(销售) 或 Support(客服)。",
    )

    graph = router.compile()

    # 测试
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "我想买一个手机"}]},
        config={"configurable": {"thread_id": "test-1"}},
    )
    print("Router 示例 - 销售:", response["messages"][-1].content)

    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "我的订单什么时候到？"}]},
        config={"configurable": {"thread_id": "test-2"}},
    )
    print("Router 示例 - 客服:", response["messages"][-1].content)


# ============== 示例 2: Supervisor Agent ==============

async def example_supervisor_agent():
    """Supervisor Agent 示例

    监督 Agent 协调多个 Worker 完成复杂任务。
    """
    llm_service = LLMService(default_model="gpt-4o-mini")

    # 创建 Worker Agent
    researcher = AgentGraph(
        llm_service=llm_service,
        system_prompt="你是研究员，负责收集和分析信息。",
    )

    writer = AgentGraph(
        llm_service=llm_service,
        system_prompt="你是写手，负责整理和撰写报告。",
    )

    reviewer = AgentGraph(
        llm_service=llm_service,
        system_prompt="你是审核员，负责审核报告质量。",
    )

    # 创建监督 Agent
    supervisor = SupervisorAgent(
        llm_service=llm_service,
        workers={
            "Researcher": researcher,
            "Writer": writer,
            "Reviewer": reviewer,
        },
        supervisor_prompt="协调 Researcher、Writer、Reviewer 完成报告撰写任务。",
    )

    graph = supervisor.compile()

    # 测试
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "写一份关于 AI 的报告"}]},
        config={"configurable": {"thread_id": "test-3"}},
    )
    print("Supervisor 示例:", response["messages"][-1].content)


# ============== 示例 3: Handoff Agent (Swarm) ==============

async def example_handoff_agent():
    """Handoff Agent 示例

    Agent 可以主动切换到其他 Agent。
    """
    llm_service = LLMService(default_model="gpt-4o-mini")

    # 创建可切换的 Agent
    alice = HandoffAgent(
        name="Alice",
        llm_service=llm_service,
        tools=[search_products, calculate_discount],
        handoff_targets=["Bob"],
        system_prompt="你是 Alice，销售专家。遇到技术问题时转给 Bob。",
    )

    bob = HandoffAgent(
        name="Bob",
        llm_service=llm_service,
        tools=[check_order_status, process_refund],
        handoff_targets=["Alice"],
        system_prompt="你是 Bob，客服专家。遇到销售咨询时转给 Alice。",
    )

    # 创建 Swarm
    graph = create_swarm(
        agents=[alice, bob],
        default_agent="Alice",
    )

    # 测试
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "我要退款"}]},
        config={"configurable": {"thread_id": "test-4"}},
    )
    print("Handoff 示例:", response["messages"][-1].content)


# ============== 示例 4: 使用便捷函数 ==============

async def example_convenience_function():
    """使用便捷函数创建多 Agent 系统"""
    from app.core.agent.multi_agent import create_multi_agent_system

    llm_service = LLMService(default_model="gpt-4o-mini")

    # Router 模式
    router_graph = create_multi_agent_system(
        mode="router",
        llm_service=llm_service,
        agents={
            "Agent1": AgentGraph(llm_service=llm_service),
            "Agent2": AgentGraph(llm_service=llm_service),
        },
    )

    # Supervisor 模式
    supervisor_graph = create_multi_agent_system(
        mode="supervisor",
        llm_service=llm_service,
        workers={
            "Worker1": AgentGraph(llm_service=llm_service),
            "Worker2": AgentGraph(llm_service=llm_service),
        },
    )

    print("多 Agent 系统创建成功")


# ============== 示例 5: 集成到 FastAPI ==============

"""
# app/api/v1/multi_chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.agent.multi_agent import create_multi_agent_system
from app.core.llm import get_llm_service

router = APIRouter(prefix="/multi-chat", tags=["multi-agent"])

llm_service = get_llm_service()

# 创建多 Agent 系统
sales_agent = AgentGraph(
    llm_service=llm_service,
    system_prompt="你是销售专家...",
)

support_agent = AgentGraph(
    llm_service=llm_service,
    system_prompt="你是客服专家...",
)

multi_graph = create_multi_agent_system(
    mode="router",
    llm_service=llm_service,
    agents={"Sales": sales_agent, "Support": support_agent},
)


class MultiChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: str = Field(..., description="会话 ID")


@router.post("/chat")
async def multi_chat(request: MultiChatRequest):
    response = await multi_graph.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config={"configurable": {"thread_id": request.session_id}},
    )
    return {"response": response["messages"][-1].content}
"""


if __name__ == "__main__":
    pass

    # 需要设置 OPENAI_API_KEY
    # asyncio.run(example_router_agent())
    # asyncio.run(example_supervisor_agent())
    # asyncio.run(example_handoff_agent())
    # asyncio.run(example_convenience_function())
