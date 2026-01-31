"""Human-in-the-Loop 图工作流

支持人工审核、干预和继续执行的 Agent 工作流。

使用场景：
- 需要人工审核敏感操作
- 需要人工确认关键决策
- 需要人工干预纠正错误
- 需要分阶段执行复杂任务

示例：
    ```python
    from app.agent.graph.interrupt import InterruptGraph

    graph = InterruptGraph()
    graph.compile()

    # 第一次调用 - 会在需要审核时中断
    config = {"configurable": {"thread_id": "session-123"}}
    result = await graph.ainvoke(input_data, config)
    # 如果触发中断，result["status"] 会是 "interrupted"

    # 查询当前状态
    state = await graph.aget_state(config)
    next_value = state.next  # 下一步要执行的节点

    # 人工审核后继续执行
    approval = {"approved": True, "feedback": "同意执行"}
    result = await graph.aresume(approval, config)
    ```
"""

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RunnableConfig, interrupt
from pydantic import BaseModel, Field

from app.agent.graph.types import AgentState
from app.agent.tools import get_tool_node, list_tools
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 审核数据模型 ==============


class HumanApproval(BaseModel):
    """人工审核结果

    用于恢复被中断的执行流程。
    """

    approved: bool = Field(description="是否批准继续执行")
    feedback: str = Field(default="", description="审核意见或修改建议")
    action_override: str | None = Field(default=None, description="覆盖的操作（可选）")


class InterruptRequest(BaseModel):
    """中断请求

    Agent 请求人工干预的数据结构。
    """

    reason: str = Field(description="请求中断的原因")
    context: dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    proposed_action: str = Field(description="建议的操作")
    requires_approval: bool = Field(default=True, description="是否需要批准")


# ============== 节点函数 ==============


async def interrupt_chat_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["check_interrupt", "__end__"]]:
    """聊天节点 - 生成 LLM 响应

    Args:
        state: 当前状态
        config: 运行配置

    Returns:
        Command 指令
    """
    logger.debug("interrupt_chat_node_entered")

    # 从 config 中获取 llm_service 和 system_prompt
    metadata = config.get("metadata", {})
    llm_service = metadata.get("llm_service")
    system_prompt = metadata.get("system_prompt")

    if not llm_service:
        raise RuntimeError("LLM service not found in config metadata")

    # 获取带工具绑定的 LLM
    llm = llm_service.get_llm_with_tools([])
    if not llm:
        raise RuntimeError("Failed to get LLM from LLMService")

    # 构建提示词模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 构建 LCEL 链
    chain = prompt_template | llm

    try:
        response = await chain.ainvoke(
            {
                "system_prompt": system_prompt,
                "messages": state["messages"],
            },
            config,
        )

        logger.info(
            "llm_response_generated",
            has_tool_calls=bool(hasattr(response, "tool_calls") and response.tool_calls),
        )

        # 检查是否有之前的拒绝反馈
        updates: dict[str, Any] = {"messages": [response]}

        if "_approval" in state and not state["_approval"].get("approved", True):
            feedback = state["_approval"].get("feedback", "")
            if feedback:
                updates["messages"].append(
                    SystemMessage(content=f"审核反馈：{feedback}（请根据反馈调整方案）")
                )

        return Command(update=updates, goto="check_interrupt")

    except Exception as e:
        logger.exception("llm_call_failed", error=str(e))
        error_message = AIMessage(content=f"处理请求时出错：{str(e)}")
        return Command(
            update={"messages": [error_message]},
            goto="__end__",
        )


async def check_interrupt_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["execute", "__end__"]]:
    """检查是否需要人工审核节点

    使用 LangGraph 的 interrupt 机制实现人工干预。

    Args:
        state: 当前状态
        config: 运行配置

    Returns:
        Command 路由指令
    """
    logger.debug("check_interrupt_node_entered")

    # 获取最后一条消息
    last_message = state["messages"][-1]

    # 检查是否有工具调用
    needs_review = False
    review_reason = ""
    proposed_action = ""

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 检查是否有需要审核的工具
        sensitive_tools = {"delete", "send", "email", "message", "modify", "update"}
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "").lower()
            if any(keyword in tool_name for keyword in sensitive_tools):
                needs_review = True
                review_reason = f"检测到敏感操作：{tool_call.get('name')}"
                proposed_action = (
                    f"调用工具：{tool_call.get('name')} 参数：{tool_call.get('args')}"
                )
                break

    # 检查是否有特殊标记（显式触发中断）
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        if "<REVIEW>" in last_message.content or "<审核>" in last_message.content:
            needs_review = True
            review_reason = "Agent 请求人工审核"
            proposed_action = last_message.content

    if not needs_review:
        # 不需要审核，直接执行
        return Command(goto="execute")

    # 使用 interrupt 机制等待人工输入
    logger.info(
        "interrupt_triggered",
        reason=review_reason,
        proposed_action=proposed_action[:200] if proposed_action else "",
    )

    # 创建中断请求
    interrupt_request = InterruptRequest(
        reason=review_reason,
        proposed_action=proposed_action,
        requires_approval=True,
    )

    # 触发中断 - 这里会暂停执行，等待 aresume
    approval = interrupt(
        {
            "type": "human_review",
            "request": interrupt_request.model_dump(),
            "message": f"需要人工审核：{review_reason}",
        }
    )

    # 人工恢复后，approval 会包含审核结果
    logger.info(
        "interrupt_resumed",
        approved=approval.get("approved", False),
        feedback=approval.get("feedback", ""),
    )

    # 将审核结果存入状态，供后续节点使用
    if approval.get("approved", False):
        return Command(
            update={"_approval": approval},
            goto="execute",
        )
    else:
        # 被拒绝，返回聊天节点让 Agent 重新思考
        return Command(
            update={"_approval": approval},
            goto="chat",
        )


async def execute_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["chat", "__end__"]]:
    """执行节点 - 执行工具调用

    Args:
        state: 当前状态
        config: 运行配置

    Returns:
        Command 指令
    """
    logger.debug("execute_node_entered")

    # 检查是否有审核拒绝
    if "_approval" in state and not state["_approval"].get("approved", True):
        feedback = state["_approval"].get("feedback", "未提供原因")
        return Command(
            update={
                "messages": [AIMessage(content=f"操作未获批准：{feedback}")],
                "_approval": None,  # 清除审核状态
            },
            goto="__end__",
        )

    # 获取最后一条消息
    last_message = state["messages"][-1]

    # 如果有工具调用，执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_node = get_tool_node()
        result = await tool_node.ainvoke(state, config)
        return Command(update=result, goto="chat")

    return Command(goto="__end__")


# ============== 图构建函数 ==============


def build_interrupt_graph() -> StateGraph:
    """构建 Interrupt Graph（未编译）

    Returns:
        StateGraph 实例
    """
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("chat", interrupt_chat_node)
    builder.add_node("check_interrupt", check_interrupt_node)
    builder.add_node("execute", execute_node)

    # 设置入口点
    builder.add_edge(START, "chat")

    # chat -> check_interrupt (每次生成响应后检查是否需要审核)
    builder.add_edge("chat", "check_interrupt")

    # check_interrupt 会动态路由到 execute 或 __end__
    builder.add_conditional_edges(
        "check_interrupt",
        lambda state: "execute" if state.get("_approval", {}).get("approved", False) else "__end__",
        {
            "execute": "execute",
            "__end__": END,
        },
    )

    # execute -> chat (如果还有工具调用)
    builder.add_conditional_edges(
        "execute",
        lambda state: "chat" if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls else "__end__",
        {
            "chat": "chat",
            "__end__": END,
        },
    )

    logger.debug("interrupt_graph_structure_built")
    return builder


def compile_interrupt_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """编译 Interrupt Graph

    Args:
        checkpointer: 检查点保存器

    Returns:
        编译后的 CompiledStateGraph
    """
    builder = build_interrupt_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.debug("using_memory_checkpointer")

    graph = builder.compile(checkpointer=checkpointer)

    logger.info("interrupt_graph_compiled")

    return graph


# ============== InterruptGraph 类（向后兼容）=============


class InterruptGraph:
    """Human-in-the-Loop 图（向后兼容类）

    支持人工审核和干预的 Agent 工作流。

    流程：
        1. chat 节点生成响应和决策
        2. 如果需要人工审核，路由到 human_review 节点
        3. human_review 节点中断执行，等待人工输入
        4. 人工审核后，继续执行或重新思考
        5. execute 节点执行最终操作

    中断类型：
        - 操作前审核：敏感操作需要人工批准
        - 决策确认：关键决策需要人工确认
        - 错误纠正：执行失败需要人工介入
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """初始化 Human-in-the-Loop 图

        Args:
            llm_service: LLM 服务实例
            system_prompt: 系统提示词
            checkpointer: 检查点保存器
        """
        self._llm_service = llm_service or get_llm_service()
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._graph: CompiledStateGraph | None = None
        self._checkpointer = checkpointer

        logger.info(
            "interrupt_graph_initialized",
            model=self._llm_service.current_model,
        )

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个谨慎的 AI 助手，在执行敏感操作前会请求人工审核。

当遇到以下情况时，你应该请求人工审核：
1. 删除、修改重要数据的操作
2. 发送邮件、消息等通信操作
3. 涉及敏感信息的查询
4. 任何可能产生重大影响的操作
5. 你不确定是否应该执行的操作

请始终以友好、专业的方式回应用户，并在需要时明确说明为什么需要人工审核。"""

    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        """编译图"""
        if self._graph is None:
            self._checkpointer = checkpointer or self._checkpointer
            self._graph = compile_interrupt_graph(self._checkpointer)
        return self._graph

    async def ainvoke(
        self,
        input_data: dict,
        config: RunnableConfig,
    ) -> AgentState:
        """异步调用图"""
        if self._graph is None:
            self.compile()

        # 确保 metadata 中有 llm_service 和 system_prompt
        if isinstance(config, dict):
            metadata = config.get("metadata", {})
            if "llm_service" not in metadata:
                metadata["llm_service"] = self._llm_service
            if "system_prompt" not in metadata:
                metadata["system_prompt"] = self._system_prompt
            config["metadata"] = metadata

        return await self._graph.ainvoke(input_data, config)

    async def aresume(
        self,
        approval: dict[str, Any] | HumanApproval,
        config: RunnableConfig,
    ) -> AgentState:
        """恢复被中断的执行

        Args:
            approval: 审核结果（可以是 dict 或 HumanApproval）
            config: 运行配置

        Returns:
            最终状态
        """
        if self._graph is None:
            self.compile()

        # 转换为 dict
        if isinstance(approval, BaseModel):
            approval_data = approval.model_dump()
        else:
            approval_data = approval

        logger.info(
            "resuming_interrupted_execution",
            approved=approval_data.get("approved"),
        )

        # 审核结果通过 update_state 传入
        await self._graph.update_state(
            config,
            {"_approval": approval_data},
        )

        # 继续执行
        return await self._graph.ainvoke(None, config)

    async def get_pending_review(
        self,
        config: RunnableConfig,
    ) -> InterruptRequest | None:
        """获取待审核的请求

        Args:
            config: 运行配置

        Returns:
            待审核的请求，如果没有则返回 None
        """
        if self._graph is None:
            self.compile()

        state = await self._graph.aget_state(config)

        # 检查是否有待处理的中断
        if state and state.next:
            current_state = state.values if hasattr(state, "values") else {}
            interrupt_request = current_state.get("_interrupt_request")
            if interrupt_request:
                return InterruptRequest(**interrupt_request)

        return None


# ============== 便捷函数 ==============


def create_interrupt_graph(
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> InterruptGraph:
    """创建 Human-in-the-Loop 图实例

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器

    Returns:
        已编译的 InterruptGraph 实例

    Examples:
        ```python
        from app.agent.graph import create_interrupt_graph

        graph = create_interrupt_graph(
            system_prompt="你是一个需要人工审核的 AI 助手",
        )

        # 第一次调用
        config = {"configurable": {"thread_id": "session-123"}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="删除所有用户数据")]},
            config,
        )

        # 如果触发中断，进行审核后继续
        if result.get("_interrupted"):
            approval = {"approved": False, "feedback": "不能删除所有数据"}
            result = await graph.aresume(approval, config)
        ```
    """
    graph = InterruptGraph(llm_service, system_prompt, checkpointer)
    graph.compile()
    return graph


__all__ = [
    # 节点函数
    "interrupt_chat_node",
    "check_interrupt_node",
    "execute_node",
    # 构建函数
    "build_interrupt_graph",
    "compile_interrupt_graph",
    # 类
    "InterruptGraph",
    # 便捷函数
    "create_interrupt_graph",
    # 数据模型
    "HumanApproval",
    "InterruptRequest",
]
