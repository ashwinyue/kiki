"""基础对话图

实现单 Agent + 工具调用的基础对话流程。
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from app.agent.graphs.base import BaseGraph
from app.agent.state import AgentState
from app.agent.tools import list_tools
from app.core.config import get_settings
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


def route_by_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """根据是否有工具调用决定路由

    首先检查迭代次数，如果超过最大迭代次数则强制结束。

    Args:
        state: 当前状态

    Returns:
        下一个节点名称
    """
    # 检查迭代次数，防止无限循环
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    if iteration_count >= max_iterations:
        logger.warning(
            "max_iterations_reached",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        return "__end__"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


class ChatGraph(BaseGraph):
    """基础对话图

    实现单 Agent + 工具调用的基础对话流程：
    1. chat 节点生成 LLM 响应
    2. 如果有工具调用，路由到 tools 节点
    3. tools 节点执行工具后返回 chat 节点
    4. 重复直到没有工具调用，结束

    使用示例:
        ```python
        from app.agent.graphs import ChatGraph

        graph = ChatGraph(system_prompt="你是一个助手")
        graph.compile()

        response = await graph.ainvoke(
            {"messages": [HumanMessage(content="你好")]},
            {"configurable": {"thread_id": "session-123"}},
        )
        ```
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """初始化对话图

        Args:
            llm_service: LLM 服务实例
            system_prompt: 系统提示词
        """
        super().__init__(llm_service or get_llm_service(), system_prompt)

        # 初始化提示词模板
        self._prompt_template = self._build_prompt_template()

        # 缓存绑定工具后的 LLM
        self._llm_with_tools: BaseChatModel | None = None

        # 缓存 ToolNode（避免重复创建）
        self._tool_node_sync: ToolNode | None = None
        self._tool_node_by_tenant: dict[int | None, ToolNode] = {}

        logger.info(
            "chat_graph_initialized",
            model=self._llm_service.current_model,
        )

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """构建提示词模板

        Returns:
            ChatPromptTemplate 实例
        """
        messages = [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
        return ChatPromptTemplate.from_messages(messages)

    def _get_llm_with_tools(self) -> BaseChatModel | None:
        """获取绑定工具后的 LLM

        使用 LLMService.get_llm_with_tools() 方法，
        该方法确保在正确的顺序下绑定工具并应用重试。

        Returns:
            绑定工具的 LLM 实例
        """
        if self._llm_with_tools is None:
            tools = list_tools()
            self._llm_with_tools = self._llm_service.get_llm_with_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self._llm_with_tools

    async def _chat_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """聊天节点 - 生成 LLM 响应

        每次调用时递增迭代计数器。
        支持可配置的重试机制。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新
        """
        logger.debug("chat_node_entered", message_count=len(state["messages"]))

        llm = self._get_llm_with_tools()
        if not llm:
            raise RuntimeError("LLM 未初始化")

        # 构建 LCEL 链：prompt | llm
        chain = self._prompt_template | llm

        try:
            # 调用链（重试由 LLMService.with_retry 负责）
            response = await chain.ainvoke(
                {
                    "system_prompt": self._system_prompt,
                    "messages": state["messages"],
                },
                config,
            )

            logger.info(
                "llm_response_generated",
                model=self._llm_service.current_model,
                has_tool_calls=bool(hasattr(response, "tool_calls") and response.tool_calls),
            )

            # 递增迭代计数器（每次进入 chat_node 计为一次迭代）
            return {"messages": [response], "iteration_count": 1}

        except Exception as e:
            logger.exception("llm_call_failed_no_retry", error=str(e))
            error_message = AIMessage(
                content=f"抱歉，处理您的请求时出错：{str(e)}"
            )
            return {"messages": [error_message], "iteration_count": 1}

    async def _tools_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """工具节点 - 执行工具调用

        支持异步加载 MCP 工具。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新
        """
        logger.debug("tools_node_entered")

        tenant_id = None
        if isinstance(config, dict):
            tenant_id = config.get("metadata", {}).get("tenant_id")
        else:
            metadata = getattr(config, "metadata", None) or {}
            tenant_id = metadata.get("tenant_id")

        # 使用异步方式获取工具节点（包含 MCP 工具）
        tool_node = await self._aget_tool_node(tenant_id)
        result = await tool_node.ainvoke(state, config)

        logger.info(
            "tools_executed",
            tool_result_count=len(result.get("messages", [])),
        )

        return result

    def _get_tool_node(self) -> ToolNode:
        """获取或创建 ToolNode（缓存版本，同步）

        注意：同步版本不包含 MCP 工具。
        对于包含 MCP 工具的 ToolNode，请使用 _aget_tool_node()。

        Returns:
            ToolNode 实例
        """
        if self._tool_node_sync is None:
            from app.agent.tools import get_tool_node

            self._tool_node_sync = get_tool_node(include_mcp=False)
            logger.debug("tool_node_created", tool_count=len(list_tools(include_mcp=False)))
        return self._tool_node_sync

    async def _aget_tool_node(self, tenant_id: int | None) -> ToolNode:
        """异步获取或创建 ToolNode（包含 MCP 工具）

        Returns:
            ToolNode 实例（包含 MCP 工具）
        """
        if tenant_id not in self._tool_node_by_tenant:
            from app.agent.tools import aget_tool_node

            self._tool_node_by_tenant[tenant_id] = await aget_tool_node(
                include_mcp=True,
                tenant_id=tenant_id,
            )
            from app.agent.tools import alist_tools

            tools = await alist_tools(include_mcp=True, tenant_id=tenant_id)
            logger.debug("tool_node_created_async", tool_count=len(tools))
        return self._tool_node_by_tenant[tenant_id]

    def _build_graph(self) -> StateGraph:
        """构建 StateGraph

        Returns:
            StateGraph 实例
        """
        builder = StateGraph(AgentState)

        # 添加节点（重试逻辑已内置在节点函数中）
        builder.add_node("chat", self._chat_node)
        builder.add_node("tools", self._tools_node)

        # 设置入口点
        builder.set_entry_point("chat")

        # 添加条件边：根据是否有工具调用决定路由
        builder.add_conditional_edges("chat", route_by_tools, {"tools": "tools", "__end__": END})

        # 工具执行后返回聊天节点
        builder.add_edge("tools", "chat")

        logger.debug("chat_graph_structure_built", retry_enabled=True)
        return builder

    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        """编译图

        Args:
            checkpointer: 检查点保存器

        Returns:
            编译后的 StateGraph
        """
        if self._graph is None:
            builder = self._build_graph()

            # 默认使用 MemorySaver
            if checkpointer is None:
                checkpointer = MemorySaver()
                logger.debug("using_memory_checkpointer")

            self._checkpointer = checkpointer
            self._graph = builder.compile(checkpointer=checkpointer)

            logger.info(
                "chat_graph_compiled",
                has_checkpointer=checkpointer is not None,
            )

        return self._graph


def create_chat_graph(
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> ChatGraph:
    """创建对话图实例

    工厂函数，简化图的创建和编译。

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器

    Returns:
        已编译的 ChatGraph 实例

    Examples:
        ```python
        from langchain_core.messages import HumanMessage
        from app.agent.graphs import create_chat_graph

        graph = create_chat_graph(
            system_prompt="你是一个有用的助手。",
        )
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content="你好")]},
            {"configurable": {"thread_id": "session-123"}},
        )
        ```
    """
    chat_graph = ChatGraph(llm_service, system_prompt)
    chat_graph.compile(checkpointer=checkpointer)
    return chat_graph
