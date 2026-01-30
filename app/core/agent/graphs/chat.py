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
from langgraph.types import RunnableConfig

from app.core.agent.graphs.base import BaseGraph
from app.core.agent.state import AgentState
from app.core.agent.tools import get_tool_node, list_tools
from app.core.llm import LLMService, get_llm_service
from app.core.logging import get_logger

logger = get_logger(__name__)


def route_by_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """根据是否有工具调用决定路由

    Args:
        state: 当前状态

    Returns:
        下一个节点名称
    """
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
        from app.core.agent.graphs import ChatGraph

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

        Returns:
            绑定工具的 LLM 实例
        """
        if self._llm_with_tools is None:
            tools = list_tools()
            llm = self._llm_service.get_llm()
            if llm and tools:
                self._llm_with_tools = llm.bind_tools(tools)
                logger.debug("tools_bound_to_llm", tool_count=len(tools))
            else:
                self._llm_with_tools = llm
        return self._llm_with_tools

    async def _chat_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """聊天节点 - 生成 LLM 响应

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新
        """
        logger.debug("chat_node_entered", message_count=len(state["messages"]))

        try:
            llm = self._get_llm_with_tools()
            if not llm:
                raise RuntimeError("LLM 未初始化")

            # 构建 LCEL 链：prompt | llm
            chain = self._prompt_template | llm

            # 调用链
            response = await chain.ainvoke({
                "system_prompt": self._system_prompt,
                "messages": state["messages"],
            }, config)

            logger.info(
                "llm_response_generated",
                model=self._llm_service.current_model,
                has_tool_calls=bool(hasattr(response, "tool_calls") and response.tool_calls),
            )

            return {"messages": [response]}

        except Exception as e:
            logger.exception("llm_call_failed", error=str(e))
            error_message = AIMessage(content=f"抱歉，处理您的请求时出错：{str(e)}")
            return {"messages": [error_message]}

    async def _tools_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """工具节点 - 执行工具调用

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新
        """
        logger.debug("tools_node_entered")

        tool_node = get_tool_node()
        result = await tool_node.ainvoke(state, config)

        logger.info(
            "tools_executed",
            tool_result_count=len(result.get("messages", [])),
        )

        return result

    def _build_graph(self) -> StateGraph:
        """构建 StateGraph

        Returns:
            StateGraph 实例
        """
        builder = StateGraph(AgentState)

        # 添加节点
        builder.add_node("chat", self._chat_node)
        builder.add_node("tools", self._tools_node)

        # 设置入口点
        builder.set_entry_point("chat")

        # 添加条件边：根据是否有工具调用决定路由
        builder.add_conditional_edges(
            "chat",
            route_by_tools,
            {"tools": "tools", "__end__": END}
        )

        # 工具执行后返回聊天节点
        builder.add_edge("tools", "chat")

        logger.debug("chat_graph_structure_built")
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
        from app.core.agent.graphs import create_chat_graph

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
