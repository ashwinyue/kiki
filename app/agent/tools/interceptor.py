"""工具执行拦截器

支持在工具执行前进行中断和人工审批，参考 DeerFlow 设计。

使用示例:
```python
from app.agent.tools.interceptor import wrap_tools_with_interceptor

tools = [search_tool, database_tool]

# 需要审批的工具
interrupt_tools = ["database_delete", "send_email"]

wrapped_tools = wrap_tools_with_interceptor(tools, interrupt_tools)
agent = create_react_agent(llm, wrapped_tools)
```
"""

import json
import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from app.observability.log_sanitizer import (
    sanitize_agent_name,
    sanitize_log_input,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)


class ToolInterceptor:
    """工具拦截器

    在工具执行前拦截并触发中断，等待用户审批。

    Attributes:
        interrupt_before_tools: 需要中断的工具名称列表
    """

    def __init__(self, interrupt_before_tools: list[str] | None = None):
        """初始化拦截器

        Args:
            interrupt_before_tools: 需要在执行前中断的工具名称列表。
                                   如果为 None 或空，则不触发中断。
        """
        self.interrupt_before_tools = interrupt_before_tools or []
        logger.info(
            "tool_interceptor_initialized",
            interrupt_tools=len(self.interrupt_before_tools),
            tools=self.interrupt_before_tools,
        )

    def should_interrupt(self, tool_name: str) -> bool:
        """检查是否应在此工具前中断

        Args:
            tool_name: 工具名称

        Returns:
            如果工具应触发中断则返回 True，否则返回 False
        """
        should_interrupt = tool_name in self.interrupt_before_tools
        if should_interrupt:
            logger.info(
                "tool_marked_for_interrupt",
                tool_name=sanitize_tool_name(tool_name),
            )
        return should_interrupt

    @staticmethod
    def _format_tool_input(tool_input: Any) -> str:
        """格式化工具输入以便在中断消息中显示

        尝试格式化为 JSON 以提高可读性，失败则回退到字符串表示。

        Args:
            tool_input: 要格式化的工具输入

        Returns:
            工具输入的格式化表示
        """
        if tool_input is None:
            return "No input"

        # 首先尝试序列化为 JSON 以提高可读性
        try:
            # 处理可 JSON 序列化的对象
            if isinstance(tool_input, (dict, list, tuple)):
                return json.dumps(tool_input, indent=2, default=str, ensure_ascii=False)
            if isinstance(tool_input, str):
                return tool_input
            # 其他类型，尝试转换为 dict
            if hasattr(tool_input, "__dict__"):
                return json.dumps(tool_input.__dict__, indent=2, default=str, ensure_ascii=False)
            # 回退到字符串表示
            return str(tool_input)
        except (TypeError, ValueError):
            # JSON 序列化失败，使用字符串表示
            return str(tool_input)

    @staticmethod
    def wrap_tool(
        tool: BaseTool, interceptor: "ToolInterceptor"
    ) -> BaseTool:
        """包装工具以添加中断逻辑

        通过创建包装器来添加中断能力。

        Args:
            tool: 要包装的工具
            interceptor: ToolInterceptor 实例

        Returns:
            具有中断能力的包装工具
        """
        original_func = tool.func
        safe_tool_name = sanitize_tool_name(tool.name)
        logger.debug("wrapping_tool", tool_name=safe_tool_name)

        def intercepted_func(*args: Any, **kwargs: Any) -> Any:
            """执行带有中断检查的工具"""
            tool_name = tool.name
            safe_tool_name = sanitize_tool_name(tool_name)
            logger.debug("executing_tool", tool_name=safe_tool_name)

            # 格式化工具输入用于显示
            tool_input = args[0] if args else kwargs
            tool_input_repr = ToolInterceptor._format_tool_input(tool_input)
            safe_tool_input = sanitize_log_input(tool_input_repr, max_length=100)
            logger.debug("tool_input", tool_name=safe_tool_name, input=safe_tool_input)

            should_interrupt = interceptor.should_interrupt(tool_name)
            logger.debug(
                "should_interrupt_check",
                tool_name=safe_tool_name,
                should_interrupt=should_interrupt,
            )

            if should_interrupt:
                logger.info(
                    "interrupting_before_tool",
                    tool_name=safe_tool_name,
                    input=safe_tool_input,
                )

                # 触发中断并等待用户反馈
                try:
                    feedback = interrupt(
                        f"即将执行工具: '{tool_name}'\\n\\n输入:\\n{tool_input_repr}\\n\\n是否批准执行?"
                    )
                    safe_feedback = sanitize_log_input(feedback, max_length=100)
                    logger.debug("interrupt_feedback", feedback=safe_feedback)
                except Exception as e:
                    logger.error("interrupt_error", error=str(e))
                    raise

                # 检查用户是否批准
                is_approved = ToolInterceptor._parse_approval(feedback)
                logger.info(
                    "tool_approval_decision",
                    tool_name=safe_tool_name,
                    approved=is_approved,
                )

                if not is_approved:
                    logger.warning("tool_execution_rejected", tool_name=safe_tool_name)
                    return {
                        "error": "工具执行被用户拒绝",
                        "tool": tool_name,
                        "status": "rejected",
                    }

                logger.info("tool_execution_approved", tool_name=safe_tool_name)

            # 执行原始工具
            try:
                logger.debug("calling_original_function", tool_name=safe_tool_name)
                result = original_func(*args, **kwargs)
                logger.info("tool_execution_completed", tool_name=safe_tool_name)
                return result
            except Exception as e:
                logger.error(
                    "tool_execution_error",
                    tool_name=safe_tool_name,
                    error=str(e),
                )
                raise

        # 替换函数并更新工具
        # 使用 object.__setattr__ 绕过 Pydantic 验证
        logger.debug("attaching_intercepted_function", tool_name=safe_tool_name)
        object.__setattr__(tool, "func", intercepted_func)
        return tool

    @staticmethod
    def _parse_approval(feedback: str) -> bool:
        """解析用户反馈以确定工具执行是否被批准

        Args:
            feedback: 用户的反馈字符串

        Returns:
            如果反馈表示批准则返回 True，否则返回 False
        """
        if not feedback:
            logger.warning("empty_feedback_received")
            return False

        feedback_lower = feedback.lower().strip()

        # 检查批准关键词
        approval_keywords = [
            "approved",
            "approve",
            "yes",
            "y",
            "proceed",
            "continue",
            "ok",
            "okay",
            "accepted",
            "accept",
            "[approved]",
            "批准",
            "同意",
            "继续",
            "确认",
            "是",
        ]

        for keyword in approval_keywords:
            if keyword in feedback_lower:
                return True

        # 如果没有找到批准关键词，默认拒绝
        logger.warning("no_approval_keywords_found", feedback=feedback_lower[:100])
        return False


def wrap_tools_with_interceptor(
    tools: list[BaseTool],
    interrupt_before_tools: list[str] | None = None,
) -> list[BaseTool]:
    """用中断逻辑包装多个工具

    Args:
        tools: 要包装的工具列表
        interrupt_before_tools: 需要中断的工具名称列表

    Returns:
        包装后的工具列表

    Examples:
        >>> tools = [search_tool, delete_tool]
        >>> wrapped = wrap_tools_with_interceptor(
        ...     tools,
        ...     interrupt_before_tools=["delete"]
        ... )
        >>> # delete_tool 现在会在执行前等待用户批准
    """
    if not interrupt_before_tools:
        logger.debug("no_tool_interrupts_configured")
        return tools

    logger.info(
        "wrapping_tools_with_interceptor",
        total_tools=len(tools),
        interrupt_tools=interrupt_before_tools,
    )
    interceptor = ToolInterceptor(interrupt_before_tools)

    wrapped_tools = []
    for tool in tools:
        try:
            wrapped_tool = ToolInterceptor.wrap_tool(tool, interceptor)
            wrapped_tools.append(wrapped_tool)
            logger.debug("tool_wrapped", tool_name=tool.name)
        except Exception as e:
            logger.error("tool_wrap_failed", tool_name=tool.name, error=str(e))
            # 如果包装失败，添加原始工具
            wrapped_tools.append(tool)

    logger.info("tools_wrapped", count=len(wrapped_tools))
    return wrapped_tools


class ToolExecutionResult:
    """工具执行结果

    Attributes:
        tool_name: 工具名称
        success: 是否成功
        result: 执行结果
        error: 错误信息（如果有）
        status: 状态（completed/rejected/error）
    """

    def __init__(
        self,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
        status: str = "completed",
    ):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "status": self.status,
        }


def create_tool_interceptor(
    interrupt_before_tools: list[str] | None = None,
    approval_callback: Callable[[str, str], bool] | None = None,
) -> ToolInterceptor:
    """创建工具拦截器的工厂函数

    Args:
        interrupt_before_tools: 需要中断的工具名称列表
        approval_callback: 自定义审批回调函数，接收 (tool_name, tool_input) 返回 bool

    Returns:
        ToolInterceptor 实例
    """
    if approval_callback:
        # 如果提供了自定义回调，创建自定义拦截器
        class CustomToolInterceptor(ToolInterceptor):
            def __init__(self, tools: list[str], callback: Callable):
                self.interrupt_before_tools = tools
                self.approval_callback = callback
                super().__init__(tools)

            def should_interrupt(self, tool_name: str) -> bool:
                if tool_name in self.interrupt_before_tools:
                    # 使用自定义回调
                    tool_input = "pending execution"
                    return self.approval_callback(tool_name, tool_input)
                return False

        return CustomToolInterceptor(interrupt_before_tools or [], approval_callback)

    return ToolInterceptor(interrupt_before_tools)
