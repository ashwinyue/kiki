"""数学计算工具

示例工具，展示如何创建计算类工具。
"""

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)


@tool
async def calculate(expression: str) -> str:
    """计算数学表达式

    安全地计算数学表达式并返回结果。
    支持基本运算：加减乘除、括号等。

    Args:
        expression: 数学表达式（如 "2 + 2", "10 * 5", "(3 + 4) * 2"）

    Returns:
        计算结果字符串

    Examples:
        ```python
        result = await calculate("2 + 2")
        # 返回: "计算结果: 4"

        result = await calculate("(3 + 4) * 2")
        # 返回: "计算结果: 14"
        ```
    """
    logger.info("calculate_called", expression=expression)

    try:
        # 使用受限的 eval 计算数学表达式
        # 注意：这不是完全安全的，只应在受信任的环境中使用
        # 对于生产环境，建议使用 simpleeval 等专门的库
        allowed_names = {}
        result = eval(expression, {"__builtins__": None}, allowed_names)  # noqa: S307
        return f"计算结果: {result}"
    except Exception as e:
        logger.warning("calculate_failed", expression=expression, error=str(e))
        return f"计算错误: {str(e)}"
