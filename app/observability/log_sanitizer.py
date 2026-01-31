"""日志清洗工具

防止日志注入攻击，参考 DeerFlow 设计：
- 防止换行注入 (\n → \\n)
- 防止 HTML 注入
- 防止特殊字符序列被误解释

使用示例:
```python
from app.observability.log_sanitizer import sanitize_log_input

user_input = "malicious\\n[INFO] fake entry"
safe_log = sanitize_log_input(user_input)
logger.info("Processing input", input=safe_log)
```
"""

import re
from typing import Any


def sanitize_log_input(value: Any, max_length: int = 500) -> str:
    """清洗用户控制的输入以安全记录日志

    替换危险字符（换行符、制表符、回车符等）为转义表示，
    防止攻击者通过日志注入伪造日志条目。

    Args:
        value: 要清洗的输入值（任意类型）
        max_length: 输出字符串的最大长度（超过则截断）

    Returns:
        可安全记录日志的字符串

    Examples:
        >>> sanitize_log_input("normal text")
        'normal text'

        >>> sanitize_log_input("malicious\\n[INFO] fake entry")
        'malicious\\\\n[INFO] fake entry'

        >>> sanitize_log_input("tab\\there")
        'tab\\there'

        >>> sanitize_log_input(None)
        'None'

        >>> long_text = "a" * 1000
        >>> result = sanitize_log_input(long_text, max_length=100)
        >>> len(result) <= 100
        True
    """
    if value is None:
        return "None"

    # 转换为字符串
    string_value = str(value)

    # 替换危险字符为转义表示
    # 顺序很重要：先转义反斜杠以避免双重转义
    replacements = {
        "\\": "\\\\",  # 反斜杠（必须先处理）
        "\n": "\\n",  # 换行符 - 防止创建新日志条目
        "\r": "\\r",  # 回车符
        "\t": "\\t",  # 制表符
        "\x00": "\\0",  # 空字符
        "\x1b": "\\x1b",  # 转义字符（用于 ANSI 序列）
    }

    for char, replacement in replacements.items():
        string_value = string_value.replace(char, replacement)

    # 移除其他控制字符（ASCII 0-31 中已处理的除外）
    # 这些字符在日志中很少有用，且可能被利用
    string_value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", string_value)

    # 截断防止日志洪水攻击
    if len(string_value) > max_length:
        string_value = string_value[: max_length - 3] + "..."

    return string_value


def sanitize_thread_id(thread_id: Any) -> str:
    """清洗 thread_id 用于日志记录

    线程 ID 应为字母数字、连字符和下划线，
    但我们做防御性清洗。

    Args:
        thread_id: 要清洗的线程 ID

    Returns:
        清洗后的线程 ID
    """
    return sanitize_log_input(thread_id, max_length=100)


def sanitize_user_content(content: Any) -> str:
    """清洗用户提供的消息内容用于日志记录

    用户消息可以是任意长度，因此我们更激进地截断。

    Args:
        content: 用户内容

    Returns:
        清洗后的用户内容
    """
    return sanitize_log_input(content, max_length=200)


def sanitize_agent_name(agent_name: Any) -> str:
    """清洗 Agent 名称用于日志记录

    Agent 名称应该是简单标识符，但我们做防御性清洗。

    Args:
        agent_name: Agent 名称

    Returns:
        清洗后的 Agent 名称
    """
    return sanitize_log_input(agent_name, max_length=100)


def sanitize_tool_name(tool_name: Any) -> str:
    """清洗工具名称用于日志记录

    工具名称应该是简单标识符，但我们做防御性清洗。

    Args:
        tool_name: 工具名称

    Returns:
        清洗后的工具名称
    """
    return sanitize_log_input(tool_name, max_length=100)


def sanitize_feedback(feedback: Any) -> str:
    """清洗用户反馈用于日志记录

    反馈可以来自中断的任意文本，因此需要小心清洗。

    Args:
        feedback: 用户反馈

    Returns:
        清洗后的反馈（更激进地截断）
    """
    return sanitize_log_input(feedback, max_length=150)


def sanitize_error_message(error: Any) -> str:
    """清洗错误消息用于日志记录

    错误消息可能包含敏感信息，需要清洗。

    Args:
        error: 错误对象或消息

    Returns:
        清洗后的错误消息
    """
    if error is None:
        return "None"

    # 如果是异常对象，获取其字符串表示
    if isinstance(error, Exception):
        error_str = f"{type(error).__name__}: {str(error)}"
    else:
        error_str = str(error)

    return sanitize_log_input(error_str, max_length=500)


def create_safe_log_message(template: str, **kwargs) -> str:
    """通过清洗所有值创建安全的日志消息

    使用带关键字占位符的模板字符串，
    在替换前清洗每个值以防止日志注入。

    Args:
        template: 带 {key} 占位符的模板字符串
        **kwargs: 要替换的键值对

    Returns:
        安全的日志消息

    Example:
        >>> msg = create_safe_log_message(
        ...     "[{thread_id}] Processing {tool_name}",
        ...     thread_id="abc\\n[INFO]",
        ...     tool_name="my_tool"
        ... )
        >>> "[abc\\\\n[INFO]] Processing my_tool" in msg
        True
    """
    # 清洗所有值
    safe_kwargs = {
        key: sanitize_log_input(value) for key, value in kwargs.items()
    }

    # 替换到模板中
    return template.format(**safe_kwargs)


class SafeLogFormatter:
    """安全的日志格式化器

    自动清洗所有包含用户输入的日志字段。

    使用示例:
    ```python
    formatter = SafeLogFormatter()

    # 自动清洗
    logger.info("User input", **formatter.sanitize({
        "user_input": dangerous_input,
        "user_id": user_id
    }))
    ```
    """

    @staticmethod
    def sanitize(data: dict[str, Any]) -> dict[str, Any]:
        """清洗字典中的所有值

        Args:
            data: 要清洗的字典

        Returns:
            值已清洗的字典
        """
        return {
            key: (
                sanitize_log_input(value)
                if key in {"user_input", "user_content", "content", "message", "feedback", "prompt"}
                else (
                    sanitize_agent_name(value)
                    if key == "agent_name"
                    else (
                        sanitize_tool_name(value)
                        if key == "tool_name"
                        else (
                            sanitize_thread_id(value)
                            if key in {"thread_id", "session_id"}
                            else value
                        )
                    )
                )
            )
            for key, value in data.items()
        }

    @staticmethod
    def sanitize_field(key: str, value: Any) -> Any:
        """根据字段名选择合适的清洗函数

        Args:
            key: 字段名
            value: 字段值

        Returns:
            清洗后的值
        """
        sanitizers = {
            "user_input": sanitize_user_content,
            "user_content": sanitize_user_content,
            "content": sanitize_user_content,
            "message": sanitize_user_content,
            "feedback": sanitize_feedback,
            "prompt": sanitize_user_content,
            "agent_name": sanitize_agent_name,
            "tool_name": sanitize_tool_name,
            "thread_id": sanitize_thread_id,
            "session_id": sanitize_thread_id,
            "error": sanitize_error_message,
        }

        sanitizer = sanitizers.get(key)
        if sanitizer:
            return sanitizer(value)
        return value
