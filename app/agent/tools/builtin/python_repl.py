"""Python REPL 工具

安全的 Python 代码执行环境，用于数据分析场景。
使用 langchain-experimental 的 PythonREPL 实现。

依赖安装:
    uv add langchain-experimental

使用示例:
```python
from app.agent.tools.builtin.python_repl import python_repl

result = await python_repl("print('Hello, World!')")
# => Hello, World!

result = await python_repl("import pandas as pd; df = pd.DataFrame({'a': [1,2,3]}); df.describe()")
# => 返回 DataFrame 的统计信息
```
"""

import asyncio
import functools
import io
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from langchain_core.tools import tool

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
try:
    from langchain_experimental.utilities import PythonREPL as LangChainPythonREPL

    _python_repl_available = True
except ImportError:
    _python_repl_available = False
    LangChainPythonREPL = None  # type: ignore
    logger.warning("langchain_experimental_not_installed")

# 默认超时
_DEFAULT_TIMEOUT = 30.0

# 禁止的操作
_BUILTIN_IMPORTS = {
    "os": ["system", "popen", "spawn", "fork", "exec", "remove", "rmdir"],
    "subprocess": ["run", "call", "Popen", "check_output"],
    "shutil": ["rmtree", "move", "copy"],
    "requests": ["post", "put", "delete", "patch"],
    "http": ["client"],
}

# 禁止的函数
_BANNED_FUNCTIONS = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "globals",
    "locals",
    "vars",
    "dir",
}


def _run_in_executor(timeout: float | None = None):
    """装饰器：将同步函数在线程池中执行

    Args:
        timeout: 超时时间（秒），None 表示不超时

    Returns:
        装饰后的异步函数
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, func, *args, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        return wrapper
    return decorator


def _validate_code(code: str) -> tuple[bool, str | None]:
    """验证代码安全性

    Args:
        code: 要执行的代码

    Returns:
        (是否安全, 错误信息)
    """
    if not code:
        return False, "代码为空"

    code_lower = code.lower()

    # 检查禁止的函数
    for func in _BANNED_FUNCTIONS:
        if func in code:
            # 允许 dir() 用于对象检查
            if func == "dir" and "dir(" in code and "dir()" not in code:
                continue
            return False, f"禁止使用函数: {func}"

    # 检查禁止的导入
    for module, banned_funcs in _BUILTIN_IMPORTS.items():
        if f"import {module}" in code_lower or f"from {module}" in code_lower:
            for func in banned_funcs:
                if f"{module}.{func}" in code_lower or f"{func}(" in code:
                    return False, f"禁止使用: {module}.{func}"

    # 检查文件操作
    if any(keyword in code_lower for keyword in ["open(", "file(", "with open"]):
        return False, "禁止文件操作"

    # 检查网络操作
    if any(keyword in code_lower for keyword in ["requests.", "urllib.", "httpx.", "socket."]):
        return False, "禁止网络操作"

    # 检查系统命令
    if any(keyword in code_lower for keyword in ["os.system", "subprocess.", "commands.", "popen"]):
        return False, "禁止系统命令"

    return True, None


class SafePythonREPL:
    """安全的 Python REPL 执行环境

    提供受限的 Python 执行环境，适合数据分析场景。
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT):
        """初始化 REPL

        Args:
            timeout: 执行超时时间（秒）
        """
        self.timeout = timeout
        self._repl = LangChainPythonREPL() if _python_repl_available else None
        self._globals: dict[str, Any] = {}
        self._locals: dict[str, Any] = {}

    def run(self, code: str) -> str:
        """执行 Python 代码

        Args:
            code: Python 代码

        Returns:
            执行结果字符串
        """
        if not _python_repl_available:
            return "错误: PythonREPL 不可用，请安装 langchain-experimental"

        # 验证代码
        is_safe, error = _validate_code(code)
        if not is_safe:
            logger.warning("unsafe_code_blocked", code=code[:100], error=error)
            return f"安全拒绝: {error}"

        try:
            # 使用 langchain-experimental 的 PythonREPL
            result = self._repl.run(code)

            if result.startswith("Error:"):
                logger.error("python_repl_error", result=result)
                return result

            logger.info("python_repl_success", code_length=len(code))
            return result

        except Exception as e:
            logger.error("python_repl_exception", error=str(e))
            return f"执行异常: {str(e)}"

    def run_with_capture(self, code: str) -> str:
        """执行代码并捕获输出

        Args:
            code: Python 代码

        Returns:
            执行结果和输出
        """
        if not _python_repl_available:
            return "错误: PythonREPL 不可用，请安装 langchain-experimental"

        # 验证代码
        is_safe, error = _validate_code(code)
        if not is_safe:
            return f"安全拒绝: {error}"

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = self._repl.run(code)

            # 获取捕获的输出
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # 组合结果
            parts = []
            if stdout_output:
                parts.append(f"输出:\n{stdout_output}")
            if stderr_output and "Error:" not in result:
                parts.append(f"错误输出:\n{stderr_output}")
            if result and result != "None":
                parts.append(f"结果:\n{result}")

            return "\n\n".join(parts) if parts else "执行成功（无输出）"

        except Exception as e:
            logger.error("python_repl_exception", error=str(e))
            return f"执行异常: {str(e)}"


# 全局 REPL 实例
_repl_instance: SafePythonREPL | None = None


def get_repl() -> SafePythonREPL:
    """获取全局 REPL 实例

    Returns:
        SafePythonREPL 实例
    """
    global _repl_instance
    if _repl_instance is None:
        _repl_instance = SafePythonREPL()
    return _repl_instance


@_run_in_executor(timeout=_DEFAULT_TIMEOUT)
def _sync_run_python(code: str, capture_output: bool = False) -> str:
    """同步执行 Python 代码

    Args:
        code: Python 代码
        capture_output: 是否捕获输出

    Returns:
        执行结果
    """
    repl = get_repl()
    if capture_output:
        return repl.run_with_capture(code)
    return repl.run(code)


@tool
async def python_repl(
    code: str,
    capture_output: bool = False,
) -> str:
    """安全的 Python 代码执行工具

    用于数据分析和计算任务。支持常用的数据科学库（pandas, numpy, math 等）。

    **安全限制**:
    - 禁止文件操作（open, file）
    - 禁止网络操作（requests, urllib）
    - 禁止系统命令（os.system, subprocess）
    - 禁止危险函数（eval, exec, __import__）

    Args:
        code: 要执行的 Python 代码
        capture_output: 是否捕获 print 输出

    Returns:
        执行结果或错误信息

    Examples:
        >>> # 简单计算
        >>> result = await python_repl("2 + 2")
        >>> # => 4

        >>> # 数据分析
        >>> result = await python_repl('''
        ... import pandas as pd
        ... df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df.sum()
        ... ''')
        >>> # => a     6\\nb    15\\ndtype: int64

        >>> # 捕获 print 输出
        >>> result = await python_repl('''
        ... for i in range(3):
        ...     print(f"Item {i}")
        ... ''', capture_output=True)
        >>> # => 输出:\\nItem 0\\nItem 1\\nItem 2
    """
    logger.info(
        "python_repl_called",
        code_length=len(code),
        capture_output=capture_output,
    )

    try:
        result = await _sync_run_python(code, capture_output)
        logger.info("python_repl_completed")
        return result
    except TimeoutError:
        logger.warning("python_repl_timeout")
        return f"执行超时（{_DEFAULT_TIMEOUT}秒）"
    except Exception as e:
        logger.error("python_repl_failed", error=str(e))
        return f"执行失败: {str(e)}"


def is_python_repl_available() -> bool:
    """检查 Python REPL 是否可用

    Returns:
        是否可用
    """
    return _python_repl_available


__all__ = [
    "python_repl",
    "SafePythonREPL",
    "get_repl",
    "is_python_repl_available",
]
