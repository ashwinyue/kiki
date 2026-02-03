"""工具装饰器

提供工具日志记录和性能监控功能，参考 DeerFlow 设计。

使用示例:
```python
from app.agent.tools.decorators import log_io, create_logged_tool
from langchain_community.tools import TavilySearchResults

# 方式1: 使用装饰器
@log_io
@tool
def my_tool(input: str) -> str:
    return "result"

# 方式2: 创建日志版本的工具类
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
```
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.observability.log_sanitizer import (
    sanitize_log_input,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_io(func: Callable) -> Callable:
    """记录工具输入输出的装饰器

    Args:
        func: 要装饰的工具函数

    Returns:
        带有输入输出日志记录的包装函数

    Examples:
        >>> @log_io
        ... @tool
        ... def my_tool(input: str) -> str:
        ...     return "result"
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = getattr(func, "__name__", str(func))
        safe_tool_name = sanitize_tool_name(func_name)

        params = []
        for arg in args:
            params.append(sanitize_log_input(arg, max_length=100))
        for k, v in kwargs.items():
            params.append(f"{k}={sanitize_log_input(v, max_length=100)}")

        logger.info(
            "tool_called",
            tool_name=safe_tool_name,
            params=", ".join(params),
        )

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            logger.info(
                "tool_completed",
                tool_name=safe_tool_name,
                elapsed_ms=round(elapsed * 1000, 2),
                result_length=len(str(result)) if result else 0,
            )

            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "tool_failed",
                tool_name=safe_tool_name,
                elapsed_ms=round(elapsed * 1000, 2),
                error=str(e),
            )
            raise

    return wrapper


async def log_io_async(func: Callable) -> Callable:
    """记录工具输入输出的异步装饰器

    Args:
        func: 要装饰的异步工具函数

    Returns:
        带有输入输出日志记录的异步包装函数
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = getattr(func, "__name__", str(func))
        safe_tool_name = sanitize_tool_name(func_name)

        params = []
        for arg in args:
            params.append(sanitize_log_input(arg, max_length=100))
        for k, v in kwargs.items():
            params.append(f"{k}={sanitize_log_input(v, max_length=100)}")

        logger.info(
            "tool_called_async",
            tool_name=safe_tool_name,
            params=", ".join(params),
        )

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time

            # 记录输出
            logger.info(
                "tool_completed_async",
                tool_name=safe_tool_name,
                elapsed_ms=round(elapsed * 1000, 2),
                result_length=len(str(result)) if result else 0,
            )

            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "tool_failed_async",
                tool_name=safe_tool_name,
                elapsed_ms=round(elapsed * 1000, 2),
                error=str(e),
            )
            raise

    return wrapper


class LoggedToolMixin:
    """为工具类添加日志功能的 Mixin 类

    通过重写 _run 方法为任何工具类添加日志记录。

    Examples:
        >>> from langchain_community.tools import TavilySearchResults
        >>>
        >>> class LoggedTavilySearch(LoggedToolMixin, TavilySearchResults):
        ...     pass
    """

    def _log_operation(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> None:
        """记录工具操作的辅助方法

        Args:
            method_name: 方法名称
            *args: 位置参数
            **kwargs: 关键字参数
        """
        tool_name = self.__class__.__name__.replace("Logged", "")
        safe_tool_name = sanitize_tool_name(tool_name)
        params = []
        for arg in args:
            params.append(sanitize_log_input(arg, max_length=100))
        for k, v in kwargs.items():
            params.append(f"{k}={sanitize_log_input(v, max_length=100)}")

        logger.debug(
            "tool_operation",
            tool_name=safe_tool_name,
            method=method_name,
            params=", ".join(params),
        )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """重写 _run 方法以添加日志记录"""
        self._log_operation("_run", *args, **kwargs)

        start_time = time.time()
        result = super()._run(*args, **kwargs)  # type: ignore
        elapsed = time.time() - start_time

        tool_name = self.__class__.__name__.replace("Logged", "")
        safe_tool_name = sanitize_tool_name(tool_name)

        logger.debug(
            "tool_executed",
            tool_name=safe_tool_name,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        return result

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """重写 _arun 方法以添加日志记录"""
        self._log_operation("_arun", *args, **kwargs)

        start_time = time.time()
        result = await super()._arun(*args, **kwargs)  # type: ignore
        elapsed = time.time() - start_time

        tool_name = self.__class__.__name__.replace("Logged", "")
        safe_tool_name = sanitize_tool_name(tool_name)

        logger.debug(
            "tool_executed_async",
            tool_name=safe_tool_name,
            elapsed_ms=round(elapsed * 1000, 2),
        )
        return result


def create_logged_tool(base_tool_class: type[T]) -> type[T]:
    """创建带有日志功能的工具类

    工厂函数，通过多重继承为任何工具类添加日志功能。

    Args:
        base_tool_class: 要增强日志功能的原始工具类

    Returns:
        继承自 LoggedToolMixin 和基础工具类的新类

    Examples:
        >>> from langchain_community.tools import (
        ...     TavilySearchResults,
        ...     DuckDuckGoSearchResults,
        ... )
        >>>
        >>> LoggedTavilySearch = create_logged_tool(TavilySearchResults)
        >>> LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
    """

    class LoggedTool(LoggedToolMixin, base_tool_class):
        pass

    # 设置更具描述性的类名
    LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
    LoggedTool.__qualname__ = f"Logged{base_tool_class.__qualname__}"

    return LoggedTool


def track_tool_metrics(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """记录工具指标的装饰器

    Args:
        func: 要装饰的函数
        name: 工具名称（默认使用函数名）

    Returns:
        装饰后的函数

    Examples:
        >>> @track_tool_metrics
        ... @tool
        ... def my_tool(input: str) -> str:
        ...     return "result"

        >>> @track_tool_metrics(name="custom_name")
        ... @tool
        ... def my_tool(input: str) -> str:
        ...     return "result"
    """
    from app.observability.metrics import track_tool_call

    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - start_time

                # 记录工具调用指标
                track_tool_call(tool_name, elapsed, success=True)

                return result
            except Exception as e:
                elapsed = time.time() - start_time
                track_tool_call(tool_name, elapsed, success=False, error=str(e))
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
