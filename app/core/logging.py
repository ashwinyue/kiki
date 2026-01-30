"""日志配置

使用 structlog 实现结构化日志。

使用示例:
```python
from app.core.logging import get_logger, bind_context

log = get_logger(__name__)
log.info("event_occurred", user_id="123", action="login")

# 绑定上下文
bind_context(request_id="abc-123")
log.info("request_processed")  # 自动包含 request_id
```
"""

from typing import Any

import structlog


def configure_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_format: str = "console",
) -> None:
    """配置 structlog

    Args:
        environment: 环境 (development/staging/production)
        log_level: 日志级别
        log_format: 日志格式 (console/json)
    """
    is_production = environment == "production"
    should_json = log_format == "json" or is_production

    # 选择渲染器
    if should_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # 配置 structlog
    structlog.configure(
        processors=[
            # 合并上下文变量
            structlog.contextvars.merge_contextvars,
            # 添加日志级别
            structlog.stdlib.add_log_level,
            # 添加 logger 名称
            structlog.stdlib.add_logger_name,
            # 添加时间戳
            structlog.processors.TimeStamper(fmt="iso"),
            # 添加调用信息（仅在调试模式）
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
            if not is_production
            else structlog.processors.CallsiteParameterAdder([]),
            # 异常信息
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Unicode 解码
            structlog.processors.UnicodeDecoder(),
            # 渲染器
            renderer,
        ],
        # 包装类（添加过滤功能）
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        # 上下文类
        context_class=dict,
        # Logger 工厂（使用标准库 logger）
        logger_factory=structlog.stdlib.LoggerFactory(),
        # 缓存 logger
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **kwargs) -> structlog.stdlib.BoundLogger:
    """获取日志记录器

    Args:
        name: 日志记录器名称（通常使用 __name__）
        **kwargs: 额外的上下文变量

    Returns:
        BoundLogger 实例

    Examples:
        ```python
        log = get_logger(__name__)
        log.info("Processing request", action="process", item_id="456")

        # 或者带上下文变量
        log = get_logger(__name__, user_id="123")
        log.info("Processing request")
        ```
    """
    if name:
        kwargs["name"] = name
    return structlog.get_logger(**kwargs)


def bind_context(**kwargs) -> None:
    """绑定上下文变量（所有日志自动包含）

    Args:
        **kwargs: 上下文变量

    Examples:
        ```python
        bind_context(request_id="abc-123", user_id="456")
        log = get_logger()
        log.info("Request processed")  # 自动包含 request_id 和 user_id
        ```
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """解绑上下文变量

    Args:
        *keys: 要解绑的变量名
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """清空所有上下文变量"""
    structlog.contextvars.clear_contextvars()
