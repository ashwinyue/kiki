"""日志配置

使用 structlog 实现结构化日志。
"""

import structlog


def configure_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_format: str = "console",
) -> None:
    """配置 structlog"""
    is_production = environment == "production"
    should_json = log_format == "json" or is_production

    if should_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
            if not is_production
            else structlog.processors.CallsiteParameterAdder([]),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
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
