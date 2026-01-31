"""配置管理模块

提供环境配置、运行时配置、依赖注入和错误处理。
"""

from app.config.dependencies import (
    AgentContainer,
    get_agent_dep,
    get_checkpointer_dep,
    get_context_manager_dep,
    get_llm_service_dep,
    get_memory_manager_dep,
    get_memory_manager_factory_dep,
    get_settings_dep,
)
from app.config.errors import (
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    RetryStrategy,
    classify_error,
    default_retry_strategy,
    get_user_friendly_message,
    handle_tool_error,
)
from app.config.runtime import (
    AgentRuntimeConfig,
    Configuration,
    get_agent_runtime_config,
    get_runtime_config,
)
from app.config.settings import Settings, get_settings

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "get_settings_dep",
    # Runtime Config
    "Configuration",
    "AgentRuntimeConfig",
    "get_runtime_config",
    "get_agent_runtime_config",
    # Dependencies
    "AgentContainer",
    "get_llm_service_dep",
    "get_agent_dep",
    "get_memory_manager_dep",
    "get_memory_manager_factory_dep",
    "get_context_manager_dep",
    "get_checkpointer_dep",
    # Errors
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "RetryStrategy",
    "classify_error",
    "get_user_friendly_message",
    "handle_tool_error",
    "default_retry_strategy",
]
