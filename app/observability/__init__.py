"""可观测性模块

提供日志、指标、追踪、审计等可观测性功能。
"""

from app.observability.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    SensitiveDataMasker,
    audit_event,
    ensure_audit_logger_started,
    get_audit_logger,
    get_audit_logs,
    record_agent_event,
    record_event,
    record_tool_call,
)
from app.observability.log_sanitizer import (
    SafeLogFormatter,
    create_safe_log_message,
    sanitize_agent_name,
    sanitize_error_message,
    sanitize_feedback,
    sanitize_log_input,
    sanitize_thread_id,
    sanitize_tool_name,
    sanitize_user_content,
)
from app.observability.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    unbind_context,
)
from app.observability.metrics import (
    increment_active_sessions,
    increment_active_users,
    record_llm_tokens,
    track_http_request,
    track_llm_request,
    track_tool_call,
)

__all__ = [
    # logging
    "bind_context",
    "unbind_context",
    "clear_context",
    "configure_logging",
    "get_logger",
    # log_sanitizer
    "sanitize_log_input",
    "sanitize_thread_id",
    "sanitize_user_content",
    "sanitize_agent_name",
    "sanitize_tool_name",
    "sanitize_feedback",
    "sanitize_error_message",
    "create_safe_log_message",
    "SafeLogFormatter",
    # metrics
    "track_http_request",
    "track_llm_request",
    "track_tool_call",
    "record_llm_tokens",
    "increment_active_sessions",
    "increment_active_users",
    # audit
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "SensitiveDataMasker",
    "get_audit_logger",
    "ensure_audit_logger_started",
    "record_event",
    "record_agent_event",
    "record_tool_call",
    "get_audit_logs",
    "audit_event",
]
