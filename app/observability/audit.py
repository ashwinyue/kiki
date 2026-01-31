"""审计日志模块

提供企业级审计日志功能，参考 WeKnora99 的事件驱动架构。
支持敏感操作记录、数据脱敏、异步持久化。

使用示例:
```python
from app.observability.audit import (
    AuditEvent,
    AuditLogger,
    record_agent_event,
    get_audit_logs,
)

# 记录 Agent 事件
await record_agent_event(
    event_type="agent:tool_call",
    agent_id="chat-agent",
    user_id="user-123",
    data={"tool": "search_web", "query": "..."}
)

# 查询审计日志
logs = await get_audit_logs(user_id="user-123", limit=100)
```
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.observability.log_sanitizer import sanitize_log_input
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ============== 事件类型定义 ==============

class AuditEventType(str, Enum):
    """审计事件类型

    参考 WeKnora99 的事件类型设计，覆盖 Agent 全生命周期。
    """

    # Agent 生命周期
    AGENT_STARTED = "agent:started"
    AGENT_COMPLETED = "agent:completed"
    AGENT_FAILED = "agent:failed"
    AGENT_MESSAGE = "agent:message"

    # Multi-Agent 协作
    MULTI_AGENT_SYNC = "multiagent:sync"
    MULTI_AGENT_HANDOFF = "multiagent:handoff"

    # 工具调用
    TOOL_CALL_START = "tool:call_start"
    TOOL_CALL_SUCCESS = "tool:call_success"
    TOOL_CALL_FAILED = "tool:call_failed"
    TOOL_CALL_BLOCKED = "tool:call_blocked"

    # 检索相关
    RETRIEVAL_START = "retrieval:start"
    RETRIEVAL_SUCCESS = "retrieval:success"
    RETRIEVAL_FAILED = "retrieval:failed"

    # 用户交互
    QUERY_RECEIVED = "query:received"
    RESPONSE_SENT = "response:sent"
    CLARIFICATION_REQUEST = "clarification:request"
    CLARIFICATION_RESPONSE = "clarification:response"

    # Human-in-the-Loop
    INTERRUPT_TRIGGERED = "interrupt:triggered"
    INTERRUPT_APPROVED = "interrupt:approved"
    INTERRUPT_REJECTED = "interrupt:rejected"

    # 安全相关
    AUTH_SUCCESS = "auth:success"
    AUTH_FAILED = "auth:failed"
    PERMISSION_DENIED = "permission:denied"

    # 系统事件
    ERROR = "system:error"
    WARNING = "system:warning"
    CONFIG_CHANGED = "config:changed"


# ============== 敏感数据脱敏 ==============

class SensitiveDataMasker:
    """敏感数据脱敏器

    对审计日志中的敏感数据进行脱敏处理。
    """

    # 敏感字段模式
    PATTERNS = {
        "api_key": r"""(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]+)""",
        "token": r"""(?i)(token|access[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]+)""",
        "password": r"""(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^'\"\s]+)""",
        "secret": r"""(?i)(secret)\s*[:=]\s*['\"]?([^'\"\s]+)""",
        "credit_card": r"""\b(?:\d[ -]*?){13,16}\b""",
        "email": r"""([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})""",
        "phone": r"""(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})""",
        "ip": r"""(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})""",
    }

    def __init__(self, mask_char: str = "***"):
        """初始化脱敏器

        Args:
            mask_char: 脱敏字符
        """
        self.mask_char = mask_char

    def mask(self, text: str, patterns: list[str] | None = None) -> str:
        """脱敏处理

        Args:
            text: 原始文本
            patterns: 自定义模式列表（可选）

        Returns:
            脱敏后的文本
        """
        if not text:
            return text

        result = text
        all_patterns = patterns or list(self.PATTERNS.keys())

        for pattern_key in all_patterns:
            pattern = self.PATTERNS.get(pattern_key)
            if pattern:

                def replace_match(m):
                    return f"{m.group(1)}{self.mask_char}"

                import re

                result = re.sub(pattern, replace_match, result, flags=re.IGNORECASE)

        return result

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """脱敏字典

        Args:
            data: 原始字典

        Returns:
            脱敏后的字典
        """
        if not data:
            return data

        result = {}

        for key, value in data.items():
            # 检查是否是敏感字段
            if any(keyword in key.lower() for keyword in ["api", "key", "token", "secret", "password", "credit", "card", "email", "phone", "ip"]):
                if isinstance(value, str):
                    value = self.mask(value)
                elif isinstance(value, dict):
                    value = self.mask_dict(value)

            result[key] = value

        return result


_masker = SensitiveDataMasker()


# ============== 审计事件数据类 ==============

@dataclass
class AuditEvent:
    """审计事件

    Attributes:
        id: 事件唯一标识
        event_type: 事件类型
        timestamp: 事件时间戳
        user_id: 用户 ID
        session_id: 会话 ID
        agent_id: Agent ID
        trace_id: 追踪 ID（用于链路追踪）
        data: 事件数据
        metadata: 元数据
        status: 事件状态
        error: 错误信息（如果有）
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    trace_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: str | None = None

    def to_dict(self, mask_sensitive: bool = True) -> dict[str, Any]:
        """转换为字典

        Args:
            mask_sensitive: 是否脱敏敏感数据

        Returns:
            事件字典
        """
        data = self.data.copy() if mask_sensitive else self.data

        if mask_sensitive:
            data = _masker.mask_dict(data)

        return {
            "id": self.id,
            "event_type": self.event_type.value if self.event_type else None,
            "timestamp": self.timestamp.isoformat(),
            "user_id": sanitize_log_input(self.user_id or "") if self.user_id else None,
            "session_id": sanitize_log_input(self.session_id or "") if self.session_id else None,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "data": data,
            "metadata": self.metadata,
            "status": self.status,
            "error": self.error,
        }

    def to_json(self) -> str:
        """转换为 JSON 字符串

        Returns:
            JSON 字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ============== 审计日志记录器 ==============

class AuditLogger:
    """审计日志记录器

    提供异步的审计日志记录功能，支持多种存储后端。
    """

    def __init__(
        self,
        enable_console: bool = True,
        enable_file: bool = False,
        enable_db: bool = False,
        retention_days: int = 90,
    ):
        """初始化审计日志记录器

        Args:
            enable_console: 是否输出到控制台
            enable_file: 是否写入文件
            enable_db: 是否写入数据库
            retention_days: 日志保留天数
        """
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_db = enable_db
        self.retention_days = retention_days
        self._queue: asyncio.Queue[AuditEvent] = asyncio.Queue(maxsize=10000)
        self._worker_task: asyncio.Task | None = None
        self._is_running = False

        logger.info(
            "audit_logger_initialized",
            console=enable_console,
            file=enable_file,
            db=enable_db,
            retention_days=retention_days,
        )

    async def start(self) -> None:
        """启动审计日志后台处理

        开始异步处理队列中的审计事件。
        """
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._worker())

        logger.info("audit_logger_started")

    async def stop(self) -> None:
        """停止审计日志处理

        等待队列中的事件处理完毕。
        """
        if not self._is_running:
            return

        self._is_running = False

        # 等待队列处理完毕
        while not self._queue.empty():
            await asyncio.sleep(0.1)

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("audit_logger_stopped")

    async def record(self, event: AuditEvent) -> None:
        """记录审计事件

        将事件放入队列，由后台 worker 异步处理。

        Args:
            event: 审计事件
        """
        try:
            await asyncio.wait_for(self._queue.put(event), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("audit_queue_full")

        # 控制台输出实时日志
        if self.enable_console:
            logger.info(
                "audit_event",
                event_type=event.event_type.value if event.event_type else "unknown",
                user_id=event.user_id,
                session_id=event.session_id,
                agent_id=event.agent_id,
                status=event.status,
            )

    async def _worker(self) -> None:
        """后台 Worker

        从队列中获取事件并进行持久化处理。
        """
        logger.info("audit_worker_started")

        while self._is_running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._persist_event(event)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("audit_worker_error", error=str(e))

        logger.info("audit_worker_stopped")

    async def _persist_event(self, event: AuditEvent) -> None:
        """持久化事件

        Args:
            event: 审计事件
        """
        # 文件存储
        if self.enable_file:
            await self._persist_to_file(event)

        # 数据库存储
        if self.enable_db:
            await self._persist_to_db(event)

    async def _persist_to_file(self, event: AuditEvent) -> None:
        """持久化到文件

        Args:
            event: 审计事件
        """
        import aiofiles

        # 确保目录存在
        log_dir = "logs/audit"
        try:
            os.makedirs(log_dir, exist_ok=True)

            # 按日期分文件
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = f"{log_dir}/audit_{date_str}.logl"

            async with aiofiles.open(log_file, mode="a") as f:
                await f.write(event.to_json() + "\n")

        except Exception as e:
            logger.error("audit_file_write_failed", error=str(e))

    async def _persist_to_db(self, event: AuditEvent) -> None:
        """持久化到数据库

        Args:
            event: 审计事件
        """
        # TODO: 实现数据库持久化
        # 需要创建 audit_logs 表
        pass

    def is_running(self) -> bool:
        """检查是否正在运行

        Returns:
            是否运行中
        """
        return self._is_running


# ============== 全局审计日志记录器 =============

_global_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志记录器

    Returns:
        AuditLogger 实例
    """
    global _global_audit_logger

    if _global_audit_logger is None:
        # 从配置读取设置
        enable_audit = getattr(settings, "audit_enabled", False)
        enable_db = getattr(settings, "audit_db_enabled", False)
        enable_file = getattr(settings, "audit_file_enabled", False)
        retention_days = getattr(settings, "audit_retention_days", 90)

        if enable_audit:
            _global_audit_logger = AuditLogger(
                enable_console=True,
                enable_file=enable_file,
                enable_db=enable_db,
                retention_days=retention_days,
            )

    return _global_audit_logger


async def ensure_audit_logger_started() -> AuditLogger:
    """确保审计日志记录器已启动

    Returns:
        AuditLogger 实例
    """
    logger = get_audit_logger()

    if logger and not logger.is_running():
        await logger.start()

    return logger


# ============== 便捷函数 ==============

async def record_event(
    event_type: AuditEventType,
    data: dict[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "success",
    error: str | None = None,
) -> AuditEvent:
    """记录审计事件

    Args:
        event_type: 事件类型
        data: 事件数据
        user_id: 用户 ID
        session_id: 会话 ID
        agent_id: Agent ID
        trace_id: 追踪 ID
        metadata: 元数据
        status: 状态
        error: 错误信息

    Returns:
        AuditEvent 实例

    Examples:
        ```python
        # 记录工具调用
        await record_event(
            AuditEventType.TOOL_CALL_START,
            data={"tool": "search_web", "query": "..."},
            user_id="user-123",
        )

        # 记录错误
        await record_event(
            AuditEventType.ERROR,
            data={"error": "API timeout"},
            status="failed",
        )
        ```
    """
    event = AuditEvent(
        event_type=event_type,
        data=data or {},
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        trace_id=trace_id,
        metadata=metadata or {},
        status=status,
        error=error,
    )

    logger = await ensure_audit_logger_started()
    await logger.record(event)

    return event


async def record_agent_event(
    event_type: AuditEventType,
    agent_id: str,
    data: dict[str, Any],
    **kwargs,
) -> AuditEvent:
    """记录 Agent 事件（便捷函数）

    Args:
        event_type: 事件类型
        agent_id: Agent ID
        data: 事件数据
        **kwargs: 其他参数

    Returns:
        AuditEvent 实例

    Examples:
        ```python
        await record_agent_event(
            AuditEventType.AGENT_STARTED,
            agent_id="chat-agent",
            data={"model": "gpt-4o"},
        )
        ```
    """
    return await record_event(
        event_type=event_type,
        data=data,
        agent_id=agent_id,
        **kwargs,
    )


async def record_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any | None = None,
    status: str = "success",
    **kwargs,
) -> AuditEvent:
    """记录工具调用事件

    Args:
        tool_name: 工具名称
        tool_input: 工具输入
        tool_output: 工具输出
        status: 状态
        **kwargs: 其他参数

    Returns:
        AuditEvent 实例

    Examples:
        ```python
        await record_tool_call(
            "search_web",
            {"query": "Python教程"},
            tool_output="找到...",
        )
        ```
    """
    event_type = (
        AuditEventType.TOOL_CALL_SUCCESS
        if status == "success"
        else AuditEventType.TOOL_CALL_FAILED
    )

    data = {
        "tool_name": tool_name,
        "tool_input": tool_input,
    }

    if tool_output is not None:
        # 限制输出大小
        output_str = str(tool_output)
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "... (truncated)"
        data["tool_output"] = output_str

    return await record_event(event_type, data=data, status=status, **kwargs)


async def get_audit_logs(
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    event_type: AuditEventType | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    """查询审计日志

    Args:
        user_id: 用户 ID
        session_id: 会话 ID
        agent_id: Agent ID
        event_type: 事件类型
        start_time: 开始时间
        end_time: 结束时间
        limit: 返回数量限制

    Returns:
        审计日志列表

    Examples:
        ```python
        # 查询用户的所有审计日志
        logs = await get_audit_logs(user_id="user-123")

        # 查询工具调用日志
        logs = await get_audit_logs(
            user_id="user-123",
            event_type=AuditEventType.TOOL_CALL_START,
        )
        ```
    """
    # TODO: 从数据库查询
    # 当前返回空列表
    logger.debug("get_audit_logs_called", user_id=user_id, limit=limit)
    return []


# 装饰器
def audit_event(
    event_type: AuditEventType | str,
    data: dict[str, Any] | None = None,
):
    """审计事件装饰器

    自动记录函数调用的审计日志。

    Args:
        event_type: 事件类型
        data: 事件数据

    Examples:
        ```python
        @audit_event(AuditEventType.AGENT_STARTED)
        async def my_agent_function(user_id: str):
            # 函数执行完成会自动记录审计日志
            return "result"
        ```
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)

                await record_event(
                    event_type=event_type if isinstance(event_type, str) else event_type,
                    data=data or {},
                    status="success",
                )

                return result

            except Exception as e:
                await record_event(
                    event_type=AuditEventType.ERROR,
                    data={"error": str(e)},
                    status="failed",
                )
                raise

        return wrapper
    return decorator


__all__ = [
    # 数据类
    "AuditEvent",
    "AuditEventType",
    "SensitiveDataMasker",
    # 记录器
    "AuditLogger",
    "get_audit_logger",
    "ensure_audit_logger_started",
    # 便捷函数
    "record_event",
    "record_agent_event",
    "record_tool_call",
    "get_audit_logs",
    # 装饰器
    "audit_event",
]
