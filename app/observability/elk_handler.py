"""ELK 日志处理器

基于 Logstash TCP Handler 的生产级实现，支持：
- 自动重连机制
- 线程安全
- 结构化 JSON 日志
- 重试与容错
- 批量发送优化

参考: ai-engineer-training2/week08/p41elk.py
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.config.settings import get_settings
from app.observability.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ELKHandler(logging.Handler):
    """ELK Logstash TCP 日志处理器

    特性：
    - 自动重连：连接断开时自动尝试重新连接
    - 线程安全：使用 threading.Lock 保护共享状态
    - 批量发送：积累多条日志后批量发送，减少网络开销
    - 结构化日志：JSON 格式，便于 Kibana 分析
    - 降级策略：ELK 不可用时自动降级到本地文件

    使用示例：
        ```python
        from app.observability.elk_handler import ELKHandler

        handler = ELKHandler(
            host="localhost",
            port=5044,
            batch_size=10,
            batch_timeout=5.0
        )
        logger.addHandler(handler)
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5044,
        timeout: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 10,
        batch_timeout: float = 5.0,
        enable_fallback: bool = True,
        fallback_path: Optional[Path] = None,
    ):
        """初始化 ELK 处理器

        Args:
            host: Logstash 服务器地址
            port: Logstash 监听端口
            timeout: 连接超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
            batch_size: 批量发送的日志条数
            batch_timeout: 批量发送超时时间（秒）
            enable_fallback: 是否启用降级到本地文件
            fallback_path: 降级文件路径
        """
        super().__init__()

        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.enable_fallback = enable_fallback
        self.fallback_path = fallback_path or Path("logs/elk_fallback.log")

        # 连接状态
        self._socket: Optional[socket.socket] = None
        self._state = ConnectionState.DISCONNECTED
        self._lock = threading.RLock()
        self._last_attempt = 0.0
        self._consecutive_errors = 0

        # 批量发送缓冲区
        self._batch: list[str] = []
        self._batch_lock = threading.Lock()
        self._last_flush = time.time()

        # 确保降级目录存在
        if self.enable_fallback:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> bool:
        """建立到 Logstash 的 TCP 连接

        Returns:
            bool: 连接是否成功
        """
        with self._lock:
            # 避免频繁重连
            now = time.time()
            if now - self._last_attempt < self.retry_delay:
                return False
            self._last_attempt = now

            self._state = ConnectionState.CONNECTING

            try:
                # 关闭旧连接
                if self._socket:
                    try:
                        self._socket.close()
                    except Exception:
                        pass

                # 创建新连接
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.host, self.port))

                self._state = ConnectionState.CONNECTED
                self._consecutive_errors = 0

                logger.debug(
                    "elk_connected",
                    host=self.host,
                    port=self.port,
                )
                return True

            except socket.timeout:
                self._state = ConnectionState.ERROR
                logger.warning(
                    "elk_connect_timeout",
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout,
                )
            except ConnectionRefusedError:
                self._state = ConnectionState.ERROR
                logger.warning(
                    "elk_connection_refused",
                    host=self.host,
                    port=self.port,
                )
            except OSError as e:
                self._state = ConnectionState.ERROR
                logger.warning(
                    "elk_connect_failed",
                    host=self.host,
                    port=self.port,
                    error=str(e),
                )

            # 连接失败清理
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None

            return False

    def _send_with_retry(self, data: str) -> bool:
        """带重试机制的数据发送

        Args:
            data: 要发送的日志数据

        Returns:
            bool: 发送是否成功
        """
        for attempt in range(self.max_retries):
            try:
                # 检查连接
                if not self._is_connected():
                    if not self._connect():
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            return False

                # 发送数据（每行一条日志）
                message = data + "\n"
                self._socket.sendall(message.encode("utf-8"))

                self._consecutive_errors = 0
                return True

            except socket.timeout:
                logger.debug("elk_send_timeout", attempt=attempt + 1)
            except (ConnectionResetError, BrokenPipeError):
                logger.debug("elk_connection_lost", attempt=attempt + 1)
                self._state = ConnectionState.DISCONNECTED
            except Exception as e:
                logger.debug(
                    "elk_send_failed",
                    attempt=attempt + 1,
                    error=str(e),
                )

            # 发送失败，清理连接
            with self._lock:
                if self._socket:
                    try:
                        self._socket.close()
                    except Exception:
                        pass
                    self._socket = None
                self._state = ConnectionState.DISCONNECTED

            # 最后一次尝试前不再等待
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return False

    def _is_connected(self) -> bool:
        """检查连接是否有效

        Returns:
            bool: 连接是否有效
        """
        if not self._socket:
            return False

        try:
            # 非阻塞检查连接状态
            self._socket.setblocking(False)
            self._socket.send(b"")
            self._socket.setblocking(True)
            return True
        except BlockingIOError:
            self._socket.setblocking(True)
            return True
        except Exception:
            self._socket.setblocking(True)
            return False

    def _fallback_write(self, log_entry: dict[str, Any]) -> None:
        """降级写入本地文件

        Args:
            log_entry: 日志条目
        """
        if not self.enable_fallback:
            return

        try:
            with open(self.fallback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(
                "elk_fallback_failed",
                path=str(self.fallback_path),
                error=str(e),
            )

    def _format_log_entry(self, record: logging.LogRecord) -> dict[str, Any]:
        """格式化日志记录为结构化 JSON

        Args:
            record: 日志记录对象

        Returns:
            dict: 结构化日志条目
        """
        # 基础字段
        log_entry: dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "thread": record.thread,
            "process": record.process,
        }

        # 代码位置
        if hasattr(record, "pathname"):
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
                "module": record.module,
            }

        # 异常信息
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.format(record) if record.exc_text else None,
            }

        # 结构化上下文（由 structlog 添加）
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # 额外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return log_entry

    def emit(self, record: logging.LogRecord) -> None:
        """发送日志记录到 Logstash

        Args:
            record: 日志记录对象
        """
        try:
            # 格式化日志
            log_entry = self._format_log_entry(record)
            json_data = json.dumps(log_entry, ensure_ascii=False)

            # 尝试发送
            success = self._send_with_retry(json_data)

            if not success:
                self._consecutive_errors += 1

                # 降级到本地文件
                if self.enable_fallback:
                    self._fallback_write(log_entry)

                # 错误过多时降低日志级别
                if self._consecutive_errors > 100:
                    self._consecutive_errors = 0
                    logger.warning(
                        "elk_too_many_errors",
                        consecutive_errors=self._consecutive_errors,
                    )

        except Exception as e:
            # 不应该到达这里，但如果发生则记录错误
            logger.error(
                "elk_handler_error",
                error=str(e),
                exc_info=True,
            )

    def flush(self) -> None:
        """刷新缓冲区"""
        with self._batch_lock:
            if self._batch:
                # 批量发送
                batch_data = "\n".join(self._batch)
                self._send_with_retry(batch_data)
                self._batch.clear()
            self._last_flush = time.time()

    def close(self) -> None:
        """关闭连接"""
        self.flush()

        with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            self._state = ConnectionState.DISCONNECTED

        super().close()


class BatchELKHandler(ELKHandler):
    """批量发送的 ELK 处理器

    积累多条日志后批量发送，减少网络开销。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """初始化批量处理器"""
        super().__init__(*args, **kwargs)
        self._batch_timer: Optional[threading.Timer] = None
        self._start_batch_timer()

    def _start_batch_timer(self) -> None:
        """启动批量发送定时器"""
        if self._batch_timer:
            self._batch_timer.cancel()

        self._batch_timer = threading.Timer(
            self.batch_timeout,
            self._flush_batch,
        )
        self._batch_timer.daemon = True
        self._batch_timer.start()

    def _flush_batch(self) -> None:
        """刷新批量缓冲区"""
        with self._batch_lock:
            if self._batch:
                batch_data = "\n".join(self._batch)
                self._send_with_retry(batch_data)
                self._batch.clear()
            self._last_flush = time.time()

        # 重启定时器
        self._start_batch_timer()

    def emit(self, record: logging.LogRecord) -> None:
        """发送日志记录（批量模式）"""
        try:
            log_entry = self._format_log_entry(record)
            json_data = json.dumps(log_entry, ensure_ascii=False)

            with self._batch_lock:
                self._batch.append(json_data)

                # 达到批量大小或超时则发送
                if len(self._batch) >= self.batch_size:
                    batch_data = "\n".join(self._batch)
                    success = self._send_with_retry(batch_data)

                    if success:
                        self._batch.clear()
                    else:
                        # 发送失败，降级处理
                        for entry in self._batch:
                            log_entry = json.loads(entry)
                            self._fallback_write(log_entry)
                        self._batch.clear()

                self._last_flush = time.time()

        except Exception as e:
            logger.error(
                "batch_elk_handler_error",
                error=str(e),
                exc_info=True,
            )

    def close(self) -> None:
        """关闭处理器"""
        if self._batch_timer:
            self._batch_timer.cancel()
        self._flush_batch()
        super().close()


# ============== 全局初始化 ==============

_elk_handler: Optional[ELKHandler] = None
_batch_elk_handler: Optional[BatchELKHandler] = None


def get_elk_handler() -> Optional[ELKHandler]:
    """获取 ELK 处理器单例

    Returns:
        ELKHandler 或 None（如果未启用）
    """
    global _elk_handler

    if _elk_handler is None and settings.elk_enabled:
        _elk_handler = ELKHandler(
            host=settings.elk_host,
            port=settings.elk_port,
            timeout=settings.elk_timeout,
            max_retries=settings.elk_max_retries,
            enable_fallback=settings.elk_fallback_enabled,
        )

    return _elk_handler if settings.elk_enabled else None


def get_batch_elk_handler() -> Optional[BatchELKHandler]:
    """获取批量 ELK 处理器单例

    Returns:
        BatchELKHandler 或 None（如果未启用）
    """
    global _batch_elk_handler

    if _batch_elk_handler is None and settings.elk_enabled:
        _batch_elk_handler = BatchELKHandler(
            host=settings.elk_host,
            port=settings.elk_port,
            timeout=settings.elk_timeout,
            max_retries=settings.elk_max_retries,
            batch_size=settings.elk_batch_size,
            batch_timeout=settings.elk_batch_timeout,
            enable_fallback=settings.elk_fallback_enabled,
        )

    return _batch_elk_handler if settings.elk_enabled else None


def setup_elk_logging(root_logger: logging.Logger) -> None:
    """设置 ELK 日志

    Args:
        root_logger: 根日志记录器
    """
    handler = get_elk_handler()
    if handler:
        root_logger.addHandler(handler)
        logger.info(
            "elk_handler_registered",
            host=settings.elk_host,
            port=settings.elk_port,
        )
