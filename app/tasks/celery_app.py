"""Celery 应用配置

对齐 WeKnora99 的 asynq 任务队列配置。

配置说明:
    - 使用 Redis 作为 broker
    - 使用 Redis 作为结果存储
    - 支持三个优先级队列: critical, default, low
    - 支持任务重试和超时控制

使用方式:
    from app.tasks.celery_app import celery_app, get_celery_app

    # 获取 Celery 实例
    celery = get_celery_app()

    # 定义任务
    @celery.task(name="document:process")
    def process_document(payload: dict) -> dict:
        # 处理逻辑
        return {"status": "completed"}

    # 发送任务
    result = process_document.delay(payload)
"""

import os
from typing import Any

from celery import Celery, Task
from celery.schedules import crontab

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()


# ============== Celery 配置 ==============


class CeleryConfig:
    """Celery 配置类"""

    # Broker 配置 (Redis)
    broker_url: str = settings.redis_url
    broker_connection_retry_on_startup: bool = True
    broker_pool_limit: int = 10

    # 结果存储 (Redis)
    result_backend: str = settings.redis_url
    result_extended: bool = True
    result_expires: int = 86400  # 24 小时

    # 任务序列化
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = ["json"]
    timezone: str = "Asia/Shanghai"
    enable_utc: bool = True

    # 任务路由 (按优先级)
    task_routes: dict[str, dict] = {
        "chunk:extract": {"queue": "default"},
        "document:process": {"queue": "default"},
        "faq:import": {"queue": "default"},
        "question:generation": {"queue": "low"},
        "summary:generation": {"queue": "low"},
        "kb:clone": {"queue": "critical"},
        "index:delete": {"queue": "critical"},
        "kb:delete": {"queue": "critical"},
        "knowledge:list_delete": {"queue": "default"},
        "datatable:summary": {"queue": "low"},
    }

    # Worker 配置
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 1000

    # 任务执行配置
    task_acks_late: bool = True  # 任务执行后才确认
    task_reject_on_worker_lost: bool = True  # Worker 丢失时拒绝任务
    task_time_limit: int = 3600  # 1 小时硬限制
    task_soft_time_limit: int = 3300  # 55 分钟软限制

    # 重试配置
    task_default_max_retries: int = 3
    task_default_retry_delay: int = 60  # 秒
    task_retry_backoff: bool = True
    task_retry_backoff_max: int = 600  # 10 分钟

    # 任务跟踪
    task_send_sent_event: bool = True
    task_track_started: bool = True

    # 安全配置
    worker_send_task_events: bool = True
    task_send_event_expires: int = 86400

    # 优化配置
    broker_transport_options: dict[str, Any] = {
        "visibility_timeout": 3600,
        "max_connections": 50,
    }

    # 定时任务
    beat_schedule: dict[str, dict] = {
        "cleanup-completed-tasks": {
            "task": "tasks.cleanup",
            "schedule": crontab(hour=2, minute=0),  # 每天凌晨 2 点执行
            "args": (7,),  # 清理 7 天前的任务
        },
    }


# ============== Celery 应用工厂 ==============


def create_celery_app(config: CeleryConfig | None = None) -> Celery:
    """创建 Celery 应用

    Args:
        config: Celery 配置

    Returns:
        Celery 应用实例
    """
    config = config or CeleryConfig()

    # 创建 Celery 应用
    celery_app = Celery(
        "kiki_tasks",
        broker=config.broker_url,
        backend=config.result_backend,
    )

    # 应用配置
    celery_app.config_from_object(config)

    # 自动发现任务
    celery_app.autodiscover_tasks(["app.tasks.handlers"])

    logger.info(
        "celery_app_created",
        broker=config.broker_url,
        backend=config.result_backend,
    )

    return celery_app


# ============== 全局 Celery 实例 ==============

_celery_app: Celery | None = None


def get_celery_app() -> Celery:
    """获取 Celery 应用单例

    Returns:
        Celery 应用实例
    """
    global _celery_app
    if _celery_app is None:
        _celery_app = create_celery_app()
    return _celery_app


# ============== 基础任务类 ==============


class DatabaseTask(Task):
    """任务基类

    保留类名以兼容现有任务定义，任务状态由 Redis 存储。
    """

    def on_success(self, retval, task_id, args, kwargs):
        """任务成功回调"""
        logger.info(
            "task_completed",
            task_id=task_id,
            task_name=self.name,
            result=str(retval)[:200],
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败回调"""
        logger.error(
            "task_failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            exc_info=einfo,
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """任务重试回调"""
        logger.warning(
            "task_retrying",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            retry_count=self.request.retries,
        )


# ============== Celery 实例 ==============

# 导出 Celery 实例
celery_app = get_celery_app()

# 导出任务装饰器
task = celery_app.task


# ============== 工具函数 ==============


def send_task(
    task_type: str,
    payload: dict[str, Any],
    tenant_id: int,
    priority: str = "default",
    queue: str | None = None,
    max_retries: int = 3,
    countdown: int | None = None,
) -> str:
    """发送任务到 Celery 队列

    Args:
        task_type: 任务类型
        payload: 任务参数
        tenant_id: 租户 ID
        priority: 优先级 (critical, default, low)
        queue: 队列名称 (可选，默认根据优先级选择)
        max_retries: 最大重试次数
        countdown: 延迟执行秒数

    Returns:
        Celery 任务 ID
    """
    # 确定队列
    if queue is None:
        queue = {
            "critical": "critical",
            "default": "default",
            "low": "low",
        }.get(priority, "default")

    # 发送任务
    result = celery_app.send_task(
        task_type,
        args=[payload],
        kwargs={"tenant_id": tenant_id},
        queue=queue,
        max_retries=max_retries,
        countdown=countdown,
    )

    logger.info(
        "task_sent",
        task_type=task_type,
        task_id=result.id,
        queue=queue,
        tenant_id=tenant_id,
    )

    return result.id


def revoke_task(celery_task_id: str, terminate: bool = False) -> None:
    """撤销任务

    Args:
        celery_task_id: Celery 任务 ID
        terminate: 是否强制终止正在执行的任务
    """
    celery_app.control.revoke(celery_task_id, terminate=terminate)

    logger.info(
        "task_revoked",
        task_id=celery_task_id,
        terminate=terminate,
    )


def get_task_status(celery_task_id: str) -> str:
    """获取任务状态

    Args:
        celery_task_id: Celery 任务 ID

    Returns:
        任务状态 (PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED)
    """
    result = celery_app.AsyncResult(celery_task_id)
    return result.state


# ============== 导出 ==============

__all__ = [
    # Celery 应用
    "celery_app",
    "get_celery_app",
    "create_celery_app",
    # 配置
    "CeleryConfig",
    # 任务基类
    "DatabaseTask",
    # 装饰器
    "task",
    # 工具函数
    "send_task",
    "revoke_task",
    "get_task_status",
]
