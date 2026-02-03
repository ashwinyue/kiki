"""任务管理 API

对齐 WeKnora99 的任务管理功能，提供任务的 CRUD 操作。
使用 FastAPI 标准依赖注入模式。

API 端点:
    - POST /tasks - 创建任务
    - GET /tasks - 获取任务列表
    - GET /tasks/{task_id} - 获取任务详情
    - PUT /tasks/{task_id} - 更新任务
    - DELETE /tasks/{task_id} - 取消任务
    - GET /tasks/{task_id}/logs - 获取任务日志
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status

from app.middleware import TenantIdDep, UserIdDep
from app.models.task import TaskCreate, TaskList, TaskPublic, TaskUpdate
from app.observability.logging import get_logger
from app.schemas.response import Response
from app.tasks import (
    TaskPriority,
    TaskStatus,
    TaskType,
    generate_task_id,
    revoke_task,
    send_task,
)
from app.tasks.store import TaskStore

logger = get_logger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])


# ============== 工具函数 ==============


async def verify_task_access(
    task_id: str,
    tenant_id: int,
) -> bool:
    """验证任务访问权限

    Args:
        task_id: 任务 ID
        tenant_id: 租户 ID

    Returns:
        是否有权限
    """
    store = TaskStore()
    task = await store.get_task(task_id, tenant_id)
    return task is not None


# ============== API 端点 ==============


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_task(
    task_data: TaskCreate,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> Response[TaskPublic]:
    """创建异步任务

    创建一个新的异步任务并发送到 Celery 队列。

    Args:
        task_data: 任务创建数据
        tenant_id: 租户 ID
        user_id: 用户 ID

    Returns:
        创建的任务信息
    """
    # 验证任务类型
    try:
        task_type = TaskType(task_data.task_type)
    except ValueError:
        return Response.error(
            code="invalid_task_type",
            message=f"无效的任务类型: {task_data.task_type}",
        )

    store = TaskStore()
    task = await store.create_task(
        task_id=task_data.task_id,
        task_type=task_data.task_type,
        tenant_id=tenant_id,
        priority=task_data.priority,
        status=TaskStatus.PENDING,
        title=task_data.title,
        description=task_data.description,
        payload=task_data.payload,
        business_id=task_data.business_id,
        business_type=task_data.business_type,
        parent_task_id=task_data.parent_task_id,
        max_retries=task_data.max_retries,
        total_items=task_data.total_items,
        created_by=user_id,
    )

    # 发送任务到 Celery
    try:
        celery_task_id = send_task(
            task_type=task_data.task_type,
            payload=task_data.payload or {},
            tenant_id=tenant_id,
            priority=task_data.priority.value,
        )

        # 更新 Celery 任务 ID
        task = await store.update_task(
            task_data.task_id,
            tenant_id,
            celery_task_id=celery_task_id,
        )

        logger.info(
            "task_created_and_sent",
            task_id=task.task_id,
            task_type=task_data.task_type,
            celery_task_id=celery_task_id,
        )

    except Exception as e:
        logger.error("send_task_failed", task_id=task.task_id, error=str(e))
        # 发送失败不影响任务创建，只记录错误

    # 转换为公开模型
    return Response.success(
        data=_task_to_public(task),
        message="任务创建成功",
    )


@router.get("")
async def list_tasks(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    task_type: str | None = Query(None, description="任务类型"),
    status: str | None = Query(None, description="任务状态"),
    business_id: str | None = Query(None, description="业务 ID"),
) -> Response[TaskList]:
    """获取任务列表

    Args:
        tenant_id: 租户 ID
        page: 页码
        size: 每页数量
        task_type: 任务类型筛选
        status: 状态筛选
        business_id: 业务 ID 筛选

    Returns:
        任务列表
    """
    store = TaskStore()
    tasks, total = await store.list_tasks(
        tenant_id,
        page=page,
        size=size,
        task_type=task_type,
        status=status,
        business_id=business_id,
    )
    items = [_task_to_public(task) for task in tasks]

    task_list = TaskList(
        items=items,
        total=total,
        page=page,
        size=size,
    )

    return Response.success(data=task_list)


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    tenant_id: TenantIdDep,
) -> Response[TaskPublic]:
    """获取任务详情

    Args:
        task_id: 任务 ID
        tenant_id: 租户 ID

    Returns:
        任务详情
    """
    store = TaskStore()
    task = await store.get_task(task_id, tenant_id)

    if not task:
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )

    return Response.success(data=_task_to_public(task))


@router.put("/{task_id}")
async def update_task(
    task_id: str,
    task_update: TaskUpdate,
    tenant_id: TenantIdDep,
) -> Response[TaskPublic]:
    """更新任务

    仅支持更新状态和进度信息，通常由任务处理器内部调用。

    Args:
        task_id: 任务 ID
        task_update: 更新数据
        tenant_id: 租户 ID

    Returns:
        更新后的任务信息
    """
    store = TaskStore()
    task = await store.get_task(task_id, tenant_id)

    if not task:
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )

    fields: dict[str, Any] = {}
    if task_update.status is not None:
        fields["status"] = task_update.status
    if task_update.progress is not None:
        fields["progress"] = task_update.progress
    if task_update.current_step is not None:
        fields["current_step"] = task_update.current_step
    if task_update.processed_items is not None:
        fields["processed_items"] = task_update.processed_items
    if task_update.failed_items is not None:
        fields["failed_items"] = task_update.failed_items
    if task_update.result is not None:
        fields["result"] = task_update.result
    if task_update.error_message is not None:
        fields["error_message"] = task_update.error_message
    if task_update.error_stack is not None:
        fields["error_stack"] = task_update.error_stack
    if task_update.celery_task_id is not None:
        fields["celery_task_id"] = task_update.celery_task_id
    if task_update.retry_count is not None:
        fields["retry_count"] = task_update.retry_count
    if task_update.extra_metadata is not None:
        fields["extra_metadata"] = task_update.extra_metadata

    task = await store.update_task(task_id, tenant_id, **fields)

    return Response.success(data=_task_to_public(task))


@router.delete("/{task_id}")
async def cancel_task_endpoint(
    task_id: str,
    tenant_id: TenantIdDep,
) -> Response[dict]:
    """取消任务

    Args:
        task_id: 任务 ID
        tenant_id: 租户 ID

    Returns:
        操作结果
    """
    store = TaskStore()
    task = await store.get_task(task_id, tenant_id)

    if not task:
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )

    # 检查任务状态
    if task.get("status") in ("completed", "failed", "cancelled"):
        return Response.error(
            code="task_not_cancellable",
            message=f"任务已完成或已取消，当前状态: {task.get('status')}",
        )

    # 撤销 Celery 任务
    if task.get("celery_task_id"):
        try:
            revoke_task(task.get("celery_task_id"), terminate=False)
        except Exception as e:
            logger.error("revoke_task_failed", task_id=task_id, error=str(e))

    # 更新状态
    await store.update_task(task_id, tenant_id, status=TaskStatus.CANCELLED)

    return Response.success(
        data={"task_id": task_id, "status": "cancelled"},
        message="任务已取消",
    )


@router.get("/{task_id}/logs")
async def get_task_logs(
    task_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    level: str | None = Query(None, description="日志级别"),
) -> Response[dict]:
    """获取任务日志

    Args:
        task_id: 任务 ID
        tenant_id: 租户 ID
        page: 页码
        size: 每页数量
        level: 日志级别筛选

    Returns:
        任务日志列表
    """
    store = TaskStore()
    task = await store.get_task(task_id, tenant_id)
    if not task:
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )
    items = await store.list_logs(
        task_id,
        tenant_id,
        page=page,
        size=size,
        level=level,
    )

    return Response.success(
        data={
            "items": items,
            "page": page,
            "size": size,
        }
    )


# ============== 便捷端点 ==============


@router.post("/enqueue/{task_type}")
async def enqueue_task(
    task_type: str,
    payload: dict[str, Any],
    tenant_id: TenantIdDep,
    user_id: str = Depends(get_current_user_id),
    business_id: str | None = None,
    priority: str = Query("default", description="优先级: critical, default, low"),
) -> Response[TaskPublic]:
    """快速创建并发送任务

    便捷端点，自动生成任务 ID 并创建记录。

    Args:
        task_type: 任务类型
        payload: 任务参数
        tenant_id: 租户 ID
        user_id: 用户 ID
        business_id: 业务 ID
        priority: 优先级

    Returns:
        创建的任务信息
    """
    # 验证任务类型
    try:
        task_type_enum = TaskType(task_type)
    except ValueError:
        return Response.error(
            code="invalid_task_type",
            message=f"无效的任务类型: {task_type}",
        )

    # 生成任务 ID
    task_id = generate_task_id(
        task_type=task_type,
        tenant_id=tenant_id,
        business_id=business_id,
    )

    store = TaskStore()
    task = await store.create_task(
        task_id=task_id,
        task_type=task_type,
        tenant_id=tenant_id,
        priority=priority,
        status=TaskStatus.PENDING,
        payload=payload,
        business_id=business_id,
        created_by=user_id,
    )

    # 发送任务到 Celery
    try:
        celery_task_id = send_task(
            task_type=task_type,
            payload=payload,
            tenant_id=tenant_id,
            priority=priority,
        )
        task = await store.update_task(task_id, tenant_id, celery_task_id=celery_task_id)
    except Exception as e:
        logger.error("send_task_failed", task_id=task_id, error=str(e))

    return Response.success(
        data=_task_to_public(task),
        message="任务已加入队列",
    )


# ============== 工具函数 ==============


def _task_to_public(task: dict[str, Any] | None) -> TaskPublic:
    """将 Task 模型转换为 TaskPublic

    Args:
        task: Task 记录

    Returns:
        TaskPublic 实例
    """
    if task is None:
        raise ValueError("task is None")
    return TaskPublic(
        task_id=task.get("task_id", ""),
        task_type=task.get("task_type", ""),
        tenant_id=task.get("tenant_id", 0),
        priority=TaskPriority(task.get("priority")),
        status=TaskStatus(task.get("status")),
        title=task.get("title"),
        description=task.get("description"),
        progress=task.get("progress", 0),
        current_step=task.get("current_step"),
        total_items=task.get("total_items"),
        processed_items=task.get("processed_items"),
        failed_items=task.get("failed_items"),
        result=task.get("result"),
        error_message=task.get("error_message"),
        retry_count=task.get("retry_count", 0),
        max_retries=task.get("max_retries", 3),
        business_id=task.get("business_id"),
        business_type=task.get("business_type"),
        parent_task_id=task.get("parent_task_id"),
        duration=task.get("duration"),
        created_at=task.get("created_at"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
    )


__all__ = ["router"]
