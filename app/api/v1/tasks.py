"""任务管理 API

对齐 WeKnora99 的任务管理功能，提供任务的 CRUD 操作。

API 端点:
    - POST /tasks - 创建任务
    - GET /tasks - 获取任务列表
    - GET /tasks/{task_id} - 获取任务详情
    - PUT /tasks/{task_id} - 更新任务
    - DELETE /tasks/{task_id} - 取消任务
    - GET /tasks/{task_id}/logs - 获取任务日志
"""

from typing import Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_current_user_id,
    get_current_tenant_id,
    get_session,
)
from app.models.database import TaskCreate, TaskPublic, TaskList, TaskUpdate
from app.observability.logging import get_logger
from app.schemas.response import Response
from app.tasks import (
    ParsedTaskID,
    TaskStatus,
    TaskType,
    cancel_task,
    generate_task_id,
    get_celery_app,
    revoke_task,
    send_task,
    validate_task_id,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])


# ============== 工具函数 ==============


async def verify_task_access(
    task_id: str,
    tenant_id: int,
    session: AsyncSession,
) -> bool:
    """验证任务访问权限

    Args:
        task_id: 任务 ID
        tenant_id: 租户 ID
        session: 数据库会话

    Returns:
        是否有权限
    """
    from sqlalchemy import select

    from app.models.database import Task

    stmt = select(Task).where(
        Task.task_id == task_id,
        Task.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()
    return task is not None


# ============== API 端点 ==============


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_task(
    task_data: TaskCreate,
    session: AsyncSession = Depends(get_session),
    tenant_id: int = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
) -> Response[TaskPublic]:
    """创建异步任务

    创建一个新的异步任务并发送到 Celery 队列。

    Args:
        task_data: 任务创建数据
        session: 数据库会话
        tenant_id: 租户 ID
        user_id: 用户 ID

    Returns:
        创建的任务信息
    """
    from app.models.database import Task

    # 验证任务类型
    try:
        task_type = TaskType(task_data.task_type)
    except ValueError:
        return Response.error(
            code="invalid_task_type",
            message=f"无效的任务类型: {task_data.task_type}",
        )

    # 创建任务记录
    task = Task(
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

    session.add(task)
    await session.commit()
    await session.refresh(task)

    # 发送任务到 Celery
    try:
        celery_task_id = send_task(
            task_type=task_data.task_type,
            payload=task_data.payload or {},
            tenant_id=tenant_id,
            priority=task_data.priority.value,
        )

        # 更新 Celery 任务 ID
        task.celery_task_id = celery_task_id
        await session.commit()

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
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    task_type: str | None = Query(None, description="任务类型"),
    status: str | None = Query(None, description="任务状态"),
    business_id: str | None = Query(None, description="业务 ID"),
    session: AsyncSession = Depends(get_session),
    tenant_id: int = Depends(get_current_tenant_id),
) -> Response[TaskList]:
    """获取任务列表

    Args:
        page: 页码
        size: 每页数量
        task_type: 任务类型筛选
        status: 状态筛选
        business_id: 业务 ID 筛选
        session: 数据库会话
        tenant_id: 租户 ID

    Returns:
        任务列表
    """
    from sqlalchemy import select

    from app.models.database import Task

    # 构建查询
    stmt = select(Task).where(Task.tenant_id == tenant_id)

    # 应用筛选条件
    if task_type:
        stmt = stmt.where(Task.task_type == task_type)
    if status:
        stmt = stmt.where(Task.status == status)
    if business_id:
        stmt = stmt.where(Task.business_id == business_id)

    # 按创建时间倒序
    stmt = stmt.order_by(Task.created_at.desc())

    # 分页
    offset = (page - 1) * size
    stmt = stmt.offset(offset).limit(size)

    # 执行查询
    result = await session.execute(stmt)
    tasks = result.scalars().all()

    # 获取总数
    count_stmt = select(Task).where(Task.tenant_id == tenant_id)
    if task_type:
        count_stmt = count_stmt.where(Task.task_type == task_type)
    if status:
        count_stmt = count_stmt.where(Task.status == status)
    if business_id:
        count_stmt = count_stmt.where(Task.business_id == business_id)

    count_result = await session.execute(count_stmt)
    total = len(count_result.scalars().all())

    # 转换为公开模型列表
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
    session: AsyncSession = Depends(get_session),
    tenant_id: int = Depends(get_current_tenant_id),
) -> Response[TaskPublic]:
    """获取任务详情

    Args:
        task_id: 任务 ID
        session: 数据库会话
        tenant_id: 租户 ID

    Returns:
        任务详情
    """
    from sqlalchemy import select

    from app.models.database import Task

    stmt = select(Task).where(
        Task.task_id == task_id,
        Task.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()

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
    session: AsyncSession = Depends(get_session),
    tenant_id: int = Depends(get_current_tenant_id),
) -> Response[TaskPublic]:
    """更新任务

    仅支持更新状态和进度信息，通常由任务处理器内部调用。

    Args:
        task_id: 任务 ID
        task_update: 更新数据
        session: 数据库会话
        tenant_id: 租户 ID

    Returns:
        更新后的任务信息
    """
    from sqlalchemy import select

    from app.models.database import Task

    stmt = select(Task).where(
        Task.task_id == task_id,
        Task.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()

    if not task:
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )

    # 更新字段
    if task_update.status is not None:
        task.status = task_update.status
    if task_update.progress is not None:
        task.progress = task_update.progress
    if task_update.current_step is not None:
        task.current_step = task_update.current_step
    if task_update.processed_items is not None:
        task.processed_items = task_update.processed_items
    if task_update.failed_items is not None:
        task.failed_items = task_update.failed_items
    if task_update.result is not None:
        task.result = task_update.result
    if task_update.error_message is not None:
        task.error_message = task_update.error_message
    if task_update.error_stack is not None:
        task.error_stack = task_update.error_stack
    if task_update.celery_task_id is not None:
        task.celery_task_id = task_update.celery_task_id
    if task_update.retry_count is not None:
        task.retry_count = task_update.retry_count
    if task_update.extra_metadata is not None:
        task.extra_metadata = task_update.extra_metadata

    await session.commit()
    await session.refresh(task)

    return Response.success(data=_task_to_public(task))


@router.delete("/{task_id}")
async def cancel_task_endpoint(
    task_id: str,
    session: AsyncSession = Depends(get_session),
    tenant_id: int = Depends(get_current_tenant_id),
) -> Response[dict]:
    """取消任务

    Args:
        task_id: 任务 ID
        session: 数据库会话
        tenant_id: 租户 ID

    Returns:
        操作结果
    """
    from sqlalchemy import select

    from app.models.database import Task

    stmt = select(Task).where(
        Task.task_id == task_id,
        Task.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    task = result.scalar_one_or_none()

    if not task:
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )

    # 检查任务状态
    if task.status in ("completed", "failed", "cancelled"):
        return Response.error(
            code="task_not_cancellable",
            message=f"任务已完成或已取消，当前状态: {task.status}",
        )

    # 撤销 Celery 任务
    if task.celery_task_id:
        try:
            revoke_task(task.celery_task_id, terminate=False)
        except Exception as e:
            logger.error("revoke_task_failed", task_id=task_id, error=str(e))

    # 更新状态
    task.status = TaskStatus.CANCELLED
    await session.commit()

    return Response.success(
        data={"task_id": task_id, "status": "cancelled"},
        message="任务已取消",
    )


@router.get("/{task_id}/logs")
async def get_task_logs(
    task_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    level: str | None = Query(None, description="日志级别"),
    session: AsyncSession = Depends(get_session),
    tenant_id: int = Depends(get_current_tenant_id),
) -> Response[dict]:
    """获取任务日志

    Args:
        task_id: 任务 ID
        page: 页码
        size: 每页数量
        level: 日志级别筛选
        session: 数据库会话
        tenant_id: 租户 ID

    Returns:
        任务日志列表
    """
    from sqlalchemy import select

    from app.models.database import Task, TaskLog

    # 验证任务存在
    task_stmt = select(Task).where(
        Task.task_id == task_id,
        Task.tenant_id == tenant_id,
    )
    task_result = await session.execute(task_stmt)
    if not task_result.scalar_one_or_none():
        return Response.error(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
        )

    # 查询日志
    stmt = select(TaskLog).where(
        TaskLog.task_id == task_id,
        TaskLog.tenant_id == tenant_id,
    )

    if level:
        stmt = stmt.where(TaskLog.level == level)

    stmt = stmt.order_by(TaskLog.created_at.asc())

    # 分页
    offset = (page - 1) * size
    stmt = stmt.offset(offset).limit(size)

    result = await session.execute(stmt)
    logs = result.scalars().all()

    # 转换为公开模型
    items = [
        {
            "log_id": log.log_id,
            "level": log.level,
            "message": log.message,
            "step": log.step,
            "item_id": log.item_id,
            "extra_data": log.extra_data,
            "created_at": log.created_at,
        }
        for log in logs
    ]

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
    tenant_id: int = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
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
        session: 数据库会话
        business_id: 业务 ID
        priority: 优先级

    Returns:
        创建的任务信息
    """
    from app.models.database import Task

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

    # 创建任务记录
    task = Task(
        task_id=task_id,
        task_type=task_type,
        tenant_id=tenant_id,
        priority=priority,
        status=TaskStatus.PENDING,
        payload=payload,
        business_id=business_id,
        created_by=user_id,
    )

    session.add(task)
    await session.commit()
    await session.refresh(task)

    # 发送任务到 Celery
    try:
        celery_task_id = send_task(
            task_type=task_type,
            payload=payload,
            tenant_id=tenant_id,
            priority=priority,
        )
        task.celery_task_id = celery_task_id
        await session.commit()
    except Exception as e:
        logger.error("send_task_failed", task_id=task_id, error=str(e))

    return Response.success(
        data=_task_to_public(task),
        message="任务已加入队列",
    )


# ============== 工具函数 ==============


def _task_to_public(task) -> TaskPublic:
    """将 Task 模型转换为 TaskPublic

    Args:
        task: Task 模型

    Returns:
        TaskPublic 实例
    """
    return TaskPublic(
        task_id=task.task_id,
        task_type=task.task_type,
        tenant_id=task.tenant_id,
        priority=task.priority,
        status=task.status,
        title=task.title,
        description=task.description,
        progress=task.progress,
        current_step=task.current_step,
        total_items=task.total_items,
        processed_items=task.processed_items,
        failed_items=task.failed_items,
        result=task.result,
        error_message=task.error_message,
        retry_count=task.retry_count,
        max_retries=task.max_retries,
        business_id=task.business_id,
        business_type=task.business_type,
        parent_task_id=task.parent_task_id,
        duration=task.duration,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


__all__ = ["router"]
