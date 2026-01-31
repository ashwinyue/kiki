"""Redis-backed task store.

Replaces DB persistence for task status/logs while keeping Celery + Redis.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from app.infra.redis import get_redis
from app.tasks.types import TaskPriority, TaskStatus


class TaskStore:
    """Task storage backed by Redis.

    Key layout:
        task:{tenant_id}:{task_id}          -> hash
        task:logs:{tenant_id}:{task_id}     -> list (json lines)
        tasks:tenant:{tenant_id}            -> zset (score=created_at ts)
        tasks:tenants                       -> set of tenant ids
    """

    _TENANTS_KEY = "tasks:tenants"

    def _task_key(self, tenant_id: int, task_id: str) -> str:
        return f"task:{tenant_id}:{task_id}"

    def _logs_key(self, tenant_id: int, task_id: str) -> str:
        return f"task:logs:{tenant_id}:{task_id}"

    def _tenant_zset(self, tenant_id: int) -> str:
        return f"tasks:tenant:{tenant_id}"

    async def _client(self):
        return await get_redis()

    @staticmethod
    def _encode(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, (TaskPriority, TaskStatus)):
            return value.value
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _decode_json(value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _decode_int(value: str | None) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _decode_float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _decode_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _normalize_task(self, data: dict[str, str]) -> dict[str, Any]:
        created_at = self._decode_dt(data.get("created_at"))
        started_at = self._decode_dt(data.get("started_at"))
        completed_at = self._decode_dt(data.get("completed_at"))

        duration = None
        if started_at and completed_at:
            duration = (completed_at - started_at).total_seconds()
        elif started_at:
            duration = (datetime.now(UTC) - started_at).total_seconds()

        return {
            "task_id": data.get("task_id", ""),
            "task_type": data.get("task_type", ""),
            "tenant_id": self._decode_int(data.get("tenant_id")) or 0,
            "priority": data.get("priority") or TaskPriority.DEFAULT.value,
            "status": data.get("status") or TaskStatus.PENDING.value,
            "title": data.get("title") or None,
            "description": data.get("description") or None,
            "payload": self._decode_json(data.get("payload")),
            "business_id": data.get("business_id") or None,
            "business_type": data.get("business_type") or None,
            "parent_task_id": data.get("parent_task_id") or None,
            "max_retries": self._decode_int(data.get("max_retries")) or 3,
            "total_items": self._decode_int(data.get("total_items")),
            "progress": self._decode_int(data.get("progress")) or 0,
            "current_step": data.get("current_step") or None,
            "processed_items": self._decode_int(data.get("processed_items")),
            "failed_items": self._decode_int(data.get("failed_items")),
            "result": self._decode_json(data.get("result")),
            "error_message": data.get("error_message") or None,
            "error_stack": data.get("error_stack") or None,
            "celery_task_id": data.get("celery_task_id") or None,
            "retry_count": self._decode_int(data.get("retry_count")) or 0,
            "extra_metadata": self._decode_json(data.get("extra_metadata")),
            "created_by": data.get("created_by") or None,
            "created_at": created_at or datetime.now(UTC),
            "started_at": started_at,
            "completed_at": completed_at,
            "updated_at": self._decode_dt(data.get("updated_at")) or datetime.now(UTC),
            "duration": duration,
        }

    async def create_task(
        self,
        *,
        task_id: str,
        task_type: str,
        tenant_id: int,
        priority: TaskPriority | str = TaskPriority.DEFAULT,
        status: TaskStatus | str = TaskStatus.PENDING,
        title: str | None = None,
        description: str | None = None,
        payload: dict[str, Any] | None = None,
        business_id: str | None = None,
        business_type: str | None = None,
        parent_task_id: str | None = None,
        max_retries: int = 3,
        total_items: int | None = None,
        created_by: str | None = None,
        celery_task_id: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        created_ts = int(now.timestamp())
        data = {
            "task_id": task_id,
            "task_type": task_type,
            "tenant_id": tenant_id,
            "priority": priority,
            "status": status,
            "title": title,
            "description": description,
            "payload": payload,
            "business_id": business_id,
            "business_type": business_type,
            "parent_task_id": parent_task_id,
            "max_retries": max_retries,
            "total_items": total_items,
            "progress": 0,
            "current_step": None,
            "processed_items": None,
            "failed_items": None,
            "result": None,
            "error_message": None,
            "error_stack": None,
            "celery_task_id": celery_task_id,
            "retry_count": 0,
            "extra_metadata": None,
            "created_by": created_by,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
        }
        encoded = {k: self._encode(v) for k, v in data.items() if self._encode(v) is not None}
        client = await self._client()
        key = self._task_key(tenant_id, task_id)
        zset = self._tenant_zset(tenant_id)
        pipe = client.pipeline(transaction=True)
        pipe.hset(key, mapping=encoded)
        pipe.zadd(zset, {task_id: created_ts})
        pipe.sadd(self._TENANTS_KEY, str(tenant_id))
        await pipe.execute()
        return data

    async def get_task(self, task_id: str, tenant_id: int) -> dict[str, Any] | None:
        client = await self._client()
        key = self._task_key(tenant_id, task_id)
        data = await client.hgetall(key)
        if not data:
            return None
        return self._normalize_task(data)

    async def update_task(
        self,
        task_id: str,
        tenant_id: int,
        **fields: Any,
    ) -> dict[str, Any] | None:
        if not fields:
            return await self.get_task(task_id, tenant_id)
        fields["updated_at"] = datetime.now(UTC)
        encoded = {k: self._encode(v) for k, v in fields.items() if self._encode(v) is not None}
        client = await self._client()
        key = self._task_key(tenant_id, task_id)
        if not encoded:
            return await self.get_task(task_id, tenant_id)
        await client.hset(key, mapping=encoded)
        return await self.get_task(task_id, tenant_id)

    async def list_tasks(
        self,
        tenant_id: int,
        *,
        page: int = 1,
        size: int = 20,
        task_type: str | None = None,
        status: str | None = None,
        business_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        client = await self._client()
        zset = self._tenant_zset(tenant_id)
        task_ids = await client.zrevrange(zset, 0, -1)
        if not task_ids:
            return [], 0

        # Fetch tasks in order and filter in-memory for simplicity.
        pipe = client.pipeline(transaction=False)
        for task_id in task_ids:
            pipe.hgetall(self._task_key(tenant_id, task_id))
        raw_tasks = await pipe.execute()

        filtered: list[dict[str, Any]] = []
        for raw in raw_tasks:
            if not raw:
                continue
            task = self._normalize_task(raw)
            if task_type and task.get("task_type") != task_type:
                continue
            if status and task.get("status") != status:
                continue
            if business_id and task.get("business_id") != business_id:
                continue
            filtered.append(task)

        total = len(filtered)
        start = (page - 1) * size
        end = start + size
        return filtered[start:end], total

    async def add_log(
        self,
        task_id: str,
        tenant_id: int,
        *,
        level: str,
        message: str,
        step: str | None = None,
        item_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        client = await self._client()
        log_entry = {
            "log_id": f"log_{int(time.time() * 1000)}",
            "task_id": task_id,
            "tenant_id": tenant_id,
            "level": level,
            "message": message,
            "step": step,
            "item_id": item_id,
            "extra_data": extra_data,
            "created_at": datetime.now(UTC).isoformat(),
        }
        key = self._logs_key(tenant_id, task_id)
        await client.rpush(key, json.dumps(log_entry))

    async def list_logs(
        self,
        task_id: str,
        tenant_id: int,
        *,
        page: int = 1,
        size: int = 50,
        level: str | None = None,
    ) -> list[dict[str, Any]]:
        client = await self._client()
        key = self._logs_key(tenant_id, task_id)
        entries = await client.lrange(key, 0, -1)
        if not entries:
            return []
        items: list[dict[str, Any]] = []
        for entry in entries:
            try:
                data = json.loads(entry)
            except json.JSONDecodeError:
                continue
            if level and data.get("level") != level:
                continue
            items.append(data)

        start = (page - 1) * size
        end = start + size
        return items[start:end]

    async def cleanup(self, days: int = 7) -> dict[str, Any]:
        client = await self._client()
        cutoff = datetime.now(UTC) - timedelta(days=days)
        tenants = await client.smembers(self._TENANTS_KEY)
        deleted_tasks = 0
        deleted_logs = 0

        for tenant in tenants:
            tenant_id = int(tenant)
            zset = self._tenant_zset(tenant_id)
            task_ids = await client.zrangebyscore(zset, 0, int(cutoff.timestamp()))
            if not task_ids:
                continue
            pipe = client.pipeline(transaction=False)
            for task_id in task_ids:
                pipe.hgetall(self._task_key(tenant_id, task_id))
            raw_tasks = await pipe.execute()

            to_delete: list[str] = []
            for task_id, raw in zip(task_ids, raw_tasks):
                if not raw:
                    to_delete.append(task_id)
                    continue
                task = self._normalize_task(raw)
                completed_at = task.get("completed_at")
                status = task.get("status")
                if status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
                    if completed_at and completed_at < cutoff:
                        to_delete.append(task_id)

            if not to_delete:
                continue

            pipe = client.pipeline(transaction=True)
            for task_id in to_delete:
                key = self._task_key(tenant_id, task_id)
                log_key = self._logs_key(tenant_id, task_id)
                pipe.delete(key)
                pipe.delete(log_key)
                pipe.zrem(zset, task_id)
                deleted_tasks += 1
                deleted_logs += 1
            await pipe.execute()

        return {
            "deleted_count": deleted_tasks,
            "deleted_logs": deleted_logs,
            "cutoff_time": cutoff.isoformat(),
        }
