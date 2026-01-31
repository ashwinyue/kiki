"""任务 ID 生成与解析工具

对齐 WeKnora99 的任务 ID 格式 (internal/utils/taskid.go)

格式: <taskType>_<tenantID>_<timestamp>_<uuid>_<businessID>

示例:
    - faq:import_12345_1704628851692_a1b2c3d4_kb789
    - kb:clone_12345_1704628851692_a1b2c3d4_kb789

使用方式:
    >>> from app.tasks.task_id import generate_task_id, parse_task_id
    >>> task_id = generate_task_id("faq:import", tenant_id=12345, business_id="kb789")
    >>> print(task_id)
    'faq_import_12345_1704628851692_a1b2c3d4_kb789'
    >>> parsed = parse_task_id(task_id)
    >>> print(parsed)
    ParsedTaskID(task_type='faq_import', tenant_id=12345, timestamp=1704628851692, uuid='a1b2c3d4', business_id='kb789')
"""

import re
import time
import uuid
from dataclasses import dataclass
from typing import Self


def sanitize_task_type(task_type: str) -> str:
    """清理任务类型，使其适合用于任务 ID

    Args:
        task_type: 原始任务类型

    Returns:
        清理后的任务类型
    """
    # 替换特殊字符为下划线
    task_type = task_type.replace(":", "_")
    task_type = task_type.replace("-", "_")
    task_type = task_type.replace(" ", "_")
    return task_type.lower()


def sanitize_business_id(business_id: str) -> str:
    """清理业务 ID，使其适合用于任务 ID

    Args:
        business_id: 原始业务 ID

    Returns:
        清理后的业务 ID (最多 12 字符)
    """
    # 取前 12 个字符并替换特殊字符
    if len(business_id) > 12:
        business_id = business_id[:12]
    business_id = business_id.replace("-", "")
    business_id = business_id.replace("_", "")
    business_id = business_id.replace(":", "")
    return business_id


def generate_task_id(
    task_type: str,
    tenant_id: int,
    business_id: str | None = None,
) -> str:
    """生成唯一任务 ID

    格式: <taskType>_<tenantID>_<timestamp>_<uuid>_<businessID>

    Args:
        task_type: 任务类型 (如 "faq:import", "kb:clone")
        tenant_id: 租户 ID
        business_id: 可选的业务相关 ID (如知识库 ID)

    Returns:
        任务 ID 字符串

    Examples:
        >>> generate_task_id("faq:import", 12345, "kb789")
        'faq_import_12345_1704628851692_a1b2c3d4_kb789'

        >>> generate_task_id("kb:clone", 12345)
        'kb_clone_12345_1704628851692_a1b2c3d4'
    """
    # 使用毫秒级时间戳保证时序唯一性
    timestamp = int(time.time() * 1000)

    # 生成短 UUID (前 8 位)
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]

    # 构建组件
    components = [
        sanitize_task_type(task_type),
        str(tenant_id),
        str(timestamp),
        short_uuid,
    ]

    # 添加业务 ID (如果提供)
    if business_id:
        components.append(sanitize_business_id(business_id))

    return "_".join(components)


def generate_task_id_with_prefix(
    prefix: str,
    tenant_id: int,
    business_id: str | None = None,
) -> str:
    """生成带自定义前缀的任务 ID

    Args:
        prefix: 自定义前缀
        tenant_id: 租户 ID
        business_id: 可选的业务相关 ID

    Returns:
        任务 ID 字符串
    """
    timestamp = int(time.time() * 1000)
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]

    components = [
        sanitize_task_type(prefix),
        str(tenant_id),
        str(timestamp),
        short_uuid,
    ]

    if business_id:
        components.append(sanitize_business_id(business_id))

    return "_".join(components)


@dataclass
class ParsedTaskID:
    """解析后的任务 ID

    Attributes:
        task_type: 任务类型
        tenant_id: 租户 ID
        timestamp: 时间戳 (毫秒)
        uuid: UUID 部分
        business_id: 业务 ID (可选)
        raw: 原始任务 ID
    """

    task_type: str
    tenant_id: int
    timestamp: int
    uuid: str
    business_id: str | None = None
    raw: str = ""

    @property
    def datetime(self) -> float:
        """时间戳转换为 datetime 浮点数"""
        return self.timestamp / 1000


def parse_task_id(task_id: str) -> ParsedTaskID:
    """解析任务 ID

    Args:
        task_id: 任务 ID 字符串

    Returns:
        ParsedTaskID 对象

    Raises:
        ValueError: 如果任务 ID 格式无效

    Examples:
        >>> parse_task_id("faq_import_12345_1704628851692_a1b2c3d4_kb789")
        ParsedTaskID(task_type='faq_import', tenant_id=12345, timestamp=1704628851692, uuid='a1b2c3d4', business_id='kb789')

        >>> parse_task_id("kb_clone_12345_1704628851692_a1b2c3d4")
        ParsedTaskID(task_type='kb_clone', tenant_id=12345, timestamp=1704628851692, uuid='a1b2c3d4', business_id=None)
    """
    parts = task_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid task ID format: {task_id}")

    task_type = parts[0]

    try:
        tenant_id = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid tenant ID in task ID: {parts[1]}")

    try:
        timestamp = int(parts[2])
    except ValueError:
        raise ValueError(f"Invalid timestamp in task ID: {parts[2]}")

    uuid_part = parts[3]

    business_id = parts[4] if len(parts) > 4 else None

    return ParsedTaskID(
        task_type=task_type,
        tenant_id=tenant_id,
        timestamp=timestamp,
        uuid=uuid_part,
        business_id=business_id,
        raw=task_id,
    )


def validate_task_id(task_id: str) -> bool:
    """验证任务 ID 格式

    Args:
        task_id: 任务 ID 字符串

    Returns:
        是否有效
    """
    try:
        parse_task_id(task_id)
        return True
    except ValueError:
        return False


def extract_task_type(task_id: str) -> str | None:
    """从任务 ID 中提取任务类型

    Args:
        task_id: 任务 ID 字符串

    Returns:
        任务类型，如果解析失败返回 None
    """
    try:
        parsed = parse_task_id(task_id)
        return parsed.task_type
    except ValueError:
        return None


def extract_tenant_id(task_id: str) -> int | None:
    """从任务 ID 中提取租户 ID

    Args:
        task_id: 任务 ID 字符串

    Returns:
        租户 ID，如果解析失败返回 None
    """
    try:
        parsed = parse_task_id(task_id)
        return parsed.tenant_id
    except ValueError:
        return None


# ============== 任务 ID 匹配模式 ==============


def build_task_id_pattern(
    task_type: str | None = None,
    tenant_id: int | None = None,
    business_id: str | None = None,
) -> str:
    """构建任务 ID 匹配模式

    用于数据库查询或 Redis key 匹配。

    Args:
        task_type: 任务类型 (可选)
        tenant_id: 租户 ID (可选)
        business_id: 业务 ID (可选)

    Returns:
        匹配模式字符串 (支持 * 通配符)

    Examples:
        >>> build_task_id_pattern(task_type="faq_import", tenant_id=12345)
        'faq_import_12345_*'

        >>> build_task_id_pattern(tenant_id=12345)
        '*_12345_*'
    """
    parts = []

    if task_type:
        parts.append(sanitize_task_type(task_type))
    else:
        parts.append("*")

    if tenant_id is not None:
        parts.append(str(tenant_id))
    else:
        parts.append("*")

    # timestamp 和 uuid 位置用 * 代替
    parts.extend(["*", "*"])

    if business_id:
        parts.append(sanitize_business_id(business_id))

    return "_".join(parts)


def matches_task_id(
    task_id: str,
    task_type: str | None = None,
    tenant_id: int | None = None,
    business_id: str | None = None,
) -> bool:
    """检查任务 ID 是否匹配指定条件

    Args:
        task_id: 任务 ID 字符串
        task_type: 任务类型 (可选)
        tenant_id: 租户 ID (可选)
        business_id: 业务 ID (可选)

    Returns:
        是否匹配
    """
    try:
        parsed = parse_task_id(task_id)
    except ValueError:
        return False

    if task_type and parsed.task_type != sanitize_task_type(task_type):
        return False

    if tenant_id is not None and parsed.tenant_id != tenant_id:
        return False

    if business_id and parsed.business_id != sanitize_business_id(business_id):
        return False

    return True


# ============== 导出 ==============

__all__ = [
    # 生成函数
    "generate_task_id",
    "generate_task_id_with_prefix",
    # 解析函数
    "parse_task_id",
    "validate_task_id",
    "extract_task_type",
    "extract_tenant_id",
    # 匹配函数
    "build_task_id_pattern",
    "matches_task_id",
    # 数据类
    "ParsedTaskID",
    # 工具函数
    "sanitize_task_type",
    "sanitize_business_id",
]
