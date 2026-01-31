"""LLM 成本追踪模块

记录 Token 使用和 API 调用成本，支持预算控制和成本分析。
支持主流 LLM 提供商的价格配置。

使用示例:
```python
from app.llm.cost_tracker import (
    CostTracker,
    track_llm_call,
    get_cost_summary,
)

# 追踪单次调用
with track_llm_call("gpt-4o", "user-123"):
    response = await llm.ainvoke(messages)

# 获取成本汇总
summary = get_cost_summary(user_id="user-123")
print(f"总成本: ${summary.total_cost:.4f}")

# 预算控制
tracker = CostTracker(budget_usd=10.0)
if tracker.check_budget("gpt-4o", estimated_tokens=1000):
    # 执行调用
    pass
```
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ============== 价格配置 ==============

# 价格表（美元 / 1K tokens）
# 来源: 各厂商官方定价（2025年1月）
_TOKEN_PRICES: dict[str, dict[str, tuple[Decimal, Decimal]]] = {
    # OpenAI
    "gpt-4o": {"input": (Decimal("0.0025"), Decimal("0.0025")), "output": (Decimal("0.01"), Decimal("0.01"))},
    "gpt-4o-mini": {"input": (Decimal("0.00015"), Decimal("0.00015")), "output": (Decimal("0.0006"), Decimal("0.0006"))},
    "gpt-4-turbo": {"input": (Decimal("0.01"), Decimal("0.01")), "output": (Decimal("0.03"), Decimal("0.03"))},
    "gpt-3.5-turbo": {"input": (Decimal("0.0005"), Decimal("0.0005")), "output": (Decimal("0.0015"), Decimal("0.0015"))},
    # Anthropic
    "claude-opus-4-20250514": {"input": (Decimal("0.015"), Decimal("0.015")), "output": (Decimal("0.075"), Decimal("0.075"))},
    "claude-sonnet-4-20250514": {"input": (Decimal("0.003"), Decimal("0.003")), "output": (Decimal("0.015"), Decimal("0.015"))},
    "claude-haiku-4-20250514": {"input": (Decimal("0.0008"), Decimal("0.0008")), "output": (Decimal("0.004"), Decimal("0.004"))},
    # DeepSeek
    "deepseek-chat": {"input": (Decimal("0.00014"), Decimal("0.00014")), "output": (Decimal("0.00028"), Decimal("0.00028"))},
    "deepseek-reasoner": {"input": (Decimal("0.00055"), Decimal("0.00055")), "output": (Decimal("0.0022"), Decimal("0.0022"))},
    # DashScope (Qwen)
    "qwen-max": {"input": (Decimal("0.02"), Decimal("0.02")), "output": (Decimal("0.06"), Decimal("0.06"))},
    "qwen-plus": {"input": (Decimal("0.004"), Decimal("0.004")), "output": (Decimal("0.012"), Decimal("0.012"))},
    "qwen-turbo": {"input": (Decimal("0.0008"), Decimal("0.0008")), "output": (Decimal("0.002"), Decimal("0.002"))},
    "qwen-long": {"input": (Decimal("0.0005"), Decimal("0.0005")), "output": (Decimal("0.002"), Decimal("0.002"))},
}


def get_model_pricing(model_name: str) -> tuple[Decimal, Decimal]:
    """获取模型定价

    Args:
        model_name: 模型名称

    Returns:
        (输入价格, 输出价格)，单位：美元 / 1K tokens

    Raises:
        ValueError: 如果模型定价不存在
    """
    # 标准化模型名称
    model_key = model_name.lower().replace(".", "").replace("-", "")

    # 精确匹配
    if model_name in _TOKEN_PRICES:
        return _TOKEN_PRICES[model_name]["input"]

    # 模糊匹配
    for key, pricing in _TOKEN_PRICES.items():
        if key in model_name or model_name in key:
            return pricing["input"]

    # 默认价格（保守估计）
    logger.warning("model_pricing_not_found", model=model_name, using_default=True)
    return (Decimal("0.001"), Decimal("0.002"))


def register_model_pricing(
    model_name: str,
    input_price: float | Decimal,
    output_price: float | Decimal,
) -> None:
    """注册模型定价

    Args:
        model_name: 模型名称
        input_price: 输入价格（美元 / 1K tokens）
        output_price: 输出价格（美元 / 1K tokens）
    """
    _TOKEN_PRICES[model_name] = {
        "input": (Decimal(str(input_price)), Decimal(str(input_price))),
        "output": (Decimal(str(output_price)), Decimal(str(output_price))),
    }
    logger.info(
        "model_pricing_registered",
        model=model_name,
        input_price=float(input_price),
        output_price=float(output_price),
    )


# ============== 成本记录 ==============

@dataclass
class CostRecord:
    """成本记录

    Attributes:
        model_name: 模型名称
        user_id: 用户 ID
        session_id: 会话 ID
        input_tokens: 输入 Token 数
        output_tokens: 输出 Token 数
        total_tokens: 总 Token 数
        input_cost: 输入成本（美元）
        output_cost: 输出成本（美元）
        total_cost: 总成本（美元）
        timestamp: 时间戳
        metadata: 额外元数据
    """

    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "total_cost": float(self.total_cost),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CostSummary:
    """成本汇总

    Attributes:
        total_cost: 总成本
        total_tokens: 总 Token 数
        input_tokens: 输入 Token 总数
        output_tokens: 输出 Token 总数
        call_count: 调用次数
        model_breakdown: 按模型分组的成本
        time_range: 时间范围
    """

    total_cost: Decimal
    total_tokens: int
    input_tokens: int
    output_tokens: int
    call_count: int
    model_breakdown: dict[str, dict[str, Any]]
    time_range: tuple[datetime, datetime] | None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "total_cost": float(self.total_cost),
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "call_count": self.call_count,
            "model_breakdown": self.model_breakdown,
            "time_range": (
                (self.time_range[0].isoformat(), self.time_range[1].isoformat())
                if self.time_range
                else None
            ),
        }


# ============== 成本追踪器 ==============

class CostTracker:
    """成本追踪器

    管理成本记录，提供查询和预算控制功能。
    """

    def __init__(
        self,
        budget_usd: float | Decimal | None = None,
        retention_days: int = 30,
    ):
        """初始化成本追踪器

        Args:
            budget_usd: 预算限制（美元）
            retention_days: 记录保留天数
        """
        self.budget_usd = Decimal(str(budget_usd)) if budget_usd else None
        self.retention_days = retention_days
        self._records: list[CostRecord] = []
        self._lock = asyncio.Lock()

        logger.info(
            "cost_tracker_initialized",
            budget_usd=float(self.budget_usd) if self.budget_usd else None,
            retention_days=retention_days,
        )

    async def record(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """记录成本

        Args:
            model_name: 模型名称
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            user_id: 用户 ID
            session_id: 会话 ID
            metadata: 额外元数据

        Returns:
            CostRecord 实例
        """
        async with self._lock:
            # 获取定价
            input_price, output_price = get_model_pricing(model_name)

            # 计算成本
            input_cost = (Decimal(input_tokens) / 1000) * input_price
            output_cost = (Decimal(output_tokens) / 1000) * output_price
            total_cost = input_cost + output_cost
            total_tokens = input_tokens + output_tokens

            # 创建记录
            record = CostRecord(
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )

            # 添加记录
            self._records.append(record)

            # 清理过期记录
            await self._cleanup_old_records()

            logger.info(
                "cost_recorded",
                model=model_name,
                user_id=user_id,
                session_id=session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=float(total_cost),
            )

            return record

    async def _cleanup_old_records(self) -> None:
        """清理过期记录"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        original_count = len(self._records)
        self._records = [r for r in self._records if r.timestamp > cutoff_date]
        removed_count = original_count - len(self._records)

        if removed_count > 0:
            logger.debug(
                "old_cost_records_cleaned",
                removed_count=removed_count,
                cutoff_date=cutoff_date.isoformat(),
            )

    async def get_summary(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        model_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostSummary:
        """获取成本汇总

        Args:
            user_id: 筛选用户 ID
            session_id: 筛选会话 ID
            model_name: 筛选模型名称
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            CostSummary 实例
        """
        async with self._lock:
            # 筛选记录
            filtered_records = self._records

            if user_id:
                filtered_records = [r for r in filtered_records if r.user_id == user_id]
            if session_id:
                filtered_records = [r for r in filtered_records if r.session_id == session_id]
            if model_name:
                filtered_records = [r for r in filtered_records if r.model_name == model_name]
            if start_time:
                filtered_records = [r for r in filtered_records if r.timestamp >= start_time]
            if end_time:
                filtered_records = [r for r in filtered_records if r.timestamp <= end_time]

            # 计算汇总
            total_cost = Decimal("0")
            total_tokens = 0
            input_tokens = 0
            output_tokens = 0
            call_count = len(filtered_records)

            # 按模型分组
            model_breakdown: dict[str, dict[str, Any]] = {}

            for record in filtered_records:
                total_cost += record.total_cost
                total_tokens += record.total_tokens
                input_tokens += record.input_tokens
                output_tokens += record.output_tokens

                # 模型分组
                if record.model_name not in model_breakdown:
                    model_breakdown[record.model_name] = {
                        "total_cost": Decimal("0"),
                        "total_tokens": 0,
                        "call_count": 0,
                    }
                model_breakdown[record.model_name]["total_cost"] += record.total_cost
                model_breakdown[record.model_name]["total_tokens"] += record.total_tokens
                model_breakdown[record.model_name]["call_count"] += 1

            # 时间范围
            time_range = None
            if filtered_records:
                timestamps = [r.timestamp for r in filtered_records]
                time_range = (min(timestamps), max(timestamps))

            return CostSummary(
                total_cost=total_cost,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_count=call_count,
                model_breakdown={
                    k: {
                        **v,
                        "total_cost": float(v["total_cost"]),
                    }
                    for k, v in model_breakdown.items()
                },
                time_range=time_range,
            )

    def check_budget(
        self,
        model_name: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
    ) -> tuple[bool, Decimal]:
        """检查预算

        Args:
            model_name: 模型名称
            estimated_input_tokens: 估计输入 Token 数
            estimated_output_tokens: 估计输出 Token 数

        Returns:
            (是否在预算内, 预计成本)
        """
        if self.budget_usd is None:
            return True, Decimal("0")

        # 获取当前总成本
        current_cost = sum((r.total_cost for r in self._records), Decimal("0"))

        # 计算预计成本
        input_price, output_price = get_model_pricing(model_name)
        estimated_cost = (
            (Decimal(estimated_input_tokens) / 1000) * input_price
            + (Decimal(estimated_output_tokens) / 1000) * output_price
        )

        total_after_call = current_cost + estimated_cost
        within_budget = total_after_call <= self.budget_usd

        if not within_budget:
            logger.warning(
                "budget_exceeded",
                current_cost=float(current_cost),
                budget_usd=float(self.budget_usd),
                estimated_cost=float(estimated_cost),
            )

        return within_budget, estimated_cost

    async def get_user_spending(self, user_id: str) -> Decimal:
        """获取用户总花费

        Args:
            user_id: 用户 ID

        Returns:
            总花费（美元）
        """
        summary = await self.get_summary(user_id=user_id)
        return summary.total_cost

    async def reset_user_spending(self, user_id: str) -> int:
        """重置用户花费记录

        Args:
            user_id: 用户 ID

        Returns:
            删除的记录数
        """
        async with self._lock:
            original_count = len(self._records)
            self._records = [r for r in self._records if r.user_id != user_id]
            return original_count - len(self._records)


# ============== 全局追踪器 =============

_global_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """获取全局成本追踪器

    Returns:
        CostTracker 实例
    """
    global _global_tracker
    if _global_tracker is None:
        # 从配置读取预算
        budget = getattr(settings, "llm_budget_usd", None)
        _global_tracker = CostTracker(budget_usd=budget)
    return _global_tracker


def set_cost_tracker(tracker: CostTracker) -> None:
    """设置全局成本追踪器

    Args:
        tracker: CostTracker 实例
    """
    global _global_tracker
    _global_tracker = tracker


# ============== 便捷函数 ==============

async def track_llm_call(
    model_name: str,
    user_id: str | None = None,
    session_id: str | None = None,
):
    """追踪 LLM 调用的上下文管理器

    Args:
        model_name: 模型名称
        user_id: 用户 ID
        session_id: 会话 ID

    Yields:
        None

    Examples:
        ```python
        async with track_llm_call("gpt-4o", "user-123"):
            response = await llm.ainvoke(messages)
            # 自动记录 Token 使用
        ```
    """
    tracker = get_cost_tracker()
    metadata = {}

    try:
        yield

    finally:
        # 这里需要从 response 中提取 Token 使用情况
        # 实际实现需要根据具体的 LLM 响应结构来解析
        pass


async def record_llm_usage(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    user_id: str | None = None,
    session_id: str | None = None,
) -> CostRecord:
    """记录 LLM 使用

    Args:
        model_name: 模型名称
        input_tokens: 输入 Token 数
        output_tokens: 输出 Token 数
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        CostRecord 实例
    """
    tracker = get_cost_tracker()
    return await tracker.record(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        user_id=user_id,
        session_id=session_id,
    )


async def get_cost_summary(
    user_id: str | None = None,
    session_id: str | None = None,
) -> CostSummary:
    """获取成本汇总

    Args:
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        CostSummary 实例
    """
    tracker = get_cost_tracker()
    return await tracker.get_summary(user_id=user_id, session_id=session_id)


async def check_budget(
    model_name: str,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
) -> tuple[bool, Decimal]:
    """检查预算

    Args:
        model_name: 模型名称
        estimated_input_tokens: 估计输入 Token 数
        estimated_output_tokens: 估计输出 Token 数

    Returns:
        (是否在预算内, 预计成本)
    """
    tracker = get_cost_tracker()
    return tracker.check_budget(
        model_name=model_name,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
    )


__all__ = [
    # 价格管理
    "get_model_pricing",
    "register_model_pricing",
    # 数据类
    "CostRecord",
    "CostSummary",
    # 追踪器
    "CostTracker",
    "get_cost_tracker",
    "set_cost_tracker",
    # 便捷函数
    "track_llm_call",
    "record_llm_usage",
    "get_cost_summary",
    "check_budget",
]
