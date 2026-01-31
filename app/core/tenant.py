"""租户上下文管理

使用 ContextVar 在异步环境中传递租户信息。
"""

from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

from app.observability.logging import get_logger

logger = get_logger(__name__)

# ============== 租户上下文变量 ==============

_tenant_context: ContextVar["TenantContext"] = ContextVar("tenant_context", default=None)


class TenantContext:
    """租户上下文

    存储当前请求的租户信息，在整个请求链中传递。
    """

    def __init__(
        self,
        tenant_id: int | None = None,
        tenant_name: str | None = None,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.tenant_id = tenant_id
        self.tenant_name = tenant_name
        self.api_key = api_key
        self.config = config or {}

    def __repr__(self) -> str:
        return f"TenantContext(id={self.tenant_id}, name={self.tenant_name})"

    @property
    def is_authenticated(self) -> bool:
        """是否已认证"""
        return self.tenant_id is not None


def set_tenant_context(ctx: TenantContext) -> None:
    """设置租户上下文"""
    _tenant_context.set(ctx)
    logger.debug("tenant_context_set", tenant_id=ctx.tenant_id)


def get_tenant_context() -> TenantContext | None:
    """获取租户上下文"""
    return _tenant_context.get(None)


def get_tenant_id() -> int | None:
    """获取当前租户 ID"""
    ctx = get_tenant_context()
    return ctx.tenant_id if ctx else None


def get_tenant_name() -> str | None:
    """获取当前租户名称"""
    ctx = get_tenant_context()
    return ctx.tenant_name if ctx else None


def get_tenant_config() -> dict[str, Any]:
    """获取当前租户配置"""
    ctx = get_tenant_context()
    return ctx.config if ctx else {}


def clear_tenant_context() -> None:
    """清除租户上下文"""
    _tenant_context.set(None)


# ============== 租户配置映射 ==============


class TenantConfigMapping(MutableMapping):
    """租户配置字典包装

    提供带默认值的配置访问。
    """

    def __init__(
        self,
        tenant_config: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
    ):
        self._tenant_config = tenant_config or {}
        self._defaults = defaults or {}

    def __getitem__(self, key: str) -> Any:
        if key in self._tenant_config:
            return self._tenant_config[key]
        if key in self._defaults:
            return self._defaults[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._tenant_config[key] = value

    def __delitem__(self, key: str) -> None:
        del self._tenant_config[key]

    def __iter__(self):
        return iter(self._tenant_config)

    def __len__(self) -> int:
        return len(self._tenant_config)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持默认值"""
        if key in self._tenant_config:
            return self._tenant_config[key]
        if key in self._defaults:
            return self._defaults[key]
        return default


# 解决循环导入
if __name__ == "__main__":
    pass
