"""配置加载器（DeerFlow 风格）

参考 DeerFlow 的配置加载设计，支持：
1. YAML 配置文件加载
2. 环境变量替换（$VAR_NAME 语法）
3. 配置缓存机制
4. 类型安全的辅助函数
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 配置缓存
_config_cache: dict[str, dict[str, Any]] = {}


@lru_cache(maxsize=32)
def load_yaml_config(file_path: str) -> dict[str, Any]:
    """加载 YAML 配置文件（带缓存）

    Args:
        file_path: 配置文件路径

    Returns:
        配置字典（文件不存在时返回空字典）

    特性：
        - 文件不存在返回空字典
        - 配置缓存机制
        - 递归处理环境变量替换

    示例：
        ```python
        config = load_yaml_config("conf.yaml")
        ```
    """
    # 文件不存在
    if not os.path.exists(file_path):
        logger.debug("config_file_not_found", file_path=file_path)
        return {}

    # 检查缓存
    if file_path in _config_cache:
        logger.debug("config_loaded_from_cache", file_path=file_path)
        return _config_cache[file_path]

    try:
        with open(file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # 递归处理环境变量
        processed_config = _process_env_vars(config)

        # 存入缓存
        _config_cache[file_path] = processed_config

        logger.info(
            "config_loaded",
            file_path=file_path,
            keys_count=len(processed_config),
        )

        return processed_config

    except yaml.YAMLError as e:
        logger.error("yaml_parse_error", file_path=file_path, error=str(e))
        return {}
    except Exception as e:
        logger.error("config_load_error", file_path=file_path, error=str(e))
        return {}


def _process_env_vars(config: Any) -> Any:
    """递归处理配置中的环境变量

    支持 $VAR_NAME 和 ${VAR_NAME} 语法

    Args:
        config: 任意类型的配置值

    Returns:
        处理后的值
    """
    if isinstance(config, dict):
        return {k: _process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_process_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("$"):
        # 移除 $ 前缀
        env_var = config[1:]
        # 支持 ${VAR} 语法
        if env_var.startswith("{") and env_var.endswith("}"):
            env_var = env_var[1:-1]
        # 从环境变量获取值
        return os.getenv(env_var, env_var)
    return config


def get_str_env(name: str, default: str = "") -> str:
    """获取字符串环境变量

    Args:
        name: 环境变量名
        default: 默认值

    Returns:
        环境变量值或默认值
    """
    val = os.getenv(name)
    return default if val is None else val.strip()


def get_int_env(name: str, default: int = 0) -> int:
    """获取整数环境变量

    Args:
        name: 环境变量名
        default: 默认值

    Returns:
        整数值或默认值
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        logger.warning("invalid_int_env", name=name, value=val, default=default)
        return default


def get_bool_env(name: str, default: bool = False) -> bool:
    """获取布尔环境变量

    Args:
        name: 环境变量名
        default: 默认值

    Returns:
        布尔值

    支持：1, true, yes, y, on → True
    其他 → False
    """
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_float_env(name: str, default: float = 0.0) -> float:
    """获取浮点数环境变量

    Args:
        name: 环境变量名
        default: 默认值

    Returns:
        浮点数值或默认值
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        logger.warning("invalid_float_env", name=name, value=val, default=default)
        return default


def get_list_env(
    name: str,
    default: list[str] | None = None,
    separator: str = ",",
) -> list[str]:
    """获取列表环境变量

    Args:
        name: 环境变量名
        default: 默认值
        separator: 分隔符（默认逗号）

    Returns:
        字符串列表

    示例：
        ALLOWED_HOSTS="localhost,127.0.0.1,0.0.0.0"
        → ["localhost", "127.0.0.1", "0.0.0.0"]
    """
    if default is None:
        default = []
    val = os.getenv(name)
    if val is None:
        return default

    items = [item.strip() for item in val.split(separator)]
    # 过滤空字符串
    return [item for item in items if item]


def reload_config(file_path: str) -> dict[str, Any]:
    """重新加载配置文件（清除缓存）

    Args:
        file_path: 配置文件路径

    Returns:
        配置字典
    """
    if file_path in _config_cache:
        del _config_cache[file_path]

    # 清除 lru_cache
    load_yaml_config.cache_clear()

    return load_yaml_config(file_path)


__all__ = [
    "load_yaml_config",
    "get_str_env",
    "get_int_env",
    "get_bool_env",
    "get_float_env",
    "get_list_env",
    "reload_config",
]
