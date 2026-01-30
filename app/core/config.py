"""配置管理

支持环境感知配置，从环境变量加载配置。

环境变量命名规范：
- KIKI_APP_NAME
- KIKI_DATABASE_URL
- KIKI_LLM__MODEL (嵌套使用 __)
"""

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """环境类型"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

    @property
    def is_development(self) -> bool:
        return self == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self == Environment.PRODUCTION

    @property
    def is_test(self) -> bool:
        return self == Environment.TEST


def detect_environment() -> Environment:
    """检测当前环境"""
    import os

    env = os.getenv("KIKI_ENV", os.getenv("ENVIRONMENT", "development"))
    try:
        return Environment(env)
    except ValueError:
        return Environment.DEVELOPMENT


class Settings(BaseSettings):
    """应用配置"""

    # ========== 应用配置 ==========
    app_name: str = "Kiki Agent"
    app_version: str = "0.1.0"
    environment: Environment = Field(default_factory=detect_environment)
    debug: bool = False

    # ========== 服务器配置 ==========
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"

    # ========== 数据库配置 ==========
    database_url: str = "postgresql+asyncpg://localhost:5432/kiki"
    database_pool_size: int = 20
    database_echo: bool = False

    # ========== LLM 配置 ==========
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    # ========== 认证配置 ==========
    secret_key: str = "change-me-in-production-min-32-chars"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ========== 可观测性配置 ==========
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # LangSmith
    langchain_api_key: str | None = None
    langchain_project: str = "kiki-agent"
    langchain_tracing_v2: bool = False

    # Prometheus
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"

    # ========== Redis 配置 ==========
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 10
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    redis_decode_responses: bool = True

    # ========== 对象存储配置 ==========
    storage_type: Literal["local", "minio", "cos", "base64"] = "local"

    # MinIO 配置
    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str | None = None
    minio_access_key_id: str = "minioadmin"
    minio_secret_access_key: str = "minioadmin"
    minio_bucket_name: str = "kiki"
    minio_path_prefix: str = ""
    minio_use_ssl: bool = False

    # 腾讯云 COS 配置
    cos_secret_id: str = ""
    cos_secret_key: str = ""
    cos_region: str = ""
    cos_bucket_name: str = ""
    cos_app_id: str = ""
    cos_path_prefix: str = ""
    cos_enable_old_domain: bool = True

    # 本地存储配置
    local_storage_base_dir: str = "./data/files"

    # ========== 会话上下文配置 ==========
    context_storage_type: Literal["memory", "redis"] = "memory"
    context_ttl_hours: int = 24
    context_max_messages: int = 100
    context_max_tokens: int = 128_000

    model_config = SettingsConfigDict(
        env_prefix="kiki_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """验证生产环境的密钥"""
        if info.data.get("environment") == Environment.PRODUCTION:
            if v == "change-me-in-production-min-32-chars" or len(v) < 32:
                raise ValueError(
                    "生产环境必须设置至少 32 字符的 KIKI_SECRET_KEY"
                )
        return v

    @property
    def is_development(self) -> bool:
        return self.environment.is_development

    @property
    def is_production(self) -> bool:
        return self.environment.is_production

    @property
    def is_test(self) -> bool:
        return self.environment.is_test

    def model_post_init(self, __context) -> None:
        """初始化后处理"""
        # 环境特定配置覆盖
        if self.is_production:
            self.debug = False
            if self.log_format == "console":
                self.log_format = "json"
        elif self.is_development:
            self.debug = True


# 全局配置实例
_settings: Settings | None = None


def get_settings() -> Settings:
    """获取配置实例（单例）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """重新加载配置"""
    global _settings
    _settings = Settings()
    return _settings
