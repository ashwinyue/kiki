"""配置管理

支持多源配置，优先级从高到低：
1. 环境变量（KIKI_*）
2. YAML 配置文件（conf.yaml）
3. 默认值

环境变量命名规范：
- KIKI_APP_NAME
- KIKI_DATABASE_URL
- KIKI_LLM_MODEL (单下划线分隔)
"""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

# 加载 .env 文件（必须在任何导入之前执行）
from dotenv import load_dotenv  # noqa: E402
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # noqa: E402

# YAML 配置文件路径
CONFIG_FILE = Path("conf.yaml")


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

    app_name: str = "Kiki Agent"
    app_version: str = "0.1.0"
    environment: Environment = Field(default_factory=detect_environment)
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"

    cors_allow_origins: list[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "*"]  # * 在开发环境允许所有
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    database_url: str = "postgresql+asyncpg://localhost:5432/kiki"
    database_pool_size: int = 20
    database_echo: bool = False
    llm_provider: Literal["openai", "anthropic", "ollama", "dashscope", "deepseek", "mock"] = (
        "openai"
    )
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"

    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    llm_enable_multi_provider: bool = True
    llm_default_priority: Literal["cost", "quality", "speed", "balanced"] = "balanced"
    secret_key: str = "change-me-in-production-min-32-chars"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    jwt_algorithm: str = "HS256"

    tenant_aes_key: str = ""  # AES-256, 32字节，生成方法: openssl rand -base64 32
    enable_cross_tenant: bool = False
    default_storage_quota: int = 10 * 1024 * 1024 * 1024  # 10GB

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    langchain_api_key: str | None = None
    langchain_project: str = "kiki-agent"
    langchain_tracing_v2: bool = False

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    prometheus_port: int = 9090
    metrics_path: str = "/metrics"

    elk_enabled: bool = False
    elk_host: str = "localhost"
    elk_port: int = 5044
    elk_timeout: float = 5.0
    elk_max_retries: int = 3
    elk_batch_size: int = 10
    elk_batch_timeout: float = 5.0
    elk_fallback_enabled: bool = True

    rate_limit_enabled: bool = True
    rate_limit_default_rate: float = 10.0  # 令牌/秒
    rate_limit_default_burst: int = 50
    rate_limit_per_ip_enabled: bool = True
    rate_limit_per_user_enabled: bool = True

    audit_enabled: bool = True
    audit_db_enabled: bool = False
    audit_file_enabled: bool = True
    audit_retention_days: int = 90
    audit_log_dir: str = "./logs/audit"

    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 10
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    redis_decode_responses: bool = True

    storage_type: Literal["local", "minio", "cos", "base64"] = "local"

    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str | None = None
    minio_access_key_id: str = "minioadmin"
    minio_secret_access_key: str = "minioadmin"
    minio_bucket_name: str = "kiki"
    minio_path_prefix: str = ""
    minio_use_ssl: bool = False

    cos_secret_id: str = ""
    cos_secret_key: str = ""
    cos_region: str = ""
    cos_bucket_name: str = ""
    cos_app_id: str = ""
    cos_path_prefix: str = ""

    local_storage_base_dir: str = "./data/files"

    context_storage_type: Literal["memory", "redis"] = "memory"
    context_ttl_hours: int = 24
    context_max_messages: int = 100
    context_max_tokens: int = 128_000

    agent_max_messages: int = 100
    agent_max_iterations: int = 50
    agent_max_retries: int = 3
    agent_retry_initial_interval: float = 0.5
    agent_retry_backoff_factor: float = 2.0
    agent_retry_max_interval: float = 60.0

    chat_stream_checkpoint_saver: bool = True
    chat_stream_retention_days: int = 30

    embedding_provider: Literal["openai", "dashscope", "voyage", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1024
    vector_store_type: Literal["qdrant", "pgvector", "pinecone", "chroma", "memory"] = "memory"

    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = "./data/qdrant"
    qdrant_port: int = 6333

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "kiki"
    pinecone_region: str = "us-east-1"

    chroma_persist_directory: str = "./data/chroma"

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
                raise ValueError("生产环境必须设置至少 32 字符的 KIKI_SECRET_KEY")
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

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def model_post_init(self, __context) -> None:
        """初始化后处理"""
        if self.is_production:
            self.debug = False
            if self.log_format == "console":
                self.log_format = "json"
        elif self.is_development:
            self.debug = True


def _load_yaml_config() -> dict[str, Any]:
    """从 YAML 文件加载配置

    Returns:
        配置字典，文件不存在时返回空字典
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        import yaml

        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # YAML 解析失败时返回空字典，让环境变量生效
        return {}


def _merge_config_with_defaults(
    yaml_config: dict[str, Any],
    settings_class: type[BaseSettings],
) -> dict[str, Any]:
    """合并 YAML 配置与默认值

    Args:
        yaml_config: YAML 配置字典
        settings_class: Settings 类

    Returns:
        合并后的配置字典
    """
    # 获取所有字段的默认值
    merged: dict[str, Any] = {}
    for field_name, field_info in settings_class.model_fields.items():
        # 优先使用 YAML 配置
        if field_name in yaml_config:
            merged[field_name] = yaml_config[field_name]
        elif field_info.default is not None:
            merged[field_name] = field_info.default
        elif field_info.is_required():
            # 必填字段没有默认值，让它使用 Pydantic 的验证逻辑
            pass

    return merged


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取配置实例（单例）

    配置加载优先级：
    1. 环境变量（KIKI_*）
    2. YAML 配置文件（conf.yaml）
    3. Pydantic 默认值

    Returns:
        Settings 实例
    """
    global _settings
    if _settings is None:
        # 先加载 YAML 配置
        yaml_config = _load_yaml_config()

        # 合并默认值
        merged_config = _merge_config_with_defaults(yaml_config, Settings)

        # 创建 Settings 实例（环境变量会覆盖 YAML 配置）
        _settings = Settings(**merged_config)
    return _settings


def reload_settings() -> Settings:
    """重新加载配置

    Returns:
        新的 Settings 实例
    """
    global _settings
    _settings = None
    return get_settings()
