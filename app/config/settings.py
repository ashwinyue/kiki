"""配置管理

支持环境感知配置，从环境变量加载配置。

环境变量命名规范：
- KIKI_APP_NAME
- KIKI_DATABASE_URL
- KIKI_LLM_MODEL (单下划线分隔)
"""

from enum import Enum
from typing import Literal

# 加载 .env 文件（必须在任何导入之前执行）
from dotenv import load_dotenv  # noqa: E402
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # noqa: E402


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

    # CORS 允许的源
    cors_allow_origins: list[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # TrustedHost 允许的主机
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "*"]  # * 在开发环境允许所有

    # 最大请求大小（字节）
    max_request_size: int = 10 * 1024 * 1024  # 10MB

    # ========== 数据库配置 ==========
    database_url: str = "postgresql+asyncpg://localhost:5432/kiki"
    database_pool_size: int = 20
    database_echo: bool = False

    # ========== LLM 配置 ==========
    llm_provider: Literal["openai", "anthropic", "ollama", "dashscope", "deepseek", "mock"] = (
        "openai"
    )
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    # DeepSeek 配置
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"

    # DashScope 专用配置
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 多模型路由配置
    llm_enable_multi_provider: bool = True  # 是否启用多提供商路由
    llm_default_priority: Literal["cost", "quality", "speed", "balanced"] = "balanced"

    # ========== 认证配置 ==========
    secret_key: str = "change-me-in-production-min-32-chars"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    jwt_algorithm: str = "HS256"

    # ========== 多租户配置 ==========
    # 租户 API Key 加密密钥（AES-256，32 字节）
    # 生成方法: openssl rand -base64 32
    tenant_aes_key: str = ""
    # 是否启用跨租户访问（需要用户有 can_access_all_tenants 权限）
    enable_cross_tenant: bool = False
    # 默认存储配额（字节），10GB
    default_storage_quota: int = 10 * 1024 * 1024 * 1024

    # ========== 可观测性配置 ==========
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # LangSmith
    langchain_api_key: str | None = None
    langchain_project: str = "kiki-agent"
    langchain_tracing_v2: bool = False

    # Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    # Prometheus
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"

    # ========== ELK 配置 ==========
    elk_enabled: bool = False
    elk_host: str = "localhost"
    elk_port: int = 5044
    elk_timeout: float = 5.0
    elk_max_retries: int = 3
    elk_batch_size: int = 10
    elk_batch_timeout: float = 5.0
    elk_fallback_enabled: bool = True

    # ========== 限流配置 ==========
    # 使用令牌桶算法的默认配置
    rate_limit_enabled: bool = True
    rate_limit_default_rate: float = 10.0  # 令牌/秒
    rate_limit_default_burst: int = 50  # 突发容量
    rate_limit_per_ip_enabled: bool = True
    rate_limit_per_user_enabled: bool = True

    # ========== 审计日志配置 ==========
    # 是否启用审计日志
    audit_enabled: bool = True
    # 审计日志数据库持久化
    audit_db_enabled: bool = False
    # 审计日志文件持久化
    audit_file_enabled: bool = True
    # 审计日志保留天数
    audit_retention_days: int = 90
    # 审计日志目录
    audit_log_dir: str = "./logs/audit"

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

    # 本地存储配置
    local_storage_base_dir: str = "./data/files"

    # ========== 会话上下文配置 ==========
    context_storage_type: Literal["memory", "redis"] = "memory"
    context_ttl_hours: int = 24
    context_max_messages: int = 100
    context_max_tokens: int = 128_000

    # ========== Agent 配置 ==========
    # 消息滑动窗口大小（超过此数量自动修剪，保留最近的消息）
    agent_max_messages: int = 100
    # Agent 最大迭代次数（防止无限循环）
    agent_max_iterations: int = 50
    # Agent 默认重试次数
    agent_max_retries: int = 3
    # Agent 重试初始间隔（秒）
    agent_retry_initial_interval: float = 0.5
    # Agent 重试退避因子
    agent_retry_backoff_factor: float = 2.0
    # Agent 重试最大间隔（秒）
    agent_retry_max_interval: float = 60.0

    # ========== RAG / Embedding 配置 ==========
    # Embedding 提供商: openai, dashscope, voyage, ollama
    embedding_provider: Literal["openai", "dashscope", "voyage", "ollama"] = "openai"
    # Embedding 模型名称
    embedding_model: str = "text-embedding-3-small"
    # 向量维度（text-embedding-v4 支持 64/128/256/512/768/1024/1536/2048）
    embedding_dimensions: int = 1024
    # 向量存储类型: qdrant, pgvector, pinecone, chroma, memory
    vector_store_type: Literal["qdrant", "pgvector", "pinecone", "chroma", "memory"] = "memory"

    # RAG 分块配置
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7

    # Qdrant 配置
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = "./data/qdrant"
    qdrant_port: int = 6333

    # Pinecone 配置
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "kiki"
    pinecone_region: str = "us-east-1"

    # Chroma 配置
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
