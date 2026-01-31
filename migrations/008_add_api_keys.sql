-- ============================================
-- Kiki API Key 表
-- 数据库: PostgreSQL
-- 依赖: 005_add_tenant_and_session_message_fields.sql
-- ============================================

-- 确保 updated_at 触发器函数存在
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- API Key 状态枚举
CREATE TYPE api_key_status AS ENUM (
    'active',
    'inactive',
    'revoked',
    'expired'
);

-- API Key 类型枚举
CREATE TYPE api_key_type AS ENUM (
    'personal',
    'service',
    'mcp',
    'webhook'
);

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,
    hashed_key VARCHAR(255) NOT NULL,
    key_type api_key_type NOT NULL DEFAULT 'personal',
    status api_key_status NOT NULL DEFAULT 'active',
    user_id INTEGER NOT NULL,
    scopes TEXT[] DEFAULT '{}',
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    rate_limit INTEGER,
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_status ON api_keys(user_id, status);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_type ON api_keys(key_type);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);

-- 注释
COMMENT ON TABLE api_keys IS 'API Key 认证表';
COMMENT ON COLUMN api_keys.key_prefix IS 'API Key 前缀，用于显示和快速查找';
COMMENT ON COLUMN api_keys.hashed_key IS '加密后的完整 API Key';
COMMENT ON COLUMN api_keys.key_type IS 'API Key 类型：personal(个人), service(服务间), mcp(MCP专用), webhook(Webhook)';
COMMENT ON COLUMN api_keys.status IS 'API Key 状态：active(活跃), inactive(非活跃), revoked(已吊销), expired(已过期)';
COMMENT ON COLUMN api_keys.scopes IS '权限范围数组';
COMMENT ON COLUMN api_keys.rate_limit IS '速率限制（每分钟请求数）';

-- 触发器
DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
CREATE TRIGGER update_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
