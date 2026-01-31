-- ============================================
-- Kiki 多租户与会话/消息增强
-- 数据库: PostgreSQL
-- 依赖: 001_init_schema.sql, 002_init_agents.sql, 003_update_threads_and_executions.sql, 004_convert_json_to_jsonb.sql
-- ============================================

-- 确保 updated_at 触发器函数存在
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 1. 租户表
-- ============================================
CREATE TABLE IF NOT EXISTS tenants (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    api_key VARCHAR(64) NOT NULL UNIQUE,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_tenants_api_key ON tenants(api_key);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);

COMMENT ON TABLE tenants IS '租户表';
COMMENT ON COLUMN tenants.api_key IS '租户 API Key（唯一）';

DROP TRIGGER IF EXISTS update_tenants_updated_at ON tenants;
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 2. users 增强
-- ============================================
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS tenant_id INTEGER,
    ADD COLUMN IF NOT EXISTS can_access_all_tenants BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);

-- ============================================
-- 3. chatsessions 增强
-- ============================================
ALTER TABLE chatsessions
    ADD COLUMN IF NOT EXISTS tenant_id INTEGER,
    ADD COLUMN IF NOT EXISTS agent_id INTEGER,
    ADD COLUMN IF NOT EXISTS agent_config JSONB,
    ADD COLUMN IF NOT EXISTS context_config JSONB,
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

CREATE INDEX IF NOT EXISTS idx_chatsessions_tenant_id ON chatsessions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_chatsessions_agent_id ON chatsessions(agent_id);

-- ============================================
-- 4. threads 增强
-- ============================================
ALTER TABLE threads
    ADD COLUMN IF NOT EXISTS tenant_id INTEGER,
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

CREATE INDEX IF NOT EXISTS idx_threads_tenant_id ON threads(tenant_id);

-- ============================================
-- 5. messages 增强
-- ============================================
ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS request_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS knowledge_references JSONB,
    ADD COLUMN IF NOT EXISTS agent_steps JSONB,
    ADD COLUMN IF NOT EXISTS is_completed BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

CREATE INDEX IF NOT EXISTS idx_messages_request_id ON messages(request_id);
CREATE INDEX IF NOT EXISTS idx_messages_is_completed ON messages(is_completed);

DROP TRIGGER IF EXISTS update_messages_updated_at ON messages;
CREATE TRIGGER update_messages_updated_at
    BEFORE UPDATE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 6. agents/tools/agent_tools 增强
-- ============================================
ALTER TABLE agents
    ADD COLUMN IF NOT EXISTS tenant_id INTEGER,
    ADD COLUMN IF NOT EXISTS created_by_user_id INTEGER,
    ADD COLUMN IF NOT EXISTS is_builtin BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_agents_tenant_id ON agents(tenant_id);

ALTER TABLE tools
    ADD COLUMN IF NOT EXISTS tenant_id INTEGER,
    ADD COLUMN IF NOT EXISTS created_by_user_id INTEGER,
    ADD COLUMN IF NOT EXISTS is_builtin BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_tools_tenant_id ON tools(tenant_id);

ALTER TABLE agent_tools
    ADD COLUMN IF NOT EXISTS tenant_id INTEGER,
    ADD COLUMN IF NOT EXISTS user_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_agent_tools_tenant_id ON agent_tools(tenant_id);
