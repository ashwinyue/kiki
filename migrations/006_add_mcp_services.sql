-- ============================================
-- Kiki MCP 服务表
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

CREATE TABLE IF NOT EXISTS mcp_services (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    transport_type VARCHAR(50) NOT NULL DEFAULT 'stdio',
    url VARCHAR(512),
    headers JSONB,
    auth_config JSONB,
    advanced_config JSONB,
    stdio_config JSONB,
    env_vars JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_mcp_services_tenant_id ON mcp_services(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mcp_services_enabled ON mcp_services(enabled);
CREATE INDEX IF NOT EXISTS idx_mcp_services_deleted_at ON mcp_services(deleted_at);

COMMENT ON TABLE mcp_services IS 'MCP 服务配置表';

DROP TRIGGER IF EXISTS update_mcp_services_updated_at ON mcp_services;
CREATE TRIGGER update_mcp_services_updated_at
    BEFORE UPDATE ON mcp_services
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
