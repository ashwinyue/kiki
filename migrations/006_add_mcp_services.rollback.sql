-- ============================================
-- Kiki MCP 服务表 回滚
-- ============================================

DROP TRIGGER IF EXISTS update_mcp_services_updated_at ON mcp_services;
DROP INDEX IF EXISTS idx_mcp_services_tenant_id;
DROP INDEX IF EXISTS idx_mcp_services_enabled;
DROP INDEX IF EXISTS idx_mcp_services_deleted_at;
DROP TABLE IF EXISTS mcp_services;
