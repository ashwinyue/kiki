-- ============================================
-- Kiki 多租户与会话/消息增强 回滚
-- ============================================

-- agent_tools
DROP INDEX IF EXISTS idx_agent_tools_tenant_id;
ALTER TABLE agent_tools
    DROP COLUMN IF EXISTS tenant_id,
    DROP COLUMN IF EXISTS user_id;

-- tools
DROP INDEX IF EXISTS idx_tools_tenant_id;
ALTER TABLE tools
    DROP COLUMN IF EXISTS tenant_id,
    DROP COLUMN IF EXISTS created_by_user_id,
    DROP COLUMN IF EXISTS is_builtin;

-- agents
DROP INDEX IF EXISTS idx_agents_tenant_id;
ALTER TABLE agents
    DROP COLUMN IF EXISTS tenant_id,
    DROP COLUMN IF EXISTS created_by_user_id,
    DROP COLUMN IF EXISTS is_builtin;

-- messages
DROP INDEX IF EXISTS idx_messages_request_id;
DROP INDEX IF EXISTS idx_messages_is_completed;
DROP TRIGGER IF EXISTS update_messages_updated_at ON messages;
ALTER TABLE messages
    DROP COLUMN IF EXISTS request_id,
    DROP COLUMN IF EXISTS knowledge_references,
    DROP COLUMN IF EXISTS agent_steps,
    DROP COLUMN IF EXISTS is_completed,
    DROP COLUMN IF EXISTS updated_at,
    DROP COLUMN IF EXISTS deleted_at;

-- threads
DROP INDEX IF EXISTS idx_threads_tenant_id;
ALTER TABLE threads
    DROP COLUMN IF EXISTS tenant_id,
    DROP COLUMN IF EXISTS deleted_at;

-- chatsessions
DROP INDEX IF EXISTS idx_chatsessions_tenant_id;
DROP INDEX IF EXISTS idx_chatsessions_agent_id;
ALTER TABLE chatsessions
    DROP COLUMN IF EXISTS tenant_id,
    DROP COLUMN IF EXISTS agent_id,
    DROP COLUMN IF EXISTS agent_config,
    DROP COLUMN IF EXISTS context_config,
    DROP COLUMN IF EXISTS deleted_at;

-- users
DROP INDEX IF EXISTS idx_users_tenant_id;
ALTER TABLE users
    DROP COLUMN IF EXISTS tenant_id,
    DROP COLUMN IF EXISTS can_access_all_tenants;

-- tenants
DROP TRIGGER IF EXISTS update_tenants_updated_at ON tenants;
DROP INDEX IF EXISTS idx_tenants_api_key;
DROP INDEX IF EXISTS idx_tenants_status;
DROP TABLE IF EXISTS tenants;
