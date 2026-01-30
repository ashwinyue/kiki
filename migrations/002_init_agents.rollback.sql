-- ============================================
-- Kiki Agent 模块回滚脚本
-- 警告：此脚本将删除 Agent 相关的所有表结构和数据
-- ============================================

-- 删除触发器
DROP TRIGGER IF EXISTS update_tools_updated_at ON tools;
DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;
DROP TRIGGER IF EXISTS update_prompt_templates_updated_at ON prompt_templates;
DROP TRIGGER IF EXISTS update_agent_executions_updated_at ON agent_executions;

-- 删除表（注意顺序：先删除有外键依赖的表）
DROP TABLE IF EXISTS agent_executions CASCADE;
DROP TABLE IF EXISTS agent_tools CASCADE;
DROP TABLE IF EXISTS prompt_templates CASCADE;
DROP TABLE IF EXISTS agents CASCADE;
DROP TABLE IF EXISTS tools CASCADE;

-- 删除枚举类型
DROP TYPE IF EXISTS agent_status CASCADE;
DROP TYPE IF EXISTS agent_type CASCADE;

-- 删除索引（会随表自动删除）
DROP INDEX IF EXISTS idx_tools_type;
DROP INDEX IF EXISTS idx_tools_is_active;
DROP INDEX IF EXISTS idx_agents_type;
DROP INDEX IF EXISTS idx_agents_status;
DROP INDEX IF EXISTS idx_agents_model_name;
DROP INDEX IF EXISTS idx_agent_tools_tool_id;
DROP INDEX IF EXISTS idx_agent_tools_enabled;
DROP INDEX IF EXISTS idx_prompt_templates_category;
DROP INDEX IF EXISTS idx_prompt_templates_is_active;
DROP INDEX IF EXISTS idx_agent_executions_session_id;
DROP INDEX IF EXISTS idx_agent_executions_agent_id;
DROP INDEX IF EXISTS idx_agent_executions_status;
DROP INDEX IF EXISTS idx_agent_executions_created_at;
