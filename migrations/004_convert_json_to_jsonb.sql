-- ============================================
-- Kiki JSON -> JSONB 列类型转换
-- 数据库: PostgreSQL
-- 依赖: 001_init_schema.sql, 002_init_agents.sql, 003_update_threads_and_executions.sql
-- ============================================

ALTER TABLE IF EXISTS chatsessions
    ALTER COLUMN extra_data TYPE JSONB USING extra_data::JSONB;

ALTER TABLE IF EXISTS messages
    ALTER COLUMN tool_calls TYPE JSONB USING tool_calls::JSONB,
    ALTER COLUMN extra_data TYPE JSONB USING extra_data::JSONB;

ALTER TABLE IF EXISTS memories
    ALTER COLUMN value TYPE JSONB USING value::JSONB;

ALTER TABLE IF EXISTS tools
    ALTER COLUMN config TYPE JSONB USING config::JSONB;

ALTER TABLE IF EXISTS agents
    ALTER COLUMN config TYPE JSONB USING config::JSONB;

ALTER TABLE IF EXISTS agent_tools
    ALTER COLUMN config TYPE JSONB USING config::JSONB;

ALTER TABLE IF EXISTS prompt_templates
    ALTER COLUMN variables TYPE JSONB USING variables::JSONB;

ALTER TABLE IF EXISTS agent_executions
    ALTER COLUMN input_data TYPE JSONB USING input_data::JSONB,
    ALTER COLUMN output_data TYPE JSONB USING output_data::JSONB,
    ALTER COLUMN extra_data TYPE JSONB USING extra_data::JSONB;
