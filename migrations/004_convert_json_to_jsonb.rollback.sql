-- ============================================
-- Kiki JSONB -> JSON 列类型回滚
-- ============================================

ALTER TABLE IF EXISTS agent_executions
    ALTER COLUMN input_data TYPE JSON USING input_data::JSON,
    ALTER COLUMN output_data TYPE JSON USING output_data::JSON,
    ALTER COLUMN extra_data TYPE JSON USING extra_data::JSON;

ALTER TABLE IF EXISTS prompt_templates
    ALTER COLUMN variables TYPE JSON USING variables::JSON;

ALTER TABLE IF EXISTS agent_tools
    ALTER COLUMN config TYPE JSON USING config::JSON;

ALTER TABLE IF EXISTS agents
    ALTER COLUMN config TYPE JSON USING config::JSON;

ALTER TABLE IF EXISTS tools
    ALTER COLUMN config TYPE JSON USING config::JSON;

ALTER TABLE IF EXISTS memories
    ALTER COLUMN value TYPE JSON USING value::JSON;

ALTER TABLE IF EXISTS messages
    ALTER COLUMN tool_calls TYPE JSON USING tool_calls::JSON,
    ALTER COLUMN extra_data TYPE JSON USING extra_data::JSON;

ALTER TABLE IF EXISTS chatsessions
    ALTER COLUMN extra_data TYPE JSON USING extra_data::JSON;
