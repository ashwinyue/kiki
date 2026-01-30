-- ============================================
-- Kiki Agent 模块数据库初始化脚本
-- 数据库: PostgreSQL
-- 编码: UTF-8
-- 依赖: 001_init_schema.sql
-- ============================================

-- ============================================
-- 1. Agent 类型枚举
-- ============================================
-- 使用 VARCHAR + CHECK 约束模拟枚举
CREATE DOMAIN agent_type AS VARCHAR(20)
    CHECK (VALUE IN ('single', 'router', 'supervisor', 'worker', 'handoff'));

CREATE DOMAIN agent_status AS VARCHAR(20)
    CHECK (VALUE IN ('active', 'disabled', 'deleted'));


-- ============================================
-- 2. 工具表 (tools)
-- ============================================
CREATE TABLE IF NOT EXISTS tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    type VARCHAR(50) NOT NULL,
    config JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_tools_type ON tools(type);
CREATE INDEX IF NOT EXISTS idx_tools_is_active ON tools(is_active);

-- 注释
COMMENT ON TABLE tools IS '工具定义表';
COMMENT ON COLUMN tools.name IS '工具名称（唯一）';
COMMENT ON COLUMN tools.description IS '工具描述';
COMMENT ON COLUMN tools.type IS '工具类型：function/python/mcp 等';
COMMENT ON COLUMN tools.config IS '工具配置（JSONB）';
COMMENT ON COLUMN tools.is_active IS '是否启用';


-- ============================================
-- 3. Agent 表 (agents)
-- ============================================
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description VARCHAR(500),
    agent_type agent_type NOT NULL DEFAULT 'single',
    status agent_status NOT NULL DEFAULT 'active',
    model_name VARCHAR(50) NOT NULL DEFAULT 'gpt-4o-mini',
    system_prompt TEXT NOT NULL DEFAULT '',
    temperature DECIMAL(3, 2) NOT NULL DEFAULT 0.7 CHECK (temperature >= 0.0 AND temperature <= 2.0),
    max_tokens INTEGER,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_model_name ON agents(model_name);

-- 注释
COMMENT ON TABLE agents IS 'Agent 配置表';
COMMENT ON COLUMN agents.name IS 'Agent 名称（唯一）';
COMMENT ON COLUMN agents.agent_type IS 'Agent 类型：single/router/supervisor/worker/handoff';
COMMENT ON COLUMN agents.status IS '状态：active/disabled/deleted';
COMMENT ON COLUMN agents.model_name IS '使用的模型名称';
COMMENT ON COLUMN agents.system_prompt IS '系统提示词';
COMMENT ON COLUMN agents.temperature IS '温度参数（0.0 - 2.0）';
COMMENT ON COLUMN agents.max_tokens IS '最大生成 tokens';
COMMENT ON COLUMN agents.config IS '额外配置（JSONB）';


-- ============================================
-- 4. Agent-Tool 关联表 (agent_tools)
-- ============================================
CREATE TABLE IF NOT EXISTS agent_tools (
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tool_id INTEGER NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, tool_id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_agent_tools_tool_id ON agent_tools(tool_id);
CREATE INDEX IF NOT EXISTS idx_agent_tools_enabled ON agent_tools(enabled);

-- 注释
COMMENT ON TABLE agent_tools IS 'Agent-Tool 多对多关联表';
COMMENT ON COLUMN agent_tools.enabled IS '是否启用该工具';
COMMENT ON COLUMN agent_tools.config IS '覆盖配置（JSONB）';


-- ============================================
-- 5. Prompt 模板表 (prompt_templates)
-- ============================================
CREATE TABLE IF NOT EXISTS prompt_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description VARCHAR(500),
    category VARCHAR(50),
    template TEXT NOT NULL,
    variables JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_prompt_templates_category ON prompt_templates(category);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_is_active ON prompt_templates(is_active);

-- 注释
COMMENT ON TABLE prompt_templates IS 'Prompt 模板表（支持 Jinja2）';
COMMENT ON COLUMN prompt_templates.name IS '模板名称（唯一）';
COMMENT ON COLUMN prompt_templates.category IS '分类';
COMMENT ON COLUMN prompt_templates.template IS '模板内容';
COMMENT ON COLUMN prompt_templates.variables IS '变量列表（JSONB 数组）';
COMMENT ON COLUMN prompt_templates.is_active IS '是否启用';


-- ============================================
-- 6. Agent 执行历史表 (agent_executions)
-- ============================================
CREATE TABLE IF NOT EXISTS agent_executions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    input_data JSONB,
    output_data JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'running',
    error_message TEXT,
    tokens_used INTEGER,
    duration_ms INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_agent_executions_session_id ON agent_executions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_agent_id ON agent_executions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON agent_executions(status);
CREATE INDEX IF NOT EXISTS idx_agent_executions_created_at ON agent_executions(created_at DESC);

-- 注释
COMMENT ON TABLE agent_executions IS 'Agent 执行历史表';
COMMENT ON COLUMN agent_executions.session_id IS '关联的会话 ID';
COMMENT ON COLUMN agent_executions.agent_id IS '执行的 Agent ID';
COMMENT ON COLUMN agent_executions.input_data IS '输入数据（JSONB）';
COMMENT ON COLUMN agent_executions.output_data IS '输出数据（JSONB）';
COMMENT ON COLUMN agent_executions.status IS '状态：running/success/error';
COMMENT ON COLUMN agent_executions.error_message IS '错误信息';
COMMENT ON COLUMN agent_executions.tokens_used IS '使用的 tokens 数量';
COMMENT ON COLUMN agent_executions.duration_ms IS '执行耗时（毫秒）';
COMMENT ON COLUMN agent_executions.metadata IS '元数据（JSONB）';


-- ============================================
-- 7. 自动更新 updated_at 触发器
-- ============================================
-- 为 tools 表创建触发器
DROP TRIGGER IF EXISTS update_tools_updated_at ON tools;
CREATE TRIGGER update_tools_updated_at
    BEFORE UPDATE ON tools
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 agents 表创建触发器
DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;
CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 prompt_templates 表创建触发器
DROP TRIGGER IF EXISTS update_prompt_templates_updated_at ON prompt_templates;
CREATE TRIGGER update_prompt_templates_updated_at
    BEFORE UPDATE ON prompt_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 agent_executions 表创建触发器
DROP TRIGGER IF EXISTS update_agent_executions_updated_at ON agent_executions;
CREATE TRIGGER update_agent_executions_updated_at
    BEFORE UPDATE ON agent_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================
-- 8. 初始化数据（可选）
-- ============================================

-- 创建默认工具示例
INSERT INTO tools (name, description, type, config, is_active)
VALUES
    ('web_search', '网络搜索工具', 'function', '{"endpoint": "/api/search"}', TRUE),
    ('calculator', '计算器工具', 'function', '{}', TRUE),
    ('datetime', '日期时间工具', 'function', '{}', TRUE)
ON CONFLICT (name) DO NOTHING;

-- 创建默认 Agent
INSERT INTO agents (name, description, agent_type, model_name, system_prompt, temperature)
VALUES
    (
        'default',
        '默认对话 Agent',
        'single',
        'gpt-4o-mini',
        '你是一个有用的 AI 助手，请用简洁友好的语气回答用户问题。',
        0.7
    ),
    (
        'router',
        '路由 Agent',
        'router',
        'gpt-4o-mini',
        '根据用户意图，将请求路由到合适的子 Agent。',
        0.3
    )
ON CONFLICT (name) DO NOTHING;

-- 为 default Agent 添加工具
INSERT INTO agent_tools (agent_id, tool_id, enabled)
SELECT a.id, t.id, TRUE
FROM agents a, tools t
WHERE a.name = 'default' AND t.name IN ('calculator', 'datetime')
ON CONFLICT (agent_id, tool_id) DO NOTHING;


-- ============================================
-- 9. 授权（根据实际数据库用户调整）
-- ============================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kiki_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO kiki_user;
