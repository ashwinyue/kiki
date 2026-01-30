-- ============================================
-- Kiki 项目数据库初始化脚本
-- 数据库: PostgreSQL
-- 编码: UTF-8
-- ============================================

-- ============================================
-- 1. 用户表 (users)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- 注释
COMMENT ON TABLE users IS '用户表';
COMMENT ON COLUMN users.id IS '用户ID（主键）';
COMMENT ON COLUMN users.email IS '用户邮箱（唯一）';
COMMENT ON COLUMN users.full_name IS '用户全名';
COMMENT ON COLUMN users.hashed_password IS '加密后的密码（bcrypt）';
COMMENT ON COLUMN users.is_active IS '是否激活';
COMMENT ON COLUMN users.is_superuser IS '是否为超级管理员';
COMMENT ON COLUMN users.created_at IS '创建时间';
COMMENT ON COLUMN users.updated_at IS '更新时间';


-- ============================================
-- 2. 聊天会话表 (chatsessions)
-- ============================================
CREATE TABLE IF NOT EXISTS chatsessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(500) NOT NULL DEFAULT '',
    extra_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_chatsessions_user_id ON chatsessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chatsessions_created_at ON chatsessions(created_at DESC);

-- 注释
COMMENT ON TABLE chatsessions IS '聊天会话表';
COMMENT ON COLUMN chatsessions.id IS '会话ID（UUID，主键）';
COMMENT ON COLUMN chatsessions.user_id IS '关联用户ID（外键）';
COMMENT ON COLUMN chatsessions.name IS '会话名称';
COMMENT ON COLUMN chatsessions.extra_data IS '扩展数据（JSONB）';
COMMENT ON COLUMN chatsessions.created_at IS '创建时间';
COMMENT ON COLUMN chatsessions.updated_at IS '更新时间';


-- ============================================
-- 3. 线程表 (threads)
-- 用于 LangGraph 状态持久化
-- ============================================
CREATE TABLE IF NOT EXISTS threads (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(500) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(created_at DESC);

-- 注释
COMMENT ON TABLE threads IS 'LangGraph 线程表（用于状态持久化）';
COMMENT ON COLUMN threads.id IS '线程ID（主键）';
COMMENT ON COLUMN threads.user_id IS '关联用户ID（外键）';
COMMENT ON COLUMN threads.name IS '线程名称';
COMMENT ON COLUMN threads.status IS '状态：active/archived/deleted';
COMMENT ON COLUMN threads.created_at IS '创建时间';
COMMENT ON COLUMN threads.updated_at IS '更新时间';


-- ============================================
-- 4. 消息表 (messages)
-- ============================================
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES chatsessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls JSONB,
    extra_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);

-- 注释
COMMENT ON TABLE messages IS '聊天消息表';
COMMENT ON COLUMN messages.id IS '消息ID（主键）';
COMMENT ON COLUMN messages.session_id IS '关联会话ID（外键）';
COMMENT ON COLUMN messages.role IS '角色：user/assistant/system/tool';
COMMENT ON COLUMN messages.content IS '消息内容';
COMMENT ON COLUMN messages.tool_calls IS '工具调用信息（JSONB）';
COMMENT ON COLUMN messages.extra_data IS '扩展数据（JSONB）';
COMMENT ON COLUMN messages.created_at IS '创建时间';


-- ============================================
-- 5. 自动更新 updated_at 触发器函数
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为 users 表创建触发器
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 chatsessions 表创建触发器
DROP TRIGGER IF EXISTS update_chatsessions_updated_at ON chatsessions;
CREATE TRIGGER update_chatsessions_updated_at
    BEFORE UPDATE ON chatsessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 threads 表创建触发器
DROP TRIGGER IF EXISTS update_threads_updated_at ON threads;
CREATE TRIGGER update_threads_updated_at
    BEFORE UPDATE ON threads
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 memories 表创建触发器
DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================
-- 7. 清理过期记忆的函数
-- ============================================
CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM memories
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_expired_memories IS '清理过期的记忆记录，返回删除数量';


-- ============================================
-- 8. 初始化数据（可选）
-- ============================================

-- 创建默认超级管理员（密码：admin123，需要手动修改）
-- 密码哈希（bcrypt）: $2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU9iKomuU6kG
-- 默认密码: admin123
-- INSERT INTO users (email, full_name, hashed_password, is_active, is_superuser)
-- VALUES ('admin@kiki.local', 'Admin User', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU9iKomuU6kG', TRUE, TRUE)
-- ON CONFLICT (email) DO NOTHING;


-- ============================================
-- 5. 长期记忆表 (memories)
-- 用于 LangGraph Store 跨会话持久化
-- ============================================
CREATE TABLE IF NOT EXISTS memories (
    namespace VARCHAR(255) NOT NULL,
    key VARCHAR(500) NOT NULL,
    value JSONB,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, key)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at) WHERE expires_at IS NOT NULL;

-- 注释
COMMENT ON TABLE memories IS '长期记忆表（LangGraph Store）';
COMMENT ON COLUMN memories.namespace IS '命名空间：user:123, session:abc, agent:xyz';
COMMENT ON COLUMN memories.key IS '键名：preferences, summary, config 等';
COMMENT ON COLUMN memories.value IS 'JSONB 存储的值';
COMMENT ON COLUMN memories.expires_at IS '过期时间（可选）';
COMMENT ON COLUMN memories.created_at IS '创建时间';
COMMENT ON COLUMN memories.updated_at IS '更新时间';


-- ============================================
-- 6. 自动更新 updated_at 触发器函数
-- ============================================
