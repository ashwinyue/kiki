-- ============================================
-- Kiki Agent Framework - 流式对话消息持久化
-- 版本: 002
-- 日期: 2025-02-03
-- 说明: 创建 chat_streams 表用于存储流式对话的完整消息历史
-- ============================================

-- ============================================
-- 1. 创建函数：自动更新 updated_at 字段
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';


-- ============================================
-- 2. 创建表：chat_streams
-- ============================================

CREATE TABLE IF NOT EXISTS chat_streams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    messages JSONB NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 注释
COMMENT ON TABLE chat_streams IS '流式对话消息存储表';
COMMENT ON COLUMN chat_streams.id IS '主键（UUID）';
COMMENT ON COLUMN chat_streams.thread_id IS '会话线程 ID（唯一）';
COMMENT ON COLUMN chat_streams.messages IS '消息列表（JSONB 格式）';
COMMENT ON COLUMN chat_streams.ts IS '消息时间戳';
COMMENT ON COLUMN chat_streams.created_at IS '创建时间';
COMMENT ON COLUMN chat_streams.updated_at IS '更新时间';


-- ============================================
-- 3. 创建索引
-- ============================================

-- thread_id 索引（用于快速查询）
CREATE INDEX IF NOT EXISTS idx_chat_streams_thread_id ON chat_streams(thread_id);

-- ts 索引（用于时间范围查询）
CREATE INDEX IF NOT EXISTS idx_chat_streams_ts ON chat_streams(ts);

-- created_at 索引（用于按创建时间排序）
CREATE INDEX IF NOT EXISTS idx_chat_streams_created_at ON chat_streams(created_at);

-- messages GIN 索引（用于 JSONB 内容查询）
CREATE INDEX IF NOT EXISTS idx_chat_streams_messages ON chat_streams USING GIN (messages);


-- ============================================
-- 4. 创建触发器：自动更新 updated_at
-- ============================================

DROP TRIGGER IF EXISTS update_chat_streams_updated_at ON chat_streams;
CREATE TRIGGER update_chat_streams_updated_at
    BEFORE UPDATE ON chat_streams
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================
-- 5. 创建视图：最近对话线程
-- ============================================

CREATE OR REPLACE VIEW v_recent_threads AS
SELECT
    thread_id,
    jsonb_array_length(messages) as message_count,
    ts,
    created_at,
    updated_at
FROM chat_streams
ORDER BY ts DESC;

COMMENT ON VIEW v_recent_threads IS '最近对话线程视图';


-- ============================================
-- 6. 创建函数：清理过期对话历史
-- ============================================

CREATE OR REPLACE FUNCTION cleanup_old_chat_streams(
    days_to_keep INTEGER DEFAULT 30
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM chat_streams
    WHERE ts < NOW() - INTERVAL '1 day' * days_to_keep;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_chat_streams IS '清理指定天数之前的对话历史';
