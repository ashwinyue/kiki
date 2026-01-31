-- ============================================
-- Kiki 线程与执行记录结构更新
-- 数据库: PostgreSQL
-- 依赖: 001_init_schema.sql, 002_init_agents.sql
-- ============================================

-- ============================================
-- 1. threads 增加 session_id（业务关联，无物理外键）
-- ============================================
ALTER TABLE threads
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(255);

CREATE INDEX IF NOT EXISTS idx_threads_session_id ON threads(session_id);

COMMENT ON COLUMN threads.session_id IS '关联会话ID（业务关联，无外键）';


-- ============================================
-- 2. agent_executions: session_id -> thread_id
-- ============================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'agent_executions'
          AND column_name = 'session_id'
    ) THEN
        ALTER TABLE agent_executions RENAME COLUMN session_id TO thread_id;
    END IF;
END $$;

DROP INDEX IF EXISTS idx_agent_executions_session_id;
CREATE INDEX IF NOT EXISTS idx_agent_executions_thread_id ON agent_executions(thread_id);

COMMENT ON COLUMN agent_executions.thread_id IS '关联的线程 ID';
