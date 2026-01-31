-- ============================================
-- Kiki 线程与执行记录结构更新 回滚脚本
-- ============================================

-- 1. agent_executions: thread_id -> session_id
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'agent_executions'
          AND column_name = 'thread_id'
    ) THEN
        ALTER TABLE agent_executions RENAME COLUMN thread_id TO session_id;
    END IF;
END $$;

DROP INDEX IF EXISTS idx_agent_executions_thread_id;
CREATE INDEX IF NOT EXISTS idx_agent_executions_session_id ON agent_executions(session_id);

COMMENT ON COLUMN agent_executions.session_id IS '关联的会话 ID';

-- 2. threads 删除 session_id
DROP INDEX IF EXISTS idx_threads_session_id;
ALTER TABLE threads DROP COLUMN IF EXISTS session_id;
