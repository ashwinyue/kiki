-- ============================================
-- Kiki 移除 tools 与 agent_tools 表
-- 数据库: PostgreSQL
-- 依赖: 002_init_agents.sql
-- ============================================

-- 删除触发器（如果存在）
DROP TRIGGER IF EXISTS update_tools_updated_at ON tools;

-- 删除表
DROP TABLE IF EXISTS agent_tools;
DROP TABLE IF EXISTS tools;
