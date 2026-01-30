-- ============================================
-- Kiki 项目数据库回滚脚本
-- 警告：此脚本将删除所有表结构和数据
-- ============================================

-- 删除触发器
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_chatsessions_updated_at ON chatsessions;
DROP TRIGGER IF EXISTS update_threads_updated_at ON threads;

-- 删除触发器函数
DROP FUNCTION IF EXISTS update_updated_at_column();

-- 删除表（注意顺序：先删除有外键依赖的表）
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS chatsessions CASCADE;
DROP TABLE IF EXISTS threads CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 删除索引（会随表自动删除，但显式声明更清晰）
DROP INDEX IF EXISTS idx_users_email;
DROP INDEX IF EXISTS idx_users_is_active;
DROP INDEX IF EXISTS idx_chatsessions_user_id;
DROP INDEX IF EXISTS idx_chatsessions_created_at;
DROP INDEX IF EXISTS idx_threads_user_id;
DROP INDEX IF EXISTS idx_threads_status;
DROP INDEX IF EXISTS idx_threads_created_at;
DROP INDEX IF EXISTS idx_messages_session_id;
DROP INDEX IF EXISTS idx_messages_role;
DROP INDEX IF EXISTS idx_messages_created_at;
