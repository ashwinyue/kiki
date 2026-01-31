-- ============================================
-- API Key 表回滚
-- ============================================

DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
DROP INDEX IF EXISTS idx_api_keys_status;
DROP INDEX IF EXISTS idx_api_keys_expires_at;
DROP INDEX IF EXISTS idx_api_keys_key_prefix;
DROP INDEX IF EXISTS idx_api_keys_key_type;
DROP INDEX IF EXISTS idx_api_keys_user_status;
DROP INDEX IF EXISTS idx_api_keys_user_id;
DROP TABLE IF EXISTS api_keys;
DROP TYPE IF EXISTS api_key_type;
DROP TYPE IF EXISTS api_key_status;
