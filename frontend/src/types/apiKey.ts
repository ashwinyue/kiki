/**
 * Kiki Agent Framework - API Key 相关类型
 */

/**
 * API Key 类型
 */
export type ApiKeyType = 'personal' | 'service' | 'mcp' | 'webhook';

/**
 * API Key 状态
 */
export type ApiKeyStatus = 'active' | 'revoked' | 'expired';

/**
 * API Key 信息（列表用，不含完整 key）
 */
export interface ApiKey {
  id: number;
  name: string;
  key_prefix: string;
  key_type: ApiKeyType;
  status: ApiKeyStatus;
  scopes: string[];
  expires_at: string | null;
  created_at: string;
  last_used: string | null;
  description?: string;
  rate_limit?: number;
}

/**
 * 创建 API Key 请求
 */
export interface CreateApiKeyRequest {
  name: string;
  key_type: ApiKeyType;
  scopes: string[];
  expires_in_days?: number;
  description?: string;
  rate_limit?: number;
}

/**
 * 创建 API Key 响应（含完整 key）
 */
export interface CreateApiKeyResponse extends ApiKey {
  key: string; // 完整的 API Key，仅返回一次
}

/**
 * 更新 API Key 请求
 */
export interface UpdateApiKeyRequest {
  name?: string;
  status?: ApiKeyStatus;
  scopes?: string[];
  expires_at?: string | null;
}

/**
 * API Key 验证响应
 */
export interface VerifyApiKeyResponse {
  valid: boolean;
  api_key_id?: number;
  user_id?: number;
  scopes?: string[];
  key_type?: string;
}

/**
 * API Key 统计响应
 */
export interface ApiKeyStatsResponse {
  user_id: number;
  total_keys: number;
  by_status: Record<string, number>;
  by_type: Record<string, number>;
}

/**
 * API Key 查询参数
 */
export interface ApiKeysQuery {
  key_type?: ApiKeyType;
  status?: ApiKeyStatus;
}
