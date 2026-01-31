/**
 * Kiki Agent Framework - 租户相关类型
 */

/**
 * 租户信息
 */
export interface Tenant {
  id: string;
  name: string;
  status: 'active' | 'suspended' | 'pending';
  config: Record<string, unknown>;
  api_key_prefix?: string;
  created_at: string;
  updated_at: string;
}

/**
 * 创建租户请求
 */
export interface CreateTenantRequest {
  name: string;
  config?: Record<string, unknown>;
}

/**
 * 更新租户请求
 */
export interface UpdateTenantRequest {
  name?: string;
  status?: 'active' | 'suspended' | 'pending';
  config?: Record<string, unknown>;
}

/**
 * 租户列表查询参数
 */
export interface TenantsQuery {
  status?: 'active' | 'suspended' | 'pending';
  keyword?: string;
  page?: number;
  size?: number;
}

/**
 * 租户配置响应
 */
export interface TenantConfigResponse {
  tenant_id: string;
  config: Record<string, unknown>;
}

/**
 * 更新租户配置请求
 */
export interface UpdateTenantConfigRequest {
  config: Record<string, unknown>;
}

/**
 * 轮换 API Key 响应
 */
export interface RotateApiKeyResponse {
  api_key: string;
  message: string;
}
