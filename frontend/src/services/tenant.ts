/**
 * Kiki Agent Framework - 租户管理服务
 */

import api, { request } from './api';
import type {
  Tenant,
  TenantsQuery,
  CreateTenantRequest,
  UpdateTenantRequest,
  TenantConfigResponse,
  UpdateTenantConfigRequest,
  RotateApiKeyResponse,
} from '@/types/tenant';

/**
 * 获取租户列表
 */
export async function getTenants(query?: TenantsQuery): Promise<Tenant[]> {
  return request(() =>
    api.get<Tenant[]>('/tenants', { params: query })
  );
}

/**
 * 获取租户详情
 */
export async function getTenant(tenantId: string): Promise<Tenant> {
  return request(() =>
    api.get<Tenant>(`/tenants/${tenantId}`)
  );
}

/**
 * 创建租户
 */
export async function createTenant(data: CreateTenantRequest): Promise<Tenant> {
  return request(() =>
    api.post<Tenant>('/tenants', data)
  );
}

/**
 * 更新租户
 */
export async function updateTenant(tenantId: string, data: UpdateTenantRequest): Promise<Tenant> {
  return request(() =>
    api.patch<Tenant>(`/tenants/${tenantId}`, data)
  );
}

/**
 * 删除租户
 */
export async function deleteTenant(tenantId: string): Promise<void> {
  return request(() =>
    api.delete(`/tenants/${tenantId}`)
  );
}

/**
 * 轮换租户 API Key
 */
export async function rotateTenantApiKey(tenantId: string): Promise<RotateApiKeyResponse> {
  return request(() =>
    api.post<RotateApiKeyResponse>(`/tenants/${tenantId}/rotate-api-key`)
  );
}

/**
 * 获取租户配置
 */
export async function getTenantConfig(): Promise<TenantConfigResponse> {
  return request(() =>
    api.get<TenantConfigResponse>('/tenants/me/config')
  );
}

/**
 * 更新租户配置
 */
export async function updateTenantConfig(data: UpdateTenantConfigRequest): Promise<void> {
  return request(() =>
    api.patch<void>('/tenants/me/config', data)
  );
}

export default {
  getTenants,
  getTenant,
  createTenant,
  updateTenant,
  deleteTenant,
  rotateTenantApiKey,
  getTenantConfig,
  updateTenantConfig,
};
