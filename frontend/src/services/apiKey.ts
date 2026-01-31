/**
 * Kiki Agent Framework - API Key 管理服务
 */

import api, { request } from './api';
import type {
  ApiKey,
  ApiKeysQuery,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  UpdateApiKeyRequest,
  VerifyApiKeyResponse,
  ApiKeyStatsResponse,
} from '@/types/apiKey';

/**
 * 获取 API Key 列表
 */
export async function getApiKeys(query?: ApiKeysQuery): Promise<ApiKey[]> {
  return request(() =>
    api.get<ApiKey[]>('/api-keys', { params: query })
  );
}

/**
 * 获取 API Key 详情
 */
export async function getApiKey(apiKeyId: number): Promise<ApiKey> {
  return request(() =>
    api.get<ApiKey>(`/api-keys/${apiKeyId}`)
  );
}

/**
 * 创建 API Key
 */
export async function createApiKey(data: CreateApiKeyRequest): Promise<CreateApiKeyResponse> {
  return request(() =>
    api.post<CreateApiKeyResponse>('/api-keys', data)
  );
}

/**
 * 更新 API Key
 */
export async function updateApiKey(apiKeyId: number, data: UpdateApiKeyRequest): Promise<ApiKey> {
  return request(() =>
    api.patch<ApiKey>(`/api-keys/${apiKeyId}`, data)
  );
}

/**
 * 删除 API Key
 */
export async function deleteApiKey(apiKeyId: number): Promise<void> {
  return request(() =>
    api.delete(`/api-keys/${apiKeyId}`)
  );
}

/**
 * 吊销 API Key
 */
export async function revokeApiKey(apiKeyId: number): Promise<void> {
  return request(() =>
    api.post(`/api-keys/${apiKeyId}/revoke`)
  );
}

/**
 * 验证 API Key
 */
export async function verifyApiKey(apiKey: string): Promise<VerifyApiKeyResponse> {
  return request(() =>
    api.post<VerifyApiKeyResponse>('/api-keys/verify', null, {
      headers: {
        'X-API-Key': apiKey,
      },
    })
  );
}

/**
 * 获取 API Key 统计
 */
export async function getApiKeyStats(): Promise<ApiKeyStatsResponse> {
  return request(() =>
    api.get<ApiKeyStatsResponse>('/api-keys/stats/me')
  );
}

/**
 * 创建 MCP 专用 API Key
 */
export async function createMcpApiKey(
  name: string,
  expiresInDays?: number
): Promise<CreateApiKeyResponse> {
  return request(() =>
    api.post<CreateApiKeyResponse>('/api-keys/mcp/create', null, {
      params: {
        name,
        ...(expiresInDays !== undefined && { expires_in_days: expiresInDays }),
      },
    })
  );
}

export default {
  getApiKeys,
  getApiKey,
  createApiKey,
  updateApiKey,
  deleteApiKey,
  revokeApiKey,
  verifyApiKey,
  getApiKeyStats,
  createMcpApiKey,
};
