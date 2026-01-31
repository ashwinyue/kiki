/**
 * Kiki Agent Framework - MCP 服务管理
 */

import api, { request } from './api';
import type {
  McpService,
  McpServicesQuery,
  CreateMcpServiceRequest,
  UpdateMcpServiceRequest,
} from '@/types/mcp';

/**
 * 获取 MCP 服务列表
 */
export async function getMcpServices(query?: McpServicesQuery): Promise<McpService[]> {
  return request(() =>
    api.get<McpService[]>('/mcp-services', { params: query })
  );
}

/**
 * 获取 MCP 服务详情
 */
export async function getMcpService(serviceId: string): Promise<McpService> {
  return request(() =>
    api.get<McpService>(`/mcp-services/${serviceId}`)
  );
}

/**
 * 创建 MCP 服务
 */
export async function createMcpService(data: CreateMcpServiceRequest): Promise<McpService> {
  return request(() =>
    api.post<McpService>('/mcp-services', data)
  );
}

/**
 * 更新 MCP 服务
 */
export async function updateMcpService(
  serviceId: string,
  data: UpdateMcpServiceRequest
): Promise<McpService> {
  return request(() =>
    api.patch<McpService>(`/mcp-services/${serviceId}`, data)
  );
}

/**
 * 删除 MCP 服务
 */
export async function deleteMcpService(serviceId: string): Promise<void> {
  return request(() =>
    api.delete(`/mcp-services/${serviceId}`)
  );
}

export default {
  getMcpServices,
  getMcpService,
  createMcpService,
  updateMcpService,
  deleteMcpService,
};
