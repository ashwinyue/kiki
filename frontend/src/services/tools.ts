/**
 * Kiki Agent Framework - 工具管理服务
 */

import api, { request } from './api';
import type { Tool, ToolsResponse, ToolDetailResponse } from '@/types/tools';

/**
 * 获取工具列表
 */
export async function getTools(): Promise<ToolsResponse> {
  return request(() =>
    api.get<ToolsResponse>('/tools')
  );
}

/**
 * 获取工具详情
 */
export async function getTool(toolName: string): Promise<ToolDetailResponse> {
  return request(() =>
    api.get<ToolDetailResponse>(`/tools/${toolName}`)
  );
}

export default {
  getTools,
  getTool,
};
