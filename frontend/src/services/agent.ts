/**
 * Kiki Agent Framework - Agent 管理服务
 */

import api, { request } from './api';
import type {
  Agent,
  AgentListQuery,
  AgentListResponse,
  AgentStatsResponse,
  CreateAgentRequest,
  UpdateAgentRequest,
  AgentExecution,
  ExecutionsQuery,
  AgentSystem,
} from '@/types/agent';

/**
 * 获取 Agent 列表
 */
export async function getAgents(query?: AgentListQuery): Promise<AgentListResponse> {
  return request(() =>
    api.get<AgentListResponse>('/agents/list', { params: query })
  );
}

/**
 * 获取 Agent 统计
 */
export async function getAgentStats(): Promise<AgentStatsResponse> {
  return request(() =>
    api.get<AgentStatsResponse>('/agents/stats')
  );
}

/**
 * 获取 Agent 详情
 */
export async function getAgent(agentId: string): Promise<Agent> {
  return request(() =>
    api.get<Agent>(`/agents/${agentId}`)
  );
}

/**
 * 创建 Agent
 */
export async function createAgent(data: CreateAgentRequest): Promise<Agent> {
  return request(() =>
    api.post<Agent>('/agents', data)
  );
}

/**
 * 更新 Agent
 */
export async function updateAgent(agentId: string, data: UpdateAgentRequest): Promise<Agent> {
  return request(() =>
    api.patch<Agent>(`/agents/${agentId}`, data)
  );
}

/**
 * 删除 Agent
 */
export async function deleteAgent(agentId: string): Promise<void> {
  return request(() =>
    api.delete(`/agents/${agentId}`)
  );
}

/**
 * 获取 Agent 执行历史
 */
export async function getAgentExecutions(query?: ExecutionsQuery): Promise<AgentExecution[]> {
  return request(() =>
    api.get<AgentExecution[]>('/agents/executions', { params: query })
  );
}

/**
 * 获取 Agent 系统列表
 */
export async function getAgentSystems(): Promise<AgentSystem[]> {
  return request(() =>
    api.get<AgentSystem[]>('/agents/systems')
  );
}

/**
 * 获取 Agent 系统详情
 */
export async function getAgentSystem(systemId: string): Promise<AgentSystem> {
  return request(() =>
    api.get<AgentSystem>(`/agents/systems/${systemId}`)
  );
}

/**
 * 删除 Agent 系统
 */
export async function deleteAgentSystem(systemId: string): Promise<void> {
  return request(() =>
    api.delete(`/agents/systems/${systemId}`)
  );
}

export default {
  getAgents,
  getAgentStats,
  getAgent,
  createAgent,
  updateAgent,
  deleteAgent,
  getAgentExecutions,
  getAgentSystems,
  getAgentSystem,
  deleteAgentSystem,
};
