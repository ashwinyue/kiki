/**
 * Kiki Agent Framework - Agent 相关类型
 */

/**
 * Agent 类型
 */
export type AgentType = 'single' | 'router' | 'supervisor' | 'worker' | 'handoff';

/**
 * Agent 状态
 */
export type AgentStatus = 'active' | 'disabled' | 'deleted';

/**
 * Agent 信息
 */
export interface Agent {
  id: string;
  name: string;
  description?: string;
  agent_type: AgentType;
  model_name: string;
  system_prompt: string;
  temperature: number;
  max_tokens?: number;
  config?: Record<string, unknown>;
  status: AgentStatus;
  created_at: string;
  updated_at: string;
}

/**
 * 创建 Agent 请求
 */
export interface CreateAgentRequest {
  name: string;
  description?: string;
  agent_type: AgentType;
  model_name: string;
  system_prompt: string;
  temperature: number;
  max_tokens?: number;
  config?: Record<string, unknown>;
}

/**
 * 更新 Agent 请求
 */
export interface UpdateAgentRequest {
  name?: string;
  description?: string;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
  config?: Record<string, unknown>;
}

/**
 * Agent 列表查询参数
 */
export interface AgentListQuery {
  agent_type?: AgentType;
  status?: AgentStatus;
  page?: number;
  size?: number;
}

/**
 * Agent 列表响应
 */
export interface AgentListResponse {
  agents: Agent[];
  total: number;
  page: number;
  size: number;
}

/**
 * Agent 统计响应
 */
export interface AgentStatsResponse {
  total: number;
  active: number;
  by_type: Record<string, number>;
}

/**
 * Agent 执行记录
 */
export interface AgentExecution {
  id: string;
  agent_id: string;
  status: string;
  input: string;
  output?: string;
  error?: string;
  started_at: string;
  completed_at?: string;
}

/**
 * Agent 执行查询参数
 */
export interface ExecutionsQuery {
  agent_id?: string;
  limit?: number;
}

/**
 * Agent 系统信息
 */
export interface AgentSystem {
  id: string;
  name: string;
  description?: string;
  system_type: string;
  created_at: string;
  updated_at: string;
}
