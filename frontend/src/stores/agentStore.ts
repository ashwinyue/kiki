/**
 * Kiki Agent Framework - Agent 状态管理
 *
 * 使用 Zustand 进行轻量级状态管理
 */

import { create } from 'zustand';
import type { Agent, AgentType, AgentStatus } from '@/types/agent';
import { agent as agentService } from '@/services';

/**
 * Agent 统计信息类型
 */
interface AgentStats {
  total: number;
  active: number;
  by_type: Record<string, number>;
}

/**
 * Agent 状态接口
 */
interface AgentState {
  // 状态
  agents: Agent[];
  stats: AgentStats | null;
  isLoading: boolean;
  error: string | null;
  selectedAgentId: string | null;

  // Actions
  loadAgents: (filters?: { agent_type?: AgentType; status?: AgentStatus }) => Promise<void>;
  loadStats: () => Promise<void>;
  createAgent: (data: {
    name: string;
    description?: string;
    agent_type: AgentType;
    model_name: string;
    system_prompt: string;
    temperature: number;
  }) => Promise<Agent>;
  updateAgent: (agentId: string, data: Partial<Agent>) => Promise<void>;
  deleteAgent: (agentId: string) => Promise<void>;
  selectAgent: (agentId: string | null) => void;
  clearError: () => void;
}

/**
 * Agent Store
 */
export const useAgentStore = create<AgentState>()((set, get) => ({
  // 初始状态
  agents: [],
  stats: null,
  isLoading: false,
  error: null,
  selectedAgentId: null,

  // 加载 Agent 列表
  loadAgents: async (filters) => {
    set({ isLoading: true, error: null });

    try {
      const response = await agentService.getAgents(filters);
      set({
        agents: response.agents,
        isLoading: false,
      });
    } catch (error: unknown) {
      set({
        error: (error as Error).message || '加载 Agent 失败',
        isLoading: false,
      });
      throw error;
    }
  },

  // 加载统计信息
  loadStats: async () => {
    try {
      const stats = await agentService.getAgentStats();
      set({ stats });
    } catch (error: unknown) {
      set({ error: (error as Error).message || '加载统计失败' });
      throw error;
    }
  },

  // 创建 Agent
  createAgent: async (data) => {
    set({ isLoading: true, error: null });

    try {
      const agent = await agentService.createAgent(data);

      set((state) => ({
        agents: [...state.agents, agent],
        isLoading: false,
      }));

      return agent;
    } catch (error: unknown) {
      set({
        error: (error as Error).message || '创建 Agent 失败',
        isLoading: false,
      });
      throw error;
    }
  },

  // 更新 Agent
  updateAgent: async (agentId, data) => {
    set({ isLoading: true, error: null });

    try {
      const updatedAgent = await agentService.updateAgent(agentId, data);

      set((state) => ({
        agents: state.agents.map((a) => (a.id === agentId ? updatedAgent : a)),
        isLoading: false,
      }));
    } catch (error: unknown) {
      set({
        error: (error as Error).message || '更新 Agent 失败',
        isLoading: false,
      });
      throw error;
    }
  },

  // 删除 Agent
  deleteAgent: async (agentId) => {
    set({ isLoading: true, error: null });

    try {
      await agentService.deleteAgent(agentId);

      set((state) => ({
        agents: state.agents.filter((a) => a.id !== agentId),
        selectedAgentId:
          state.selectedAgentId === agentId ? null : state.selectedAgentId,
        isLoading: false,
      }));
    } catch (error: unknown) {
      set({
        error: (error as Error).message || '删除 Agent 失败',
        isLoading: false,
      });
      throw error;
    }
  },

  // 选择 Agent
  selectAgent: (agentId) => {
    set({ selectedAgentId: agentId });
  },

  // 清除错误
  clearError: () => {
    set({ error: null });
  },
}));

export default useAgentStore;
