/**
 * Kiki Agent Framework - Agent 管理页面
 */

import { useEffect } from 'react';
import { useAgentStore } from '@/stores';
import { Button } from '@/components/Button';

export function AgentsPage() {
  const { agents, stats, isLoading, loadAgents, loadStats } = useAgentStore();

  useEffect(() => {
    loadAgents();
    loadStats();
  }, [loadAgents, loadStats]);

  return (
    <div className="page agents-page">
      <div className="page-header">
        <h1 className="page-title">Agents</h1>
        <Button variant="primary">创建 Agent</Button>
      </div>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-value">{stats.total}</span>
            <span className="stat-label">总数</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.active}</span>
            <span className="stat-label">活跃</span>
          </div>
        </div>
      )}

      <div className="agents-grid">
        {agents.map((agent) => (
          <div key={agent.id} className="agent-card">
            <h3 className="agent-name">{agent.name}</h3>
            <p className="agent-description">{agent.description || '无描述'}</p>
            <div className="agent-meta">
              <span className="agent-type">{agent.agent_type}</span>
              <span className="agent-model">{agent.model_name}</span>
            </div>
          </div>
        ))}
      </div>

      {agents.length === 0 && !isLoading && (
        <div className="empty-state">
          <p>暂无 Agent，点击上方按钮创建</p>
        </div>
      )}
    </div>
  );
}
