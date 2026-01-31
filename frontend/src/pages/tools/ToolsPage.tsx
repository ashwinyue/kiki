/**
 * Kiki Agent Framework - 工具管理页面
 */

import { useEffect, useState } from 'react';
import { tools as toolsService } from '@/services';
import type { Tool } from '@/types/tools';

export function ToolsPage() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadTools = async () => {
      try {
        const response = await toolsService.getTools();
        setTools(response.tools);
      } catch (error) {
        console.error('加载工具失败:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadTools();
  }, []);

  return (
    <div className="page tools-page">
      <div className="page-header">
        <h1 className="page-title">工具</h1>
      </div>

      {isLoading ? (
        <div className="loading-state">加载中...</div>
      ) : (
        <div className="tools-grid">
          {tools.map((tool) => (
            <div key={tool.name} className="tool-card">
              <h3 className="tool-name">{tool.name}</h3>
              <p className="tool-description">{tool.description}</p>
              <code className="tool-schema">{tool.args_schema}</code>
            </div>
          ))}
        </div>
      )}

      {tools.length === 0 && !isLoading && (
        <div className="empty-state">
          <p>暂无工具</p>
        </div>
      )}
    </div>
  );
}
