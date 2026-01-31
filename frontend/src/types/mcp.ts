/**
 * Kiki Agent Framework - MCP 服务相关类型
 */

/**
 * MCP 传输类型
 */
export type McpTransportType = 'stdio' | 'http' | 'sse';

/**
 * MCP 服务信息
 */
export interface McpService {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;
  transport_type: McpTransportType;
  url?: string;
  headers?: Record<string, string>;
  auth_config?: Record<string, unknown>;
  advanced_config?: Record<string, unknown>;
  stdio_config?: Record<string, unknown>;
  env_vars?: Record<string, string>;
  created_at: string;
  updated_at: string;
}

/**
 * 创建 MCP 服务请求
 */
export interface CreateMcpServiceRequest {
  name: string;
  description?: string;
  enabled?: boolean;
  transport_type: McpTransportType;
  url?: string;
  headers?: Record<string, string>;
  auth_config?: Record<string, unknown>;
  advanced_config?: Record<string, unknown>;
  stdio_config?: Record<string, unknown>;
  env_vars?: Record<string, string>;
}

/**
 * 更新 MCP 服务请求
 */
export interface UpdateMcpServiceRequest {
  name?: string;
  description?: string;
  enabled?: boolean;
  url?: string;
  headers?: Record<string, string>;
  auth_config?: Record<string, unknown>;
  advanced_config?: Record<string, unknown>;
  stdio_config?: Record<string, unknown>;
  env_vars?: Record<string, string>;
}

/**
 * MCP 服务列表查询参数
 */
export interface McpServicesQuery {
  include_disabled?: boolean;
}
