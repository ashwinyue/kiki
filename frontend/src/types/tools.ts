/**
 * Kiki Agent Framework - 工具相关类型
 */

/**
 * 工具信息
 */
export interface Tool {
  name: string;
  description: string;
  args_schema: string;
}

/**
 * 工具列表响应
 */
export interface ToolsResponse {
  tools: Tool[];
  count: number;
}

/**
 * 工具详情响应
 */
export interface ToolDetailResponse extends Tool {}
