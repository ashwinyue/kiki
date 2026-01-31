/**
 * Kiki Agent Framework - 通用类型定义
 */

/**
 * 通用 API 响应格式
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  meta?: {
    total: number;
    page: number;
    limit: number;
  };
}

/**
 * 分页响应格式
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

/**
 * API 错误响应
 */
export interface ApiError {
  detail: string;
  status_code: number;
  error_code?: string;
}

/**
 * 分页查询参数
 */
export interface PaginationQuery {
  page?: number;
  size?: number;
  keyword?: string;
}

/**
 * 状态枚举
 */
export type Status = 'active' | 'disabled' | 'deleted' | 'pending' | 'suspended';

/**
 * 日期时间格式 (ISO 8601)
 */
export type DateTime = string;
