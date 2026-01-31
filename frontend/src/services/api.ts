/**
 * Kiki Agent Framework - API 客户端基础配置
 *
 * 基于 Axios 的 HTTP 客户端，支持拦截器、错误处理和认证
 */

import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import type { ApiError } from '@/types/common';

/**
 * API 基础配置
 */
const API_BASE_URL = (import.meta.env as { VITE_API_BASE_URL?: string }).VITE_API_BASE_URL || '/api/v1';
const API_TIMEOUT = 30000;

/**
 * 创建 Axios 实例
 */
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * 请求拦截器
 * 自动添加认证 Token
 */
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // 从 localStorage 获取 access_token
    const token = localStorage.getItem('access_token');

    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

/**
 * 响应拦截器
 * 统一处理错误和 Token 刷新
 */
api.interceptors.response.use(
  (response) => {
    // 返回响应数据
    return response.data;
  },
  async (error: AxiosError<ApiError>) => {
    const { response } = error;

    // 处理 401 未授权错误
    if (response?.status === 401) {
      // 清除本地存储的 Token
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');

      // 如果不在登录页，跳转到登录页
      if (window.location.pathname !== '/auth/login') {
        window.location.href = '/auth/login';
      }
    }

    // 处理 429 速率限制
    if (response?.status === 429) {
      return Promise.reject({
        ...error,
        message: '请求过于频繁，请稍后再试',
      });
    }

    // 返回统一的错误格式
    return Promise.reject({
      status: response?.status,
      message: response?.data?.detail || error.message || '请求失败',
      error_code: response?.data?.error_code,
    });
  }
);

/**
 * API 错误类
 */
export class ApiRequestError extends Error {
  public status?: number;
  public errorCode?: string;

  constructor(message: string, status?: number, errorCode?: string) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.errorCode = errorCode;
  }
}

/**
 * 包装 API 请求，自动处理错误
 */
export async function request<T>(
  fn: () => Promise<T>
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (error instanceof ApiRequestError) {
      throw error;
    }

    // Axios 错误
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const data = error.response?.data as ApiError | undefined;
      const message = data?.detail || error.message || '请求失败';

      throw new ApiRequestError(message, status, data?.error_code);
    }

    // 未知错误
    throw new ApiRequestError('未知错误');
  }
}

/**
 * 设置认证 Token
 */
export function setAuthToken(token: string): void {
  localStorage.setItem('access_token', token);
}

/**
 * 获取认证 Token
 */
export function getAuthToken(): string | null {
  return localStorage.getItem('access_token');
}

/**
 * 清除认证 Token
 */
export function clearAuthToken(): void {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user');
}

/**
 * 检查是否已认证
 */
export function isAuthenticated(): boolean {
  return !!getAuthToken();
}

export default api;
