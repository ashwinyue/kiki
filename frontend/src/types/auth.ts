/**
 * Kiki Agent Framework - 认证相关类型
 */

/**
 * 用户信息
 */
export interface User {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at?: string;
}

/**
 * 认证 Token 响应
 */
export interface AuthTokens {
  access_token: string;
  token_type: 'bearer';
  expires_at: string; // ISO 8601 格式
}

/**
 * 注册请求
 */
export interface RegisterRequest {
  email: string;
  password: string; // 8-100 字符
  full_name?: string;
}

/**
 * 注册响应
 */
export interface RegisterResponse extends User, AuthTokens {}

/**
 * 登录请求
 */
export interface LoginRequest {
  username: string; // 邮箱
  password: string;
}

/**
 * 登录响应
 */
export interface LoginResponse extends AuthTokens {}

/**
 * 认证会话信息
 */
export interface AuthSession {
  id: string;
  name: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * 创建会话请求
 */
export interface CreateSessionRequest {
  name: string;
}

/**
 * 创建会话响应
 */
export interface CreateSessionResponse {
  session_id: string;
  token: string;
  name: string;
}

/**
 * 更新会话请求
 */
export interface UpdateSessionRequest {
  name: string;
}
