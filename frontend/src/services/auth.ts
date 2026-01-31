/**
 * Kiki Agent Framework - 认证服务
 */

import api, { request } from './api';
import type {
  User,
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  LoginResponse,
  AuthSession,
  CreateSessionRequest,
  CreateSessionResponse,
  UpdateSessionRequest,
} from '@/types/auth';

/**
 * 用户注册
 */
export async function register(data: RegisterRequest): Promise<RegisterResponse> {
  return request(() =>
    api.post<RegisterResponse>('/auth/register', data)
  );
}

/**
 * 用户登录 (表单格式)
 */
export async function login(data: LoginRequest): Promise<LoginResponse> {
  const params = new URLSearchParams();
  params.append('username', data.username);
  params.append('password', data.password);

  return request(() =>
    api.post<LoginResponse>('/auth/login', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })
  );
}

/**
 * 用户登录 (JSON 格式)
 */
export async function loginJson(data: LoginRequest): Promise<LoginResponse> {
  return request(() =>
    api.post<LoginResponse>('/auth/login/json', data)
  );
}

/**
 * 获取当前用户信息
 */
export async function getCurrentUser(): Promise<User> {
  return request(() =>
    api.get<User>('/auth/me')
  );
}

/**
 * 获取会话列表
 */
export async function getSessions(): Promise<AuthSession[]> {
  return request(() =>
    api.get<AuthSession[]>('/auth/sessions')
  );
}

/**
 * 创建会话
 */
export async function createSession(data: CreateSessionRequest): Promise<CreateSessionResponse> {
  const params = new URLSearchParams();
  params.append('name', data.name);

  return request(() =>
    api.post<CreateSessionResponse>('/auth/sessions', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })
  );
}

/**
 * 删除会话
 */
export async function deleteSession(sessionId: string): Promise<void> {
  return request(() =>
    api.delete(`/auth/sessions/${sessionId}`)
  );
}

/**
 * 更新会话名称
 */
export async function updateSession(sessionId: string, data: UpdateSessionRequest): Promise<void> {
  const params = new URLSearchParams();
  params.append('name', data.name);

  return request(() =>
    api.patch(`/auth/sessions/${sessionId}`, params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })
  );
}

export default {
  register,
  login,
  loginJson,
  getCurrentUser,
  getSessions,
  createSession,
  deleteSession,
  updateSession,
};
