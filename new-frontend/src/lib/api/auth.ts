import { apiClient, ApiResponse } from './index';
import type { User } from '@/types';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
  role?: 'radiologist' | 'technician' | 'admin';
}

export interface AuthResponse {
  user: User;
  token: string;
  expiresAt: string;
}

export const authApi = {
  async login(credentials: LoginRequest): Promise<ApiResponse<AuthResponse>> {
    const response = await apiClient.post<AuthResponse>('/auth/login', credentials);
    if (response.data?.token) {
      apiClient.setToken(response.data.token);
    }
    return response;
  },

  async register(data: RegisterRequest): Promise<ApiResponse<AuthResponse>> {
    const response = await apiClient.post<AuthResponse>('/auth/register', data);
    if (response.data?.token) {
      apiClient.setToken(response.data.token);
    }
    return response;
  },

  async logout(): Promise<ApiResponse<void>> {
    const response = await apiClient.post<void>('/auth/logout');
    apiClient.setToken(null);
    return response;
  },

  async refreshToken(): Promise<ApiResponse<{ token: string; expiresAt: string }>> {
    return apiClient.post('/auth/refresh');
  },

  async getCurrentUser(): Promise<ApiResponse<User>> {
    return apiClient.get('/auth/me');
  },

  async updateProfile(data: Partial<User>): Promise<ApiResponse<User>> {
    return apiClient.patch('/auth/profile', data);
  },

  async changePassword(currentPassword: string, newPassword: string): Promise<ApiResponse<void>> {
    return apiClient.post('/auth/change-password', { currentPassword, newPassword });
  },

  async forgotPassword(email: string): Promise<ApiResponse<void>> {
    return apiClient.post('/auth/forgot-password', { email });
  },

  async resetPassword(token: string, newPassword: string): Promise<ApiResponse<void>> {
    return apiClient.post('/auth/reset-password', { token, newPassword });
  },
};
