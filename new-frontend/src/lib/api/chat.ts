import { apiClient, ApiResponse } from './index';
import type { ChatMessage, ChatSession } from '@/types';

export interface SendMessageRequest {
  sessionId?: string;
  content: string;
  studyId?: string;
  attachments?: string[];
}

export interface SendMessageResponse {
  message: ChatMessage;
  sessionId: string;
}

export interface StreamingMessageCallback {
  onChunk: (chunk: string) => void;
  onComplete: (message: ChatMessage) => void;
  onError: (error: string) => void;
}

export const chatApi = {
  // Sessions
  async getSessions(): Promise<ApiResponse<ChatSession[]>> {
    return apiClient.get('/chat/sessions');
  },

  async getSession(sessionId: string): Promise<ApiResponse<ChatSession>> {
    return apiClient.get(`/chat/sessions/${sessionId}`);
  },

  async createSession(studyId?: string): Promise<ApiResponse<ChatSession>> {
    return apiClient.post('/chat/sessions', { studyId });
  },

  async deleteSession(sessionId: string): Promise<ApiResponse<void>> {
    return apiClient.delete(`/chat/sessions/${sessionId}`);
  },

  // Messages
  async getMessages(sessionId: string): Promise<ApiResponse<ChatMessage[]>> {
    return apiClient.get(`/chat/sessions/${sessionId}/messages`);
  },

  async sendMessage(request: SendMessageRequest): Promise<ApiResponse<SendMessageResponse>> {
    return apiClient.post('/chat/messages', request);
  },

  // Streaming message (for progressive responses)
  async sendMessageStreaming(
    request: SendMessageRequest,
    callbacks: StreamingMessageCallback
  ): Promise<void> {
    const response = await fetch('/api/chat/messages/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      callbacks.onError('Failed to send message');
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError('No response body');
      return;
    }

    const decoder = new TextDecoder();
    let fullContent = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        fullContent += chunk;
        callbacks.onChunk(chunk);
      }

      callbacks.onComplete({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: fullContent,
        timestamp: new Date().toISOString(),
        isStreaming: false,
      });
    } catch (error) {
      callbacks.onError(error instanceof Error ? error.message : 'Stream error');
    }
  },

  // Feedback
  async submitFeedback(
    messageId: string,
    rating: 'positive' | 'negative',
    comment?: string
  ): Promise<ApiResponse<void>> {
    return apiClient.post(`/chat/messages/${messageId}/feedback`, { rating, comment });
  },
};
