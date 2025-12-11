import { create } from 'zustand';
import type { ChatState, ChatMessage, ChatAttachment } from '@/types';

interface ChatActions {
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  updateMessage: (id: string, content: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  clearMessages: () => void;
  setIsGenerating: (isGenerating: boolean) => void;
  setSessionId: (sessionId: string | null) => void;
  setContext: (context: any) => void;
  sendChatMessage: (content: string, attachments?: ChatAttachment[]) => Promise<void>;
}

type ChatStore = ChatState & ChatActions;

export const useChatStore = create<ChatStore>((set, get) => ({
  messages: [],
  isGenerating: false,
  sessionId: null,
  currentContext: {},

  addMessage: (message) => {
    const newMessage: ChatMessage = {
      ...message,
      id: 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toISOString(),
    };
    set((state) => ({ messages: [...state.messages, newMessage] }));
    return newMessage;
  },

  updateMessage: (id, content) => set((state) => ({
    messages: state.messages.map(msg =>
      msg.id === id ? { ...msg, content, isStreaming: false } : msg
    ),
  })),

  setMessages: (messages) => set({ messages }),
  clearMessages: () => set({ messages: [] }),
  setIsGenerating: (isGenerating) => set({ isGenerating }),
  setSessionId: (sessionId) => set({ sessionId }),
  setContext: (context) => set((state) => ({
    currentContext: { ...state.currentContext, ...context },
  })),

  sendChatMessage: async (content: string, attachments?: ChatAttachment[]) => {
    const { addMessage, setIsGenerating, updateMessage } = get();

    // Add user message
    addMessage({
      role: 'user',
      content,
      attachments,
    });

    setIsGenerating(true);

    // Add placeholder for assistant message
    const assistantMessage: ChatMessage = {
      id: 'msg-' + Date.now() + '-assistant',
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };
    set((state) => ({ messages: [...state.messages, assistantMessage] }));

    try {
      const { sessionId, messages, currentContext } = get();

      // Prepare request payload
      const history = messages.slice(0, -1).map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const payload = {
        message: content,
        session_id: sessionId || `session-${Date.now()}`,
        scan_context: currentContext,
        conversation_history: history,
        stream: true
      };

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      // Handle streaming response
      if (response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentText = '';
        let isFirstChunk = true;

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          currentText += chunk;

          // Initial update or append
          if (isFirstChunk) {
            updateMessage(assistantMessage.id, currentText.trimStart());
            isFirstChunk = false;
          } else {
            updateMessage(assistantMessage.id, currentText);
          }
        }
      } else {
        // Fallback for non-streaming response
        const data = await response.json();
        updateMessage(assistantMessage.id, data.response);
      }

      set((state) => ({
        messages: state.messages.map(msg =>
          msg.id === assistantMessage.id ? { ...msg, isStreaming: false } : msg
        ),
      }));

    } catch (error) {
      console.error("Chat error:", error);
      updateMessage(assistantMessage.id, 'I apologize, but I encountered an error connecting to the medical AI. Please ensure the backend server is running.');
    } finally {
      setIsGenerating(false);
    }
  },
}));
