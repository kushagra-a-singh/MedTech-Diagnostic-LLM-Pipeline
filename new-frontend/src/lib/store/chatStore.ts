import { create } from 'zustand';
import type { ChatState, ChatMessage, ChatAttachment } from '@/types';

interface ChatActions {
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  updateMessage: (id: string, content: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  clearMessages: () => void;
  setIsGenerating: (isGenerating: boolean) => void;
  setSessionId: (sessionId: string | null) => void;
  setContext: (context: ChatState['currentContext']) => void;
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
      // Simulate streaming response
      const mockResponse = "Based on the imaging findings, I can observe the following:\n\n1. **Lung Fields**: The lung parenchyma appears clear with no evidence of consolidation, masses, or nodules.\n\n2. **Cardiac Silhouette**: The heart size is within normal limits with a cardiothoracic ratio of approximately 0.45.\n\n3. **Mediastinum**: No mediastinal widening or lymphadenopathy noted.\n\n4. **Pleural Spaces**: No pleural effusion or pneumothorax identified.\n\n**Impression**: Normal chest CT examination. Would you like me to elaborate on any specific finding?";
      
      let currentText = '';
      const words = mockResponse.split(' ');
      
      for (const word of words) {
        await new Promise(resolve => setTimeout(resolve, 30));
        currentText += (currentText ? ' ' : '') + word;
        updateMessage(assistantMessage.id, currentText);
      }
      
      set((state) => ({
        messages: state.messages.map(msg =>
          msg.id === assistantMessage.id ? { ...msg, isStreaming: false } : msg
        ),
      }));
    } catch (error) {
      updateMessage(assistantMessage.id, 'I apologize, but I encountered an error processing your request. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  },
}));
