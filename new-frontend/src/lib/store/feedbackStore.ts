import { create } from 'zustand';
import type { FeedbackState, ReportFeedback, SegmentationFeedback, ChatFeedback } from '@/types';

interface FeedbackActions {
  addReportFeedback: (feedback: Omit<ReportFeedback, 'id' | 'submittedAt'>) => void;
  addSegmentationFeedback: (feedback: Omit<SegmentationFeedback, 'id' | 'submittedAt'>) => void;
  addChatFeedback: (feedback: Omit<ChatFeedback, 'id' | 'submittedAt'>) => void;
  submitPendingFeedback: () => Promise<void>;
  clearPendingFeedback: () => void;
  loadFeedbackHistory: () => Promise<void>;
}

type FeedbackStore = FeedbackState & FeedbackActions;

export const useFeedbackStore = create<FeedbackStore>((set, get) => ({
  pendingFeedback: {
    report: [],
    segmentation: [],
    chat: [],
  },
  feedbackHistory: {
    report: [],
    segmentation: [],
    chat: [],
  },

  addReportFeedback: (feedback) => set((state) => ({
    pendingFeedback: {
      ...state.pendingFeedback,
      report: [
        ...state.pendingFeedback.report,
        {
          ...feedback,
          id: 'rf-' + Date.now(),
          submittedAt: new Date().toISOString(),
        },
      ],
    },
  })),

  addSegmentationFeedback: (feedback) => set((state) => ({
    pendingFeedback: {
      ...state.pendingFeedback,
      segmentation: [
        ...state.pendingFeedback.segmentation,
        {
          ...feedback,
          id: 'sf-' + Date.now(),
          submittedAt: new Date().toISOString(),
        },
      ],
    },
  })),

  addChatFeedback: (feedback) => set((state) => ({
    pendingFeedback: {
      ...state.pendingFeedback,
      chat: [
        ...state.pendingFeedback.chat,
        {
          ...feedback,
          id: 'cf-' + Date.now(),
          submittedAt: new Date().toISOString(),
        },
      ],
    },
  })),

  submitPendingFeedback: async () => {
    const { pendingFeedback } = get();
    
    try {
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      set((state) => ({
        feedbackHistory: {
          report: [...state.feedbackHistory.report, ...pendingFeedback.report],
          segmentation: [...state.feedbackHistory.segmentation, ...pendingFeedback.segmentation],
          chat: [...state.feedbackHistory.chat, ...pendingFeedback.chat],
        },
        pendingFeedback: {
          report: [],
          segmentation: [],
          chat: [],
        },
      }));
    } catch (error) {
      throw error;
    }
  },

  clearPendingFeedback: () => set({
    pendingFeedback: {
      report: [],
      segmentation: [],
      chat: [],
    },
  }),

  loadFeedbackHistory: async () => {
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      // Would load from API
    } catch (error) {
      throw error;
    }
  },
}));
