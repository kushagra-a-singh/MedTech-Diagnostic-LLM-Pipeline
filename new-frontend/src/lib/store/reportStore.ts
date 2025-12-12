import { create } from 'zustand';
import type { ReportState, Report, ReportSection, ReportTemplate } from '@/types';

interface ReportActions {
  setCurrentReport: (report: Report | null) => void;
  updateReportSection: (sectionId: string, content: string) => void;
  addReportSection: (section: Omit<ReportSection, 'id'>) => void;
  removeReportSection: (sectionId: string) => void;
  reorderSections: (fromIndex: number, toIndex: number) => void;
  setReportTemplates: (templates: ReportTemplate[]) => void;
  setGenerating: (isGenerating: boolean) => void;
  setSaving: (isSaving: boolean) => void;
  setStudyContext: (context: any) => void;
  generateReport: (studyId: string, templateId?: string) => Promise<void>;
  saveReport: () => Promise<void>;
  loadReport: (reportId: string) => Promise<void>;
}

type ReportStore = ReportState & ReportActions;

const defaultTemplates: ReportTemplate[] = [
  {
    id: 'ct-chest',
    name: 'CT Chest',
    modality: 'CT',
    sections: [
      { title: 'Clinical History', order: 1 },
      { title: 'Technique', order: 2 },
      { title: 'Comparison', order: 3 },
      { title: 'Findings', order: 4 },
      { title: 'Impression', order: 5 },
    ],
  },
  {
    id: 'mri-brain',
    name: 'MRI Brain',
    modality: 'MR',
    sections: [
      { title: 'Clinical History', order: 1 },
      { title: 'Technique', order: 2 },
      { title: 'Comparison', order: 3 },
      { title: 'Findings', order: 4 },
      { title: 'Impression', order: 5 },
    ],
  },
  {
    id: 'xray-chest',
    name: 'X-Ray Chest',
    modality: 'CR',
    sections: [
      { title: 'Clinical Indication', order: 1 },
      { title: 'Technique', order: 2 },
      { title: 'Findings', order: 3 },
      { title: 'Impression', order: 4 },
    ],
  },
];

export const useReportStore = create<ReportStore>((set, get) => ({
  currentReport: null,
  reportTemplates: defaultTemplates,
  isGenerating: false,
  isSaving: false,
  currentStudyContext: null,

  setCurrentReport: (report) => set({ currentReport: report }),

  setStudyContext: (context) => set({ currentStudyContext: context }),

  updateReportSection: (sectionId, content) => set((state) => {
    if (!state.currentReport) return state;
    return {
      currentReport: {
        ...state.currentReport,
        sections: state.currentReport.sections.map(section =>
          section.id === sectionId
            ? { ...section, content, isEdited: true }
            : section
        ),
        updatedAt: new Date().toISOString(),
      },
    };
  }),

  addReportSection: (section) => set((state) => {
    if (!state.currentReport) return state;
    const newSection: ReportSection = {
      ...section,
      id: 'section-' + Date.now(),
    };
    return {
      currentReport: {
        ...state.currentReport,
        sections: [...state.currentReport.sections, newSection],
        updatedAt: new Date().toISOString(),
      },
    };
  }),

  removeReportSection: (sectionId) => set((state) => {
    if (!state.currentReport) return state;
    return {
      currentReport: {
        ...state.currentReport,
        sections: state.currentReport.sections.filter(s => s.id !== sectionId),
        updatedAt: new Date().toISOString(),
      },
    };
  }),

  reorderSections: (fromIndex, toIndex) => set((state) => {
    if (!state.currentReport) return state;
    const sections = [...state.currentReport.sections];
    const [removed] = sections.splice(fromIndex, 1);
    sections.splice(toIndex, 0, removed);
    return {
      currentReport: {
        ...state.currentReport,
        sections: sections.map((s, i) => ({ ...s, order: i + 1 })),
        updatedAt: new Date().toISOString(),
      },
    };
  }),

  setReportTemplates: (templates) => set({ reportTemplates: templates }),
  setGenerating: (isGenerating) => set({ isGenerating }),
  setSaving: (isSaving) => set({ isSaving }),

  generateReport: async (studyId: string, templateId?: string) => {
    set({ isGenerating: true });
    try {
      const { currentStudyContext } = get();

      // Call backend to generate report using the uploaded DICOM context
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: 'Generate a comprehensive medical report based on the uploaded imaging study',
          session_id: `report-${studyId}`,
          scan_context: currentStudyContext,
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const data = await response.json();
      const reportText = data.response || 'No report generated';

      const template = templateId
        ? get().reportTemplates.find(t => t.id === templateId)
        : get().reportTemplates[0];

      // Parse the report text into sections (basic splitting by common headers)
      const generatedSections: ReportSection[] = [];

      // Try to split by common section headers
      const sectionPatterns = [
        { title: 'Clinical History', pattern: /Clinical History:?\s*/i },
        { title: 'Technique', pattern: /Technique:?\s*/i },
        { title: 'Comparison', pattern: /Comparison:?\s*/i },
        { title: 'Findings', pattern: /Findings:?\s*/i },
        { title: 'Impression', pattern: /Impression:?\s*/i },
      ];

      let remainingText = reportText;
      let order = 1;

      for (const { title, pattern } of sectionPatterns) {
        const match = remainingText.match(pattern);
        if (match) {
          const startIndex = match.index! + match[0].length;
          const nextPattern = sectionPatterns
            .filter(p => p.title !== title)
            .map(p => p.pattern)
            .find(p => {
              const nextMatch = remainingText.substring(startIndex).match(p);
              return nextMatch;
            });

          let content;
          if (nextPattern) {
            const nextMatch = remainingText.substring(startIndex).match(nextPattern);
            content = remainingText.substring(startIndex, startIndex + nextMatch!.index!).trim();
          } else {
            content = remainingText.substring(startIndex).trim();
          }

          generatedSections.push({
            id: `section-${order}`,
            title,
            content,
            order,
            isAIGenerated: true,
            isEdited: false,
          });
          order++;
        }
      }

      // If no sections were parsed, create one section with all content
      if (generatedSections.length === 0) {
        generatedSections.push({
          id: 'section-1',
          title: 'Report',
          content: reportText,
          order: 1,
          isAIGenerated: true,
          isEdited: false,
        });
      }

      const newReport: Report = {
        id: 'report-' + Date.now(),
        studyId,
        templateId,
        title: `${currentStudyContext?.modality || 'Medical'} Report`,
        sections: generatedSections,
        status: 'draft',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        createdBy: 'current-user',
      };

      set({ currentReport: newReport, isGenerating: false });
    } catch (error) {
      console.error('Report generation error:', error);
      set({ isGenerating: false });
      throw error;
    }
  },

  saveReport: async () => {
    set({ isSaving: true });
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      set((state) => ({
        currentReport: state.currentReport
          ? { ...state.currentReport, updatedAt: new Date().toISOString() }
          : null,
        isSaving: false,
      }));
    } catch (error) {
      set({ isSaving: false });
      throw error;
    }
  },

  loadReport: async (reportId: string) => {
    set({ isGenerating: true });
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      // Mock load - would fetch from API
      set({ isGenerating: false });
    } catch (error) {
      set({ isGenerating: false });
      throw error;
    }
  },
}));
