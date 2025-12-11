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

  setCurrentReport: (report) => set({ currentReport: report }),

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
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const template = templateId
        ? get().reportTemplates.find(t => t.id === templateId)
        : get().reportTemplates[0];

      const generatedSections: ReportSection[] = [
        {
          id: 'section-1',
          title: 'Clinical History',
          content: 'Patient presents with shortness of breath and chronic cough. Rule out pulmonary pathology.',
          order: 1,
          isAIGenerated: true,
          isEdited: false,
        },
        {
          id: 'section-2',
          title: 'Technique',
          content: 'CT of the chest was performed with intravenous contrast administration. Axial images were obtained from the thoracic inlet to the upper abdomen with multiplanar reconstructions.',
          order: 2,
          isAIGenerated: true,
          isEdited: false,
        },
        {
          id: 'section-3',
          title: 'Comparison',
          content: 'No prior studies available for comparison.',
          order: 3,
          isAIGenerated: true,
          isEdited: false,
        },
        {
          id: 'section-4',
          title: 'Findings',
          content: '**Lungs and Airways:**\n- Lung parenchyma demonstrates normal attenuation without evidence of consolidation, ground-glass opacity, or masses.\n- No pulmonary nodules identified.\n- Airways are patent without evidence of bronchiectasis.\n\n**Pleura:**\n- No pleural effusion or pneumothorax.\n- No pleural thickening or masses.\n\n**Mediastinum:**\n- Heart size is normal. No pericardial effusion.\n- Great vessels are unremarkable.\n- No mediastinal or hilar lymphadenopathy.\n\n**Chest Wall:**\n- Osseous structures are intact without suspicious lesions.\n- Soft tissues are unremarkable.',
          order: 4,
          isAIGenerated: true,
          isEdited: false,
        },
        {
          id: 'section-5',
          title: 'Impression',
          content: '1. Normal CT chest examination.\n2. No acute cardiopulmonary process identified.\n3. No evidence of pulmonary nodules or masses.',
          order: 5,
          isAIGenerated: true,
          isEdited: false,
        },
      ];

      const newReport: Report = {
        id: 'report-' + Date.now(),
        studyId,
        templateId,
        title: 'CT Chest Report',
        sections: generatedSections,
        status: 'draft',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        createdBy: 'current-user',
      };

      set({ currentReport: newReport, isGenerating: false });
    } catch (error) {
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
