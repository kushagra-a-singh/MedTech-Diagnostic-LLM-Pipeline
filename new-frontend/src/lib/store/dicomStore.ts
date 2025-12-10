import { create } from 'zustand';
import type { DicomState, DicomStudy, DicomSeries, DicomInstance, SegmentationOverlay, ViewportConfig, Measurement } from '@/types';

interface DicomActions {
  setCurrentStudy: (study: DicomStudy | null) => void;
  setCurrentSeries: (series: DicomSeries | null) => void;
  setCurrentInstance: (instance: DicomInstance | null) => void;
  setSeriesList: (series: DicomSeries[]) => void;
  setInstanceList: (instances: DicomInstance[]) => void;
  setSegmentation: (overlays: SegmentationOverlay[]) => void;
  toggleSegmentationVisibility: (id: string) => void;
  updateSegmentationOpacity: (id: string, opacity: number) => void;
  updateViewportConfig: (config: Partial<ViewportConfig>) => void;
  resetViewport: () => void;
  addMeasurement: (measurement: Measurement) => void;
  removeMeasurement: (id: string) => void;
  clearMeasurements: () => void;
  setLoading: (isLoading: boolean) => void;
  uploadDicom: (file: File) => Promise<DicomStudy>;
  loadStudy: (studyId: string) => Promise<void>;
  loadInstance: (instanceId: string) => Promise<void>;
  processSegmentation: (studyId: string) => Promise<void>;
}

type DicomStore = DicomState & DicomActions;

const defaultViewportConfig: ViewportConfig = {
  zoom: 1,
  pan: { x: 0, y: 0 },
  windowCenter: 40,
  windowWidth: 400,
  rotation: 0,
  flipH: false,
  flipV: false,
  invert: false,
};

export const useDicomStore = create<DicomStore>((set, get) => ({
  currentStudy: null,
  currentSeries: null,
  currentInstance: null,
  seriesList: [],
  instanceList: [],
  segmentation: [],
  viewportConfig: defaultViewportConfig,
  measurements: [],
  isLoading: false,

  setCurrentStudy: (study) => set({ currentStudy: study }),
  setCurrentSeries: (series) => set({ currentSeries: series }),
  setCurrentInstance: (instance) => set({ currentInstance: instance }),
  setSeriesList: (seriesList) => set({ seriesList }),
  setInstanceList: (instanceList) => set({ instanceList }),
  
  setSegmentation: (overlays) => set({ segmentation: overlays }),
  
  toggleSegmentationVisibility: (id) => set((state) => ({
    segmentation: state.segmentation.map(overlay =>
      overlay.id === id ? { ...overlay, visible: !overlay.visible } : overlay
    ),
  })),
  
  updateSegmentationOpacity: (id, opacity) => set((state) => ({
    segmentation: state.segmentation.map(overlay =>
      overlay.id === id ? { ...overlay, opacity } : overlay
    ),
  })),

  updateViewportConfig: (config) => set((state) => ({
    viewportConfig: { ...state.viewportConfig, ...config },
  })),
  
  resetViewport: () => set({ viewportConfig: defaultViewportConfig }),

  addMeasurement: (measurement) => set((state) => ({
    measurements: [...state.measurements, measurement],
  })),
  
  removeMeasurement: (id) => set((state) => ({
    measurements: state.measurements.filter(m => m.id !== id),
  })),
  
  clearMeasurements: () => set({ measurements: [] }),

  setLoading: (isLoading) => set({ isLoading }),

  uploadDicom: async (file: File) => {
    set({ isLoading: true });
    try {
      // Simulated upload
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockStudy: DicomStudy = {
        studyId: 'study-' + Date.now(),
        patientId: 'P-' + Math.random().toString(36).substr(2, 9),
        patientName: 'Demo Patient',
        studyDate: new Date().toISOString().split('T')[0],
        studyDescription: 'CT Chest w/ Contrast',
        modality: 'CT',
        seriesCount: 3,
        instanceCount: 150,
        status: 'completed',
      };
      
      set({ currentStudy: mockStudy, isLoading: false });
      return mockStudy;
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  loadStudy: async (studyId: string) => {
    set({ isLoading: true });
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const mockSeries: DicomSeries[] = [
        { seriesId: 's1', seriesNumber: 1, seriesDescription: 'Axial', modality: 'CT', instanceCount: 50, bodyPart: 'Chest' },
        { seriesId: 's2', seriesNumber: 2, seriesDescription: 'Coronal', modality: 'CT', instanceCount: 50, bodyPart: 'Chest' },
        { seriesId: 's3', seriesNumber: 3, seriesDescription: 'Sagittal', modality: 'CT', instanceCount: 50, bodyPart: 'Chest' },
      ];
      
      set({ seriesList: mockSeries, currentSeries: mockSeries[0], isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  loadInstance: async (instanceId: string) => {
    set({ isLoading: true });
    try {
      await new Promise(resolve => setTimeout(resolve, 200));
      
      const mockInstance: DicomInstance = {
        instanceId,
        instanceNumber: 1,
        sopClassUid: '1.2.840.10008.5.1.4.1.1.2',
        imageUrl: '/placeholder.svg',
        rows: 512,
        columns: 512,
        windowCenter: 40,
        windowWidth: 400,
      };
      
      set({ currentInstance: mockInstance, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  processSegmentation: async (studyId: string) => {
    set({ isLoading: true });
    try {
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const mockOverlays: SegmentationOverlay[] = [
        { id: 'seg1', name: 'Lungs', color: '#4FC3D1', opacity: 0.5, visible: true, maskUrl: '', findings: ['Normal lung parenchyma'] },
        { id: 'seg2', name: 'Heart', color: '#FF6B6B', opacity: 0.5, visible: true, maskUrl: '', findings: ['Normal cardiac silhouette'] },
        { id: 'seg3', name: 'Nodules', color: '#FFE66D', opacity: 0.7, visible: true, maskUrl: '', findings: ['No suspicious nodules detected'] },
      ];
      
      set({ segmentation: mockOverlays, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },
}));
