import { apiClient, ApiResponse, PaginatedResponse } from './index';
import type { DicomStudy, DicomSeries, DicomInstance, SegmentationOverlay } from '@/types';

export interface StudyListParams {
  page?: number;
  pageSize?: number;
  patientId?: string;
  modality?: string;
  dateFrom?: string;
  dateTo?: string;
  search?: string;
}

export interface SegmentationResult {
  overlays: SegmentationOverlay[];
  processingTime: number;
  modelVersion: string;
}

export const dicomApi = {
  // Studies
  async getStudies(params?: StudyListParams): Promise<ApiResponse<PaginatedResponse<DicomStudy>>> {
    return apiClient.get('/dicom/studies', params as Record<string, string>);
  },

  async getStudy(studyId: string): Promise<ApiResponse<DicomStudy>> {
    return apiClient.get(`/dicom/studies/${studyId}`);
  },

  async uploadStudy(file: File, onProgress?: (progress: number) => void): Promise<ApiResponse<DicomStudy>> {
    return apiClient.upload('/dicom/upload', file, onProgress);
  },

  async deleteStudy(studyId: string): Promise<ApiResponse<void>> {
    return apiClient.delete(`/dicom/studies/${studyId}`);
  },

  // Series
  async getSeries(studyId: string): Promise<ApiResponse<DicomSeries[]>> {
    return apiClient.get(`/dicom/studies/${studyId}/series`);
  },

  async getSeriesDetails(studyId: string, seriesId: string): Promise<ApiResponse<DicomSeries>> {
    return apiClient.get(`/dicom/studies/${studyId}/series/${seriesId}`);
  },

  // Instances
  async getInstances(studyId: string, seriesId: string): Promise<ApiResponse<DicomInstance[]>> {
    return apiClient.get(`/dicom/studies/${studyId}/series/${seriesId}/instances`);
  },

  async getInstance(studyId: string, seriesId: string, instanceId: string): Promise<ApiResponse<DicomInstance>> {
    return apiClient.get(`/dicom/studies/${studyId}/series/${seriesId}/instances/${instanceId}`);
  },

  async getInstancePixelData(instanceId: string): Promise<ApiResponse<ArrayBuffer>> {
    return apiClient.get(`/dicom/instances/${instanceId}/pixels`);
  },

  // Segmentation
  async processSegmentation(studyId: string, seriesId?: string): Promise<ApiResponse<SegmentationResult>> {
    return apiClient.post('/dicom/segmentation', { studyId, seriesId });
  },

  async getSegmentationStatus(jobId: string): Promise<ApiResponse<{ status: string; progress: number }>> {
    return apiClient.get(`/dicom/segmentation/${jobId}/status`);
  },

  // WADO-RS endpoints for streaming
  getInstanceWadoUrl(studyId: string, seriesId: string, instanceId: string): string {
    return `/api/wado-rs/studies/${studyId}/series/${seriesId}/instances/${instanceId}`;
  },

  getThumbnailUrl(studyId: string, seriesId: string): string {
    return `/api/wado-rs/studies/${studyId}/series/${seriesId}/thumbnail`;
  },
};
