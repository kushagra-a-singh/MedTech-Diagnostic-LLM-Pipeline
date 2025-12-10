import { apiClient, ApiResponse, PaginatedResponse } from './index';
import type { Report, ReportTemplate, ReportSection } from '@/types';

export interface ReportListParams {
  page?: number;
  pageSize?: number;
  status?: 'draft' | 'pending_review' | 'finalized';
  studyId?: string;
  dateFrom?: string;
  dateTo?: string;
}

export interface GenerateReportRequest {
  studyId: string;
  templateId: string;
  additionalContext?: string;
}

export interface GenerateReportResponse {
  report: Report;
  confidence: number;
  modelVersion: string;
}

export const reportsApi = {
  // Reports
  async getReports(params?: ReportListParams): Promise<ApiResponse<PaginatedResponse<Report>>> {
    return apiClient.get('/reports', params as Record<string, string>);
  },

  async getReport(reportId: string): Promise<ApiResponse<Report>> {
    return apiClient.get(`/reports/${reportId}`);
  },

  async generateReport(request: GenerateReportRequest): Promise<ApiResponse<GenerateReportResponse>> {
    return apiClient.post('/reports/generate', request);
  },

  async saveReport(reportId: string, sections: ReportSection[]): Promise<ApiResponse<Report>> {
    return apiClient.put(`/reports/${reportId}`, { sections });
  },

  async updateReportStatus(reportId: string, status: string): Promise<ApiResponse<Report>> {
    return apiClient.patch(`/reports/${reportId}/status`, { status });
  },

  async deleteReport(reportId: string): Promise<ApiResponse<void>> {
    return apiClient.delete(`/reports/${reportId}`);
  },

  async exportReportPdf(reportId: string): Promise<ApiResponse<Blob>> {
    return apiClient.get(`/reports/${reportId}/export/pdf`);
  },

  // Templates
  async getTemplates(): Promise<ApiResponse<ReportTemplate[]>> {
    return apiClient.get('/reports/templates');
  },

  async getTemplate(templateId: string): Promise<ApiResponse<ReportTemplate>> {
    return apiClient.get(`/reports/templates/${templateId}`);
  },

  async createTemplate(template: Omit<ReportTemplate, 'id'>): Promise<ApiResponse<ReportTemplate>> {
    return apiClient.post('/reports/templates', template);
  },

  async updateTemplate(templateId: string, template: Partial<ReportTemplate>): Promise<ApiResponse<ReportTemplate>> {
    return apiClient.put(`/reports/templates/${templateId}`, template);
  },

  async deleteTemplate(templateId: string): Promise<ApiResponse<void>> {
    return apiClient.delete(`/reports/templates/${templateId}`);
  },
};
