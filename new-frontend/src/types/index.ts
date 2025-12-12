// Authentication Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'radiologist' | 'technician' | 'admin';
  avatar?: string;
  createdAt: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  token: string | null;
  isLoading: boolean;
}

// DICOM Types
export interface DicomStudy {
  studyId: string;
  patientId: string;
  patientName: string;
  studyDate: string;
  studyDescription: string;
  modality: string;
  seriesCount: number;
  instanceCount: number;
  accessionNumber?: string;
  referringPhysician?: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
}

export interface DicomSeries {
  seriesId: string;
  seriesNumber: number;
  seriesDescription: string;
  modality: string;
  instanceCount: number;
  bodyPart?: string;
}

export interface DicomInstance {
  instanceId: string;
  instanceNumber: number;
  sopClassUid: string;
  imageUrl: string;
  rows: number;
  columns: number;
  windowCenter?: number;
  windowWidth?: number;
}

export interface SegmentationOverlay {
  id: string;
  name: string;
  color: string;
  opacity: number;
  visible: boolean;
  maskUrl: string;
  findings?: string[];
}

export interface ViewportConfig {
  zoom: number;
  pan: { x: number; y: number };
  windowCenter: number;
  windowWidth: number;
  rotation: number;
  flipH: boolean;
  flipV: boolean;
  invert: boolean;
}

export interface Measurement {
  id: string;
  type: 'length' | 'angle' | 'roi' | 'ellipse';
  points: { x: number; y: number }[];
  value: number;
  unit: string;
  label?: string;
}

export interface DicomState {
  currentStudy: DicomStudy | null;
  currentSeries: DicomSeries | null;
  currentInstance: DicomInstance | null;
  seriesList: DicomSeries[];
  instanceList: DicomInstance[];
  segmentation: SegmentationOverlay[];
  viewportConfig: ViewportConfig;
  measurements: Measurement[];
  isLoading: boolean;
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  attachments?: ChatAttachment[];
  isStreaming?: boolean;
}

export interface ChatAttachment {
  id: string;
  type: 'study' | 'image' | 'report';
  referenceId: string;
  name: string;
}

export interface ChatSession {
  id: string;
  title: string;
  studyId?: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface ChatState {
  messages: ChatMessage[];
  isGenerating: boolean;
  sessionId: string | null;
  currentContext: {
    studyId?: string;
    reportId?: string;
    image_path?: string;
    patient_id?: string;
    patient_name?: string;
    study_date?: string;
    modality?: string;
    segmentation?: any;
    similar_cases?: any[];
    report?: any;
    metadata_path?: string;
    mask_path?: string;
    embedding_path?: string;
  };
}

// Report Types
export interface ReportSection {
  id: string;
  title: string;
  content: string;
  order: number;
  isAIGenerated: boolean;
  isEdited: boolean;
}

export interface Report {
  id: string;
  studyId: string;
  templateId?: string;
  title: string;
  sections: ReportSection[];
  status: 'draft' | 'pending_review' | 'finalized';
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  signedBy?: string;
  signedAt?: string;
}

export interface ReportTemplate {
  id: string;
  name: string;
  modality: string;
  sections: Omit<ReportSection, 'id' | 'content' | 'isAIGenerated' | 'isEdited'>[];
}

export interface ReportState {
  currentReport: Report | null;
  reportTemplates: ReportTemplate[];
  isGenerating: boolean;
  isSaving: boolean;
  currentStudyContext: any;
}

// History Types
export interface HistoryItem {
  id: string;
  sessionId: string;
  studyId: string;
  patientName: string;
  studyDate: string;
  modality: string;
  status: 'in_progress' | 'completed' | 'reviewed';
  reportId?: string;
  createdAt: string;
  updatedAt: string;
}

// Feedback Types
export interface ReportFeedback {
  id: string;
  reportId: string;
  originalText: string;
  correctedText: string;
  sectionId: string;
  comment?: string;
  submittedAt: string;
  submittedBy: string;
}

export interface SegmentationFeedback {
  id: string;
  studyId: string;
  overlayId: string;
  corrections: { x: number; y: number; action: 'add' | 'remove' }[];
  comment?: string;
  submittedAt: string;
  submittedBy: string;
}

export interface ChatFeedback {
  id: string;
  messageId: string;
  sessionId: string;
  rating: 'positive' | 'negative';
  comment?: string;
  submittedAt: string;
}

export interface FeedbackState {
  pendingFeedback: {
    report: ReportFeedback[];
    segmentation: SegmentationFeedback[];
    chat: ChatFeedback[];
  };
  feedbackHistory: {
    report: ReportFeedback[];
    segmentation: SegmentationFeedback[];
    chat: ChatFeedback[];
  };
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}
