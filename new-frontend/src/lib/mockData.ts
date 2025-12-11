import type { DicomStudy, DicomSeries, HistoryItem, Report, ReportTemplate, ChatMessage } from '@/types';

// Realistic patient names
const patientNames = [
  'Maria Garcia', 'James Johnson', 'Wei Chen', 'Sarah Williams', 
  'Michael Brown', 'Emily Davis', 'Raj Patel', 'Jessica Martinez',
  'David Anderson', 'Lisa Thompson', 'Ahmed Hassan', 'Jennifer Wilson',
];

// Realistic study descriptions
const ctDescriptions = [
  'CT Chest with Contrast', 'CT Abdomen/Pelvis', 'CT Head without Contrast',
  'CT Cervical Spine', 'CT Pulmonary Angiogram', 'CT Coronary Angiography',
];

const mriDescriptions = [
  'MRI Brain with/without Contrast', 'MRI Lumbar Spine', 'MRI Knee Right',
  'MRI Shoulder Left', 'MRI Cardiac', 'MRI Abdomen',
];

const xrayDescriptions = [
  'Chest PA/Lateral', 'Abdomen Series', 'Hand AP/Lateral', 
  'Ankle AP/Lateral', 'Knee AP/Lateral', 'Spine Cervical',
];

// Generate realistic mock studies
export function generateMockStudies(count: number = 20): DicomStudy[] {
  const studies: DicomStudy[] = [];
  const today = new Date();

  for (let i = 0; i < count; i++) {
    const modality = ['CT', 'MRI', 'XR'][Math.floor(Math.random() * 3)];
    const descriptions = modality === 'CT' ? ctDescriptions : modality === 'MRI' ? mriDescriptions : xrayDescriptions;
    const description = descriptions[Math.floor(Math.random() * descriptions.length)];
    
    const studyDate = new Date(today);
    studyDate.setDate(studyDate.getDate() - Math.floor(Math.random() * 30));

    studies.push({
      studyId: `study-${Date.now()}-${i}`,
      patientId: `P${String(10000 + i).padStart(6, '0')}`,
      patientName: patientNames[Math.floor(Math.random() * patientNames.length)],
      studyDate: studyDate.toISOString().split('T')[0],
      studyDescription: description,
      modality,
      seriesCount: Math.floor(Math.random() * 5) + 1,
      instanceCount: Math.floor(Math.random() * 200) + 50,
      accessionNumber: `ACC${String(Math.floor(Math.random() * 100000)).padStart(8, '0')}`,
      referringPhysician: `Dr. ${patientNames[Math.floor(Math.random() * patientNames.length)].split(' ')[1]}`,
      status: ['pending', 'processing', 'completed'][Math.floor(Math.random() * 3)] as DicomStudy['status'],
    });
  }

  return studies;
}

// Generate mock series for a study
export function generateMockSeries(studyId: string, count: number = 3): DicomSeries[] {
  const seriesTypes = ['Axial', 'Coronal', 'Sagittal', '3D Recon', 'MIP', 'Localizer'];
  const series: DicomSeries[] = [];

  for (let i = 0; i < count; i++) {
    series.push({
      seriesId: `${studyId}-series-${i}`,
      seriesNumber: i + 1,
      seriesDescription: seriesTypes[i % seriesTypes.length],
      modality: 'CT',
      instanceCount: Math.floor(Math.random() * 100) + 20,
      bodyPart: 'Chest',
    });
  }

  return series;
}

// Generate mock history items
export function generateMockHistory(count: number = 15): HistoryItem[] {
  const items: HistoryItem[] = [];
  const today = new Date();

  for (let i = 0; i < count; i++) {
    const modality = ['CT', 'MRI', 'XR'][Math.floor(Math.random() * 3)];
    const studyDate = new Date(today);
    studyDate.setDate(studyDate.getDate() - i);
    
    const createdAt = new Date(studyDate);
    createdAt.setHours(8 + Math.floor(Math.random() * 10));
    
    const updatedAt = new Date(createdAt);
    updatedAt.setHours(updatedAt.getHours() + Math.floor(Math.random() * 4));

    const status = ['in_progress', 'completed', 'reviewed'][Math.floor(Math.random() * 3)] as HistoryItem['status'];

    items.push({
      id: `history-${i}`,
      sessionId: `sess-${Date.now()}-${i}`,
      studyId: `study-${Date.now()}-${i}`,
      patientName: patientNames[Math.floor(Math.random() * patientNames.length)],
      studyDate: studyDate.toISOString().split('T')[0],
      modality,
      status,
      reportId: status !== 'in_progress' ? `report-${i}` : undefined,
      createdAt: createdAt.toISOString(),
      updatedAt: updatedAt.toISOString(),
    });
  }

  return items;
}

// Report templates
export const reportTemplates: ReportTemplate[] = [
  {
    id: 'ct-chest',
    name: 'CT Chest Standard',
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
    id: 'ct-abdomen',
    name: 'CT Abdomen/Pelvis',
    modality: 'CT',
    sections: [
      { title: 'Clinical History', order: 1 },
      { title: 'Technique', order: 2 },
      { title: 'Comparison', order: 3 },
      { title: 'Liver', order: 4 },
      { title: 'Biliary System', order: 5 },
      { title: 'Spleen', order: 6 },
      { title: 'Kidneys', order: 7 },
      { title: 'Bowel', order: 8 },
      { title: 'Lymph Nodes', order: 9 },
      { title: 'Impression', order: 10 },
    ],
  },
  {
    id: 'mri-brain',
    name: 'MRI Brain Standard',
    modality: 'MRI',
    sections: [
      { title: 'Clinical History', order: 1 },
      { title: 'Technique', order: 2 },
      { title: 'Comparison', order: 3 },
      { title: 'Gray/White Matter', order: 4 },
      { title: 'Ventricles', order: 5 },
      { title: 'Extra-Axial Spaces', order: 6 },
      { title: 'Posterior Fossa', order: 7 },
      { title: 'Impression', order: 8 },
    ],
  },
  {
    id: 'xr-chest',
    name: 'Chest X-Ray',
    modality: 'XR',
    sections: [
      { title: 'Clinical History', order: 1 },
      { title: 'Technique', order: 2 },
      { title: 'Comparison', order: 3 },
      { title: 'Lungs', order: 4 },
      { title: 'Heart/Mediastinum', order: 5 },
      { title: 'Bones', order: 6 },
      { title: 'Impression', order: 7 },
    ],
  },
];

// Mock chat conversations
export const mockChatMessages: ChatMessage[] = [
  {
    id: '1',
    role: 'user',
    content: 'What are the main findings in this chest CT?',
    timestamp: new Date(Date.now() - 300000).toISOString(),
  },
  {
    id: '2',
    role: 'assistant',
    content: `Based on the CT chest examination, here are the main findings:

**Lungs:**
- Clear lung fields bilaterally
- No focal consolidation or masses
- No pleural effusion

**Heart and Mediastinum:**
- Heart size within normal limits
- No pericardial effusion
- Mediastinal structures appear normal

**Additional Findings:**
- Minor degenerative changes in the visualized thoracic spine
- No lymphadenopathy

Would you like me to elaborate on any specific finding or generate a formal report?`,
    timestamp: new Date(Date.now() - 280000).toISOString(),
  },
  {
    id: '3',
    role: 'user',
    content: 'Can you compare this with the previous study from 2 weeks ago?',
    timestamp: new Date(Date.now() - 200000).toISOString(),
  },
  {
    id: '4',
    role: 'assistant',
    content: `**Comparison with Prior Study (dated 2 weeks ago):**

**Stable Findings:**
- Cardiac silhouette remains stable in size
- No new lung nodules identified
- Previously noted degenerative spine changes unchanged

**Interval Changes:**
- Mild improvement in the previously noted minimal right lower lobe atelectasis
- Resolution of the previously seen trace pleural fluid

**Overall Assessment:**
The current study shows overall stability compared to the prior examination with some mild improvement in the right lower lobe changes. No new concerning findings identified.`,
    timestamp: new Date(Date.now() - 180000).toISOString(),
  },
];
