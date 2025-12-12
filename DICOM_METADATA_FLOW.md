# DICOM Metadata Flow Guide

## Overview
This guide explains how DICOM metadata flows from upload through the entire MedTech pipeline to enable context-aware chat and report generation with BioMistral.

---

## Step-by-Step Flow

### 1. DICOM File Upload

**User Action**: Upload a `.dcm` file via UploadPage

**Frontend (`UploadPage.tsx` → `dicomStore.ts`)**:
```typescript
const formData = new FormData();
formData.append('file', file);

const response = await fetch('http://localhost:8000/process-upload', {
  method: 'POST',
  body: formData,
});
```

---

### 2. Backend Processing

**API Server (`api_server.py` → `main.py`)**:

```python
# Saves file to data/uploads/
file_path = uploads_dir / file.filename

# Processes through pipeline
result = pipeline.process_medical_image(process_path, str(uploads_dir))
```

**Pipeline Components**:
1. **Segmentation** (`segmentation_pipeline.py`)
   - Loads DICOM using medical image preprocessor
   - Runs SwinUNETR model for organ segmentation
   - Saves outputs:
     - `{filename}_mask.nii.gz`: Segmentation mask
     - `{filename}_embeddings.npy`: Feature vectors
     - `{filename}_metadata.json`: Processing metadata

2. **Vector Store** (`vector_store/faiss_index.py`)
   - Indexes embeddings in FAISS
   - Enables similarity search

3. **Similar Case Retrieval** (`vector_store/retrieval_engine.py`)
   - Queries vector store with new embeddings
   - Returns top-k similar cases

4. **Optional Report Generation** (`llm/medical_llm.py`)
   - Uses BioMistral to generate preliminary findings

---

### 3. Backend Response

**Response JSON Structure**:
```json
{
  "status": "completed",
  "image_path": "data/uploads/patient_scan.dcm",
  "components": {
    "segmentation": {
      "image_path": "data/uploads/patient_scan.dcm",
      "mask_path": "data/uploads/patient_scan_mask.nii.gz",
      "embedding_path": "data/uploads/patient_scan_embeddings.npy",
      "metadata_path": "data/uploads/patient_scan_metadata.json",
      "status": "success",
      "metrics": {
        "num_classes": 5,
        "volumes": {
          "1": 1250.5,  // liver volume in cc
          "2": 180.2,   // kidney volume
          "3": 95.3     // spleen volume
        }
      }
    },
    "similar_cases": [
      {
        "metadata": {
          "image_path": "data/uploads/ID_0a4674064.dcm",
          "segmentation_metrics": {...}
        },
        "similarity": 0.94
      },
      // ... more similar cases
    ],
    "medical_report": {
      "report": "Preliminary findings...",
      "generation_timestamp": "2025-12-12T00:15:20"
    }
  }
}
```

---

### 4. Frontend State Update

**dicomStore (`dicomStore.ts`)**:

```typescript
// Extract from response
const segmentationResult = data.components?.segmentation;
const similarCases = data.components?.similar_cases || [];
const medicalReport = data.components?.medical_report;

// Update dicomStore
set({ 
  currentStudy: mockStudy,
  seriesList: [mockSeries],
  instanceList,
  currentSeries: mockSeries,
  currentInstance: instanceList[0],
});
```

**chatStore Update**:
```typescript
useChatStore.getState().setContext({
  studyId: mockStudy.studyId,
  image_path: file.name,
  patient_id: mockStudy.patientId,
  patient_name: mockStudy.patientName,
  study_date: mockStudy.studyDate,
  modality: mockStudy.modality,
  segmentation: segmentationResult,      // Full segmentation results
  similar_cases: similarCases,           // Array of similar cases
  report: medicalReport,                 // Optional preliminary report
  metadata_path: segmentationResult?.metadata_path,
  mask_path: segmentationResult?.mask_path,
  embedding_path: segmentationResult?.embedding_path,
});
```

**reportStore Update**:
```typescript
useReportStore.getState().setStudyContext({
  studyId: mockStudy.studyId,
  patientName: mockStudy.patientName,
  patientId: mockStudy.patientId,
  studyDate: mockStudy.studyDate,
  modality: mockStudy.modality,
  studyDescription: mockStudy.studyDescription,
  segmentationResult,    // Segmentation with metrics
  similarCases,          // Similar case references
});
```

---

### 5. Chat Usage

**User Action**: Navigate to Chat page, ask a question

**Frontend (`ChatPage.tsx` → `chatStore.ts`)**:

```typescript
// Build request with context
const payload = {
  message: content,
  session_id: sessionId || `session-${Date.now()}`,
  scan_context: currentContext,  // Contains all DICOM metadata
  conversation_history: history,
  stream: true
};

// Send to backend
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
```

**Backend (`api_server.py` /chat endpoint)**:

```python
# Extract context
segmentation_results = request.scan_context.get("segmentation")
similar_cases = request.scan_context.get("similar_cases", [])

# Build comprehensive context  
context = context_builder.build_context(
    segmentation_results=segmentation_results,
    similar_cases=similar_cases,
    clinical_context=clinical_context,
    conversation_history=request.conversation_history,
)

# Generate response with BioMistral
response_text = pipeline.llm.answer_question(
    request.message, 
    context, 
    stream=True
)
```

**Context Builder (`utils/context_builder.py`)**:
```python
def build_context(self, segmentation_results, similar_cases, ...):
    context = {
        "segmentation_findings": self._format_segmentation(segmentation_results),
        "similar_cases": self._format_similar_cases(similar_cases),
        "clinical_context": clinical_context,
    }
    return context
```

**Example BioMistral Prompt**:
```
Based on the medical imaging data provided, answer the following question:

Question: What organs were segmented and what are their volumes?

Segmentation Findings:
- Liver: 1250.5 cc
- Right Kidney: 180.2 cc
- Spleen: 95.3 cc

Similar Cases Context:
Case 1 (similarity: 0.94): data/uploads/ID_0a4674064.dcm
  - Similar segmentation volumes
  - Same modality (CT)

Clinical Context:
- Modality: CT
- Body Region: Abdomen
- Study Date: 2025-12-12

Answer: [BioMistral generates response]
```

---

### 6. Report Generation

**User Action**: Navigate to Reports page, click "Generate Report"

**Frontend (`ReportsPage.tsx` → `reportStore.ts`)**:

```typescript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Generate a comprehensive medical report based on the uploaded imaging study',
    session_id: `report-${studyId}`,
    scan_context: currentStudyContext,  // Full DICOM metadata
    stream: false,
  }),
});

const data = await response.json();
const reportText = data.response;
```

**Report Parsing**:
```typescript
// Parse report into sections
const sectionPatterns = [
  { title: 'Clinical History', pattern: /Clinical History:?\s*/i },
  { title: 'Technique', pattern: /Technique:?\s*/i },
  { title: 'Findings', pattern: /Findings:?\s*/i },
  { title: 'Impression', pattern: /Impression:?\s*/i },
];

// Extract each section from LLM response
for (const { title, pattern } of sectionPatterns) {
  // Find section boundaries and extract content
  // ...
}
```

**BioMistral Report Prompt**:
```
Generate a comprehensive medical report based on the uploaded imaging study.

Patient: Demo Patient
Study Date: 2025-12-12
Modality: CT
Study Description: Uploaded Study

Segmentation Findings:
- Detected organs: Liver, Kidneys, Spleen
- Liver volume: 1250.5 cc
- Right Kidney: 180.2 cc
- Spleen: 95.3 cc

Similar Cases:
- 3 similar cases found in database
- Average similarity: 0.91
- Common findings: Normal organ volumes

Please provide:
1. Clinical History
2. Technique
3. Comparison
4. Findings
5. Impression

Report: [BioMistral generates structured report]
```

---

## Data Persistence

### Frontend State (In-Memory)
- **dicomStore**: Current study, series, instances
- **chatStore**: Messages, context
- **reportStore**: Current report, study context

### Backend Filesystem
```
data/uploads/
├── patient_scan.dcm              # Original DICOM
├── patient_scan_metadata.json    # Segmentation metadata
├── patient_scan_mask.nii.gz      # Segmentation mask (compressed)
└── patient_scan_embeddings.npy   # Feature vectors for similarity search
```

### Vector Database (FAISS)
- **Location**: `data/vector_store/faiss_index.bin`
- **Metadata**: `data/vector_store/metadata.pkl`
- **Purpose**: Fast similarity search for case retrieval

---

## Key Integration Points

### 1. Upload → Stores
```
UploadPage → dicomStore.uploadDicom()
  ↓
chatStore.setContext()
  ↓
reportStore.setStudyContext()
```

### 2. Chat → Backend
```
ChatPage → chatStore.sendChatMessage()
  ↓
POST /chat with scan_context
  ↓
BioMistral with full context
```

### 3. Report → Backend
```
ReportsPage → reportStore.generateReport()
  ↓
POST /chat with scan_context
  ↓
Parse BioMistral response into sections
```

---

## Context Object Structure

**Complete Context Available to BioMistral**:

```typescript
{
  // Study identifiers
  studyId: string,
  image_path: string,
  patient_id: string,
  patient_name: string,
  study_date: string,
  modality: string,
  
  // Segmentation results
  segmentation: {
    image_path: string,
    mask_path: string,
    embedding_path: string,
    metadata_path: string,
    metrics: {
      num_classes: number,
      volumes: { [classId: string]: number },
      dice_scores?: { [classId: string]: number }
    }
  },
  
  // Similar cases from vector store
  similar_cases: [
    {
      metadata: {
        image_path: string,
        segmentation_metrics: {...}
      },
      similarity: number  // 0-1 score
    }
  ],
  
  // Optional preliminary report
  report?: {
    report: string,
    generation_timestamp: string
  },
  
  // File paths for reference
  metadata_path: string,
  mask_path: string,
  embedding_path: string
}
```

---

## Troubleshooting Context Flow

### Issue: Chat has no context
**Check**:
1. Console logs in `dicomStore.uploadDicom()`
2. Verify `chatStore.currentContext` after upload
3. Check `/chat` request payload includes `scan_context`

### Issue: Reports are generic
**Check**:
1. `reportStore.currentStudyContext` is populated
2. Backend receives `scan_context` in request
3. BioMistral prompt includes segmentation findings

### Issue: Similar cases not showing
**Check**:
1. Vector store is initialized: `/status` endpoint
2. Embeddings were indexed during upload
3. `similar_cases` array in response is not empty

---

## Testing Context Flow

### 1. Upload Test
```bash
# Watch backend logs
python api_server.py

# Upload DICOM via frontend
# Check logs for:
# - "Processing uploaded file"
# - "Step 1: Performing segmentation"
# - "Step 3: Retrieving similar cases"
```

### 2. Context Inspection
```javascript
// In browser console after upload
const chatContext = useChatStore.getState().currentContext;
console.log('Chat Context:', chatContext);

const reportContext = useReportStore.getState().currentStudyContext;
console.log('Report Context:', reportContext);
```

### 3. Chat Test
```
User: "What organs were detected?"

Expected response includes:
- Reference to segmented organs (liver, kidney, spleen, etc.)
- Mention of volumes if available
- Context from similar cases if relevant
```

### 4. Report Test
```
Click "Generate Report"

Expected:
- API call with full scan_context
- BioMistral generates medical report
- Report parsed into sections
- Sections display patient/study info
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          FRONTEND                                │
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │ UploadPage │───▶│ dicomStore │───▶│ chatStore  │            │
│  └────────────┘    └─────┬──────┘    └────────────┘            │
│                           │                                      │
│                           ▼                                      │
│                    ┌─────────────┐                              │
│                    │ reportStore │                              │
│                    └─────────────┘                              │
│                                                                  │
│  ┌────────────┐                       ┌────────────┐            │
│  │  ChatPage  │────────────────────▶│ ReportsPage │            │
│  └────────────┘  (uses context)     └────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                          │ HTTP API
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                          BACKEND                                 │
│                                                                  │
│  ┌───────────────┐    ┌─────────────────┐   ┌──────────────┐   │
│  │  API Server   │───▶│ MedTech Pipeline│──▶│   BioMistral │   │
│  │ (FastAPI)     │    │                 │   │     LLM      │   │
│  └───────────────┘    └────────┬────────┘   └──────────────┘   │
│                                 │                                │
│                    ┌────────────┼────────────┐                  │
│                    ▼            ▼            ▼                  │
│          ┌──────────────┐  ┌─────────┐  ┌────────────┐         │
│          │ Segmentation │  │ Vector  │  │  Context   │         │
│          │   Pipeline   │  │  Store  │  │  Builder   │         │
│          └──────────────┘  └─────────┘  └────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────┐
         │    Persistent Storage            │
         │  - DICOM files (.dcm)           │
         │  - Segmentation masks (.nii.gz) │
         │  - Embeddings (.npy)            │
         │  - Metadata (.json)             │
         │  - FAISS index (.bin)           │
         └─────────────────────────────────┘
```

---

**Author**: Antigravity AI  
**Last Updated**: 2025-12-12  
**Related Docs**: BIOMISTRAL_INTEGRATION.md
