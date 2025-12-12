#!/bin/bash

# Quick verification script for BioMistral integration

echo "========================================="
echo "BioMistral Integration Verification"
echo "========================================="
echo ""

# Check LLM config
echo "1. Checking LLM configuration..."
if grep -q "model_type: biomistral" configs/llm_config.yaml; then
    echo "   ✓ Model type set to biomistral"
else
    echo "   ✗ Model type NOT set to biomistral"
fi

if grep -q "model_name: BioMistral/BioMistral-7B" configs/llm_config.yaml; then
    echo "   ✓ BioMistral model name configured"
else
    echo "   ✗ BioMistral model name NOT configured"
fi

echo ""

# Check if uploads directory has DICOM files
echo "2. Checking DICOM uploads directory..."
if [ -d "data/uploads" ]; then
    dicom_count=$(find data/uploads -name "*.dcm" | wc -l)
    metadata_count=$(find data/uploads -name "*_metadata.json" | wc -l)
    mask_count=$(find data/uploads -name "*_mask.nii.gz" | wc -l)
    embedding_count=$(find data/uploads -name "*_embeddings.npy" | wc -l)
    
    echo "   Found:"
    echo "   - $dicom_count DICOM files"
    echo "   - $metadata_count metadata files"
    echo "   - $mask_count segmentation masks"
    echo "   - $embedding_count embedding files"
else
    echo "   ✗ data/uploads directory not found"
fi

echo ""

# Check frontend store files
echo "3. Checking frontend store updates..."
if grep -q "setStudyContext" new-frontend/src/lib/store/reportStore.ts; then
    echo "   ✓ reportStore has setStudyContext"
else
    echo "   ✗ reportStore missing setStudyContext"
fi

if grep -q "currentStudyContext" new-frontend/src/types/index.ts; then
    echo "   ✓ Types include currentStudyContext"
else
    echo "   ✗ Types missing currentStudyContext"
fi

echo ""

# Sample DICOM metadata check
echo "4. Checking sample DICOM metadata..."
if [ -f "data/uploads/ID_0a4674064_metadata.json" ]; then
    echo "   ✓ Sample metadata exists"
    echo "   Sample content:"
    cat data/uploads/ID_0a4674064_metadata.json | head -15
else
    echo "   ℹ No sample metadata file found (upload a DICOM to create)"
fi

echo ""
echo "========================================="
echo "Verification Complete"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Start backend: python api_server.py"
echo "2. Start frontend: cd new-frontend && npm run dev"
echo "3. Upload a DICOM file via the Upload page"
echo "4. Test chat with the uploaded study context"
echo "5. Generate a report using the Reports page"
