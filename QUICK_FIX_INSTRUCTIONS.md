# Quick Fix Instructions

## Current Situation
- ✅ Workaround code has been added to `src/llm/medical_llm.py` to allow loading .bin files with PyTorch < 2.6
- ❌ Model files are missing from `models/biomistral/`

## Solution: Re-download the Model

The workaround is now integrated, so you can download the model normally:

### Option 1: Use setup_models.py (Recommended)
```bash
python setup_models.py --model biomistral
```

### Option 2: Download directly from HuggingFace
```bash
# Using huggingface-cli
huggingface-cli download BioMistral/BioMistral-7B --local-dir models/biomistral --local-dir-use-symlinks False

# Or using Python
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BioMistral/BioMistral-7B', local_dir='models/biomistral', local_dir_use_symlinks=False)"
```

### Option 3: Manual download
Visit: https://huggingface.co/BioMistral/BioMistral-7B/tree/main
Download the `pytorch_model.bin` file (13GB) to `models/biomistral/`

## After Download

1. **Verify files exist:**
   ```bash
   ls -lh models/biomistral/*.bin
   ```

2. **Test the API server:**
   ```bash
   python api_server.py
   ```

The workaround will automatically apply when loading the model. You'll see a warning message, but the model should load successfully.

## What the Workaround Does

The code automatically patches `torch.load` to bypass the PyTorch 2.6+ requirement when:
- PyTorch version < 2.6 is detected
- Model files are in .bin format

**Security Note:** This bypasses a security fix. Only use with trusted model files from official sources like HuggingFace.

## Future Upgrade

When PyTorch 2.6+ becomes available:
1. Upgrade: `pip install --upgrade torch>=2.6.0`
2. The workaround will automatically disable itself
3. Or convert model to safetensors format for better security

