# Fix PyTorch Version Issue (CVE-2025-32434)

## Problem

PyTorch 2.5.1 and earlier versions cannot load `.bin` model files due to a security vulnerability (CVE-2025-32434). PyTorch 2.6+ is required, OR you can use safetensors format which doesn't have this restriction.

**Error Message:**
```
Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, 
we now require users to upgrade torch to at least v2.6 in order to use the function. 
This version restriction does not apply when loading files with safetensors.
```

## Solutions

### Solution 1: Use Safetensors Format (Recommended)

The easiest solution is to ensure your model is in safetensors format. You have two options:

#### Option A: Re-download Model in Safetensors Format

1. **Backup your current model** (optional):
   ```bash
   mv models/biomistral models/biomistral_backup
   ```

2. **Download model in safetensors format**:
   ```bash
   python download_model_safetensors.py BioMistral/BioMistral-7B models/biomistral
   ```

   Or manually using huggingface_hub:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="BioMistral/BioMistral-7B",
       local_dir="models/biomistral",
       local_dir_use_symlinks=False,
       ignore_patterns=["*.bin"]  # Skip .bin files
   )
   ```

#### Option B: Convert Existing Model (Requires PyTorch 2.6+)

If you can temporarily upgrade PyTorch to 2.6+:

1. **Upgrade PyTorch**:
   ```bash
   pip install --upgrade torch>=2.6.0
   ```

2. **Convert the model**:
   ```bash
   python convert_to_safetensors.py models/biomistral
   ```

3. **Optionally downgrade PyTorch back** (if needed for other dependencies):
   ```bash
   pip install torch==2.5.1
   ```

### Solution 2: Upgrade PyTorch to 2.6+

If PyTorch 2.6+ is available and compatible with your other dependencies:

```bash
pip install --upgrade torch>=2.6.0 torchvision torchaudio
```

**Note:** PyTorch 2.6 might not be available yet. Check: https://pytorch.org/get-started/locally/

### Solution 3: Use Alternative Model Format

Some models on HuggingFace Hub are available in multiple formats. Check if safetensors are available:

```bash
# Check available files
huggingface-cli repo files BioMistral/BioMistral-7B
```

## Verification

After applying a solution, verify the model loads correctly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("models/biomistral")
print("✅ Model loaded successfully!")
```

## Why Safetensors?

- **Security**: Safetensors format is safer and doesn't execute arbitrary code
- **Compatibility**: Works with all PyTorch versions
- **Performance**: Faster loading and saving
- **Future-proof**: Recommended format by HuggingFace

## Additional Notes

- The code has been updated to automatically detect and prefer safetensors files
- If safetensors are found, they will be used automatically
- Error messages now provide clear guidance on how to fix the issue

## References

- [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)
- [Safetensors Documentation](https://huggingface.co/docs/safetensors)
- [PyTorch Release Notes](https://pytorch.org/blog/pytorch-2-6-release/)

