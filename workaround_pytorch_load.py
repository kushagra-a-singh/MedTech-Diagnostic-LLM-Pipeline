#!/usr/bin/env python3
"""
Temporary workaround for PyTorch < 2.6 to load .bin files.
This patches torch.load to bypass the version check.

WARNING: This bypasses a security fix. Use only if you trust the model files.
"""

import torch
import warnings

# Store original torch.load
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    """
    Patched version of torch.load that bypasses version check.
    """
    # Remove weights_only if present (it's the new security feature)
    kwargs.pop('weights_only', None)
    
    # Call original torch.load with the internal _load function
    # This bypasses the version check
    try:
        # Try to use the internal _load method which doesn't have the check
        if hasattr(torch.serialization, '_load'):
            return torch.serialization._load(
                f, map_location, pickle_module, **kwargs
            )
        else:
            # Fallback: use original but with weights_only=False
            return _original_torch_load(f, map_location, pickle_module, weights_only=False, **kwargs)
    except Exception:
        # Final fallback: use original
        return _original_torch_load(f, map_location, pickle_module, **kwargs)

def apply_workaround():
    """Apply the workaround patch."""
    torch.load = _patched_torch_load
    warnings.warn(
        "⚠️  PyTorch load workaround applied. This bypasses security restrictions. "
        "Only use with trusted model files. Upgrade to PyTorch 2.6+ when available.",
        UserWarning,
        stacklevel=2
    )
    print("✅ Workaround applied: torch.load patched to allow .bin file loading")

def remove_workaround():
    """Remove the workaround patch."""
    torch.load = _original_torch_load
    print("✅ Workaround removed: restored original torch.load")

if __name__ == "__main__":
    apply_workaround()
    print("\nTo use this workaround in your code, add at the top of your script:")
    print("  from workaround_pytorch_load import apply_workaround")
    print("  apply_workaround()")

