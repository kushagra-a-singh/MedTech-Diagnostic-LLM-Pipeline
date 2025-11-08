#!/bin/bash

# Script to fix conda environment freezing issues
# Usage: ./fix_conda_freezing.sh

set -e

echo "🔧 Fixing Conda Freezing Issues..."
echo ""

# 1. Kill stuck conda processes
echo "1. Checking for stuck conda processes..."
CONDA_PIDS=$(ps aux | grep -E "(conda|libmamba)" | grep -v grep | grep -v "fix_conda_freezing" | awk '{print $2}' || true)
if [ ! -z "$CONDA_PIDS" ]; then
    echo "   Found conda processes: $CONDA_PIDS"
    echo "   Killing stuck processes..."
    echo "$CONDA_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
else
    echo "   ✓ No stuck processes found"
fi

# 2. Remove lock files
echo ""
echo "2. Removing lock files..."
find ~/.conda -name "*.lock" -delete 2>/dev/null || true
find ~/miniconda3 -name "*.lock" -delete 2>/dev/null || true
find ~/anaconda3 -name "*.lock" -delete 2>/dev/null || true
echo "   ✓ Lock files removed"

# 3. Clear conda cache
echo ""
echo "3. Clearing conda cache..."
conda clean --all -y || {
    echo "   ⚠️  Warning: Could not clean all caches, trying individual caches..."
    conda clean --packages -y || true
    conda clean --index-cache -y || true
    conda clean --tarballs -y || true
}
echo "   ✓ Cache cleared"

# 4. Update conda
echo ""
echo "4. Updating conda..."
conda update -n base conda -y || {
    echo "   ⚠️  Warning: Could not update conda (this is okay if already up to date)"
}

# 5. Update solver
echo ""
echo "5. Updating conda-libmamba-solver..."
conda update -n base -c conda-forge conda-libmamba-solver -y || {
    echo "   ⚠️  Warning: Could not update solver"
}

# 6. Check conda configuration
echo ""
echo "6. Checking conda configuration..."
if [ -f ~/.condarc ]; then
    echo "   Current .condarc settings:"
    cat ~/.condarc | grep -v "^#" | grep -v "^$" || echo "   (empty or commented out)"
else
    echo "   No .condarc file found (using defaults)"
fi

echo ""
echo "✅ Conda freezing fix completed!"
echo ""
echo "Next steps:"
echo "1. Try your conda command again"
echo "2. If it still freezes, try: conda install mamba -n base -c conda-forge"
echo "3. Then use 'mamba' instead of 'conda' for faster operations"
echo "4. If problems persist, see CONDA_TROUBLESHOOTING.md for more solutions"

