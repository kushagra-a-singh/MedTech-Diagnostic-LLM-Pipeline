# Conda Environment Freezing - Troubleshooting Guide

## Quick Fixes

### 1. Clear Conda Cache
```bash
# Clean all caches
conda clean --all -y

# Clear package cache only
conda clean --packages -y

# Clear index cache
conda clean --index-cache -y
```

### 2. Kill Stuck Conda Processes
```bash
# Find stuck conda processes
ps aux | grep -E "(conda|libmamba)" | grep -v grep

# Kill specific process (replace PID)
kill -9 <PID>

# Or kill all conda processes
pkill -9 conda
pkill -9 libmamba
```

### 3. Remove Lock Files (if any)
```bash
# Find and remove lock files
find ~/.conda -name "*.lock" -delete
find ~/miniconda3 -name "*.lock" -delete
find ~/anaconda3 -name "*.lock" -delete 2>/dev/null
```

### 4. Use Mamba for Faster Dependency Resolution
```bash
# Install mamba (faster solver)
conda install mamba -n base -c conda-forge

# Use mamba instead of conda
mamba install <package>
# or
mamba env create -f environment.yml
```

### 5. Update Conda
```bash
# Update conda itself
conda update -n base conda -y
conda update -n base -c conda-forge conda-libmamba-solver -y
```

### 6. Fix Network/Proxy Issues
```bash
# Check if you're behind a proxy
echo $HTTP_PROXY
echo $HTTPS_PROXY

# If needed, configure conda to use proxy
conda config --set proxy_servers.http http://proxy:port
conda config --set proxy_servers.https https://proxy:port

# Or remove proxy settings
conda config --remove-key proxy_servers.http
conda config --remove-key proxy_servers.https
```

### 7. Use Conda-Forge Channel Only
```bash
# Remove default channels and use conda-forge only
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### 8. Increase Solver Timeout
```bash
# Edit condarc file
nano ~/.condarc

# Add these settings:
solver_timeout: 600
channels:
  - conda-forge
channel_priority: strict
```

### 9. Create Environment with Specific Solver
```bash
# Use libmamba solver explicitly
CONDA_ALWAYS_YES=1 conda create -n myenv python=3.10 --solver=libmamba

# Or use classic solver (slower but more stable)
conda create -n myenv python=3.10 --solver=classic
```

### 10. Manual Environment Creation
If all else fails, create environment manually:
```bash
# Create empty environment
conda create -n medai python=3.10 -y

# Activate it
conda activate medai

# Install packages one by one or in small groups
pip install -r requirements.txt
```

## Prevention Tips

1. **Always activate environment before installing packages**
   ```bash
   conda activate medai
   conda install <package>
   ```

2. **Use specific package versions** to avoid complex dependency resolution
   ```bash
   conda install numpy=1.24.0 pandas=2.0.0 -y
   ```

3. **Install packages in small batches** rather than all at once

4. **Use pip for packages not available in conda** (but be careful with dependencies)

5. **Keep conda updated**
   ```bash
   conda update -n base conda -y
   ```

## If Nothing Works

1. **Reinstall Miniconda/Anaconda**
   ```bash
   # Backup your environments
   conda env export -n medai > medai_backup.yml
   
   # Uninstall and reinstall conda
   # Then recreate environment from backup
   conda env create -f medai_backup.yml
   ```

2. **Use Docker instead** - More isolated and predictable
3. **Use venv instead of conda** for this project if conda continues to cause issues

