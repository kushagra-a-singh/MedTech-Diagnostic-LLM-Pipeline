# Fix Next.js WSL Path Issues - Step-by-Step Guide

## Problem
Next.js is trying to run from Windows but access WSL filesystem, causing:
- UNC path errors (`\\wsl.localhost\...`)
- Permission errors trying to create `.next` in `C:\Windows\`
- Can't find `app` directory due to path resolution

## Solution Steps

### Step 1: Verify You're Running from WSL Terminal

**IMPORTANT**: You must run all commands from **WSL (Ubuntu) terminal**, NOT from Windows CMD or PowerShell.

1. Open **Ubuntu** (WSL) terminal:
   - Press `Win + R`, type `wsl` or `ubuntu`
   - Or search "Ubuntu" in Windows Start menu
   - Or use Windows Terminal and select "Ubuntu" profile

2. Verify you're in WSL:
   ```bash
   echo $SHELL
   # Should show: /bin/bash or similar (NOT cmd.exe)
   
   pwd
   # Should show: /home/samba/... (NOT C:\...)
   ```

### Step 2: Install Node.js and npm in WSL (if not installed)

```bash
# Check if Node.js is installed in WSL
node --version
npm --version

# If not installed, install using one of these methods:

# Option A: Using NodeSource (recommended)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Option B: Using nvm (Node Version Manager - better for multiple versions)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# Option C: Using apt (may have older version)
sudo apt update
sudo apt install -y nodejs npm

# Verify installation
node --version  # Should be 16+ (20+ recommended)
npm --version
```

### Step 3: Navigate to Project Directory in WSL

```bash
# Make sure you're using WSL paths (not Windows paths)
cd /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend

# Verify you're in the right place
pwd
# Should show: /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend
ls -la
# Should show: package.json, next.config.js, src/, etc.
```

### Step 4: Clean Previous Build Artifacts

```bash
# Remove any corrupted .next directory
rm -rf .next
rm -rf node_modules/.cache

# If there's a .next directory in wrong location (unlikely in WSL, but just in case)
rm -rf ~/.next
```

### Step 5: Verify App Directory Structure

```bash
# Check that app directory exists
ls -la src/app/
# Should show: page.tsx, layout.tsx, globals.css, etc.

# If app directory doesn't exist, Next.js won't work
# The structure should be:
# frontend/
#   src/
#     app/
#       page.tsx
#       layout.tsx
#       globals.css
```

### Step 6: Reinstall Dependencies (if needed)

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

# Or if you prefer yarn
# yarn install
```

### Step 7: Set Environment Variables (if needed)

```bash
# Create .env.local if it doesn't exist
cd /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend
touch .env.local

# Add this line (edit as needed):
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### Step 8: Run Next.js from WSL

```bash
# Make sure you're in the frontend directory
cd /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend

# Run dev server
npm run dev
```

**Expected output:**
```
▲ Next.js 15.5.4
- Local:        http://localhost:3000
- Network:      http://192.168.x.x:3000

✓ Ready in X seconds
```

### Step 9: Access from Browser

Open in browser:
- **From Windows**: `http://localhost:3000`
- **From WSL**: `http://localhost:3000` or the network IP shown

## Common Issues and Fixes

### Issue 1: "Command not found: node" or "Command not found: npm"

**Fix**: Node.js is not installed in WSL. Follow Step 2 above.

### Issue 2: "Couldn't find any `pages` or `app` directory"

**Fix**: 
```bash
# Verify directory structure
cd /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend
ls -la src/app/

# If missing, the app directory needs to be created (should already exist)
```

### Issue 3: Port 3000 already in use

**Fix**:
```bash
# Find process using port 3000
lsof -i :3000
# Or
netstat -tulpn | grep :3000

# Kill the process
kill -9 <PID>

# Or use a different port
npm run dev -- -p 3001
```

### Issue 4: Permission denied errors

**Fix**:
```bash
# Fix file permissions
sudo chown -R $USER:$USER /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline
chmod -R 755 /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend
```

### Issue 5: Still seeing Windows paths

**Fix**: You're still running from Windows terminal. Make sure:
1. You opened **Ubuntu** (WSL) terminal specifically
2. The prompt shows `samba@SamLenovo` (your WSL username)
3. `pwd` shows `/home/...` paths, not `C:\...`

## Quick Checklist

- [ ] Using WSL (Ubuntu) terminal, not Windows CMD/PowerShell
- [ ] Node.js installed in WSL (`node --version` works)
- [ ] npm installed in WSL (`npm --version` works)
- [ ] In correct directory: `/home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend`
- [ ] `src/app/` directory exists
- [ ] `package.json` exists
- [ ] Dependencies installed (`node_modules` exists)
- [ ] No `.next` directory corruption
- [ ] Port 3000 is available

## Alternative: Use Windows Node.js (Not Recommended)

If you must use Windows Node.js:

1. Move project to Windows filesystem (e.g., `C:\projects\...`)
2. Run from Windows CMD/PowerShell
3. **BUT**: This can cause file permission and path issues

**Recommendation**: Always use WSL for Linux-based projects.

## Still Having Issues?

1. Check Next.js logs for specific errors
2. Verify Next.js version: `npm list next`
3. Try creating a minimal test:
   ```bash
   cd /tmp
   npx create-next-app@latest test-app --typescript --app
   cd test-app
   npm run dev
   ```
4. If test app works, the issue is with your project setup
5. If test app fails, the issue is with your Node.js/WSL setup

