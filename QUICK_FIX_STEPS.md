# Quick Fix Steps for Next.js Issue

## The Problem
You're running `npm run dev` from Windows, but the project is in WSL. This causes path conflicts.

## The Solution (3 Steps)

### Step 1: Open WSL Terminal
**CRITICAL**: You MUST use Ubuntu (WSL) terminal, NOT Windows CMD/PowerShell.

- Open **Ubuntu** from Windows Start menu, OR
- Type `wsl` in Windows Run dialog (Win+R), OR  
- Open Windows Terminal and select "Ubuntu" profile

Verify you're in WSL:
```bash
pwd
# Should show: /home/samba/... (NOT C:\...)
```

### Step 2: Install Node.js in WSL (if needed)

Check if Node.js is installed:
```bash
node --version
```

If it says "command not found", install it:
```bash
# Quick install using NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
node --version
npm --version
```

### Step 3: Run Next.js from WSL

```bash
# Navigate to frontend directory
cd /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend

# Clean any corrupted files
rm -rf .next

# Install dependencies (if needed)
npm install

# Run dev server
npm run dev
```

**That's it!** Next.js should now start correctly.

---

## Or Use the Automated Fix Script

```bash
cd /home/samba/projects/MedTech-Diagnostic-LLM-Pipeline
./fix_nextjs.sh
```

This script will:
- Check if you're in WSL
- Install Node.js if needed
- Verify project structure
- Clean build artifacts
- Install dependencies
- Give you next steps

---

## Why This Happens

- **Windows CMD** can't properly handle WSL filesystem paths
- Next.js tries to create `.next` in `C:\Windows\` instead of your project
- UNC paths (`\\wsl.localhost\...`) cause permission errors

**Solution**: Always run Node.js/npm commands from **inside WSL**.

---

## Still Not Working?

1. Make sure you're in WSL terminal (check with `pwd`)
2. Make sure Node.js is installed in WSL (`node --version` works)
3. Make sure you're in the frontend directory
4. Check `FIX_NEXTJS_ISSUES.md` for detailed troubleshooting

