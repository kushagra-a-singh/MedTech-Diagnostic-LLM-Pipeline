#!/bin/bash

# Quick fix script for Next.js WSL issues
# Usage: ./fix_nextjs.sh

set -e

echo "🔧 Fixing Next.js WSL Path Issues..."
echo ""

# Check if running in WSL
if [[ ! -d /proc/version ]] || ! grep -q Microsoft /proc/version 2>/dev/null; then
    if [[ -d /mnt/c/Windows ]]; then
        echo "✓ Running in WSL"
    else
        echo "⚠️  Warning: This may not be WSL. Make sure you're running from Ubuntu terminal."
    fi
fi

# Check current directory
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# Navigate to frontend directory
FRONTEND_DIR="/home/samba/projects/MedTech-Diagnostic-LLM-Pipeline/frontend"
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Error: Frontend directory not found at $FRONTEND_DIR"
    exit 1
fi

cd "$FRONTEND_DIR"
echo "✓ Changed to frontend directory: $(pwd)"
echo ""

# Check if Node.js is installed
echo "1. Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "   ❌ Node.js not found in WSL"
    echo ""
    echo "   Installing Node.js..."
    echo "   Please choose installation method:"
    echo "   1) nvm (Node Version Manager) - Recommended"
    echo "   2) NodeSource repository"
    echo "   3) apt package manager"
    echo ""
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo "   Installing nvm..."
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
            export NVM_DIR="$HOME/.nvm"
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
            nvm install 20
            nvm use 20
            ;;
        2)
            echo "   Installing from NodeSource..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
            ;;
        3)
            echo "   Installing from apt..."
            sudo apt update
            sudo apt install -y nodejs npm
            ;;
        *)
            echo "   Invalid choice. Please install Node.js manually."
            exit 1
            ;;
    esac
else
    NODE_VERSION=$(node --version)
    echo "   ✓ Node.js found: $NODE_VERSION"
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo "   ❌ npm not found"
    echo "   Please install Node.js (which includes npm)"
    exit 1
else
    NPM_VERSION=$(npm --version)
    echo "   ✓ npm found: $NPM_VERSION"
fi

echo ""

# Check app directory
echo "2. Checking app directory structure..."
if [ -d "src/app" ]; then
    echo "   ✓ src/app directory exists"
    ls -la src/app/ | head -5
else
    echo "   ❌ src/app directory not found!"
    echo "   This is required for Next.js App Router"
    exit 1
fi

echo ""

# Clean build artifacts
echo "3. Cleaning build artifacts..."
rm -rf .next 2>/dev/null && echo "   ✓ Removed .next directory" || echo "   ✓ No .next directory to remove"
rm -rf node_modules/.cache 2>/dev/null && echo "   ✓ Removed node_modules cache" || echo "   ✓ No cache to remove"
echo ""

# Check if node_modules exists
echo "4. Checking dependencies..."
if [ ! -d "node_modules" ]; then
    echo "   ⚠️  node_modules not found. Installing dependencies..."
    npm install
else
    echo "   ✓ node_modules exists"
    echo "   Checking for updates..."
    npm install
fi

echo ""

# Verify Next.js config
echo "5. Verifying Next.js configuration..."
if [ -f "next.config.js" ]; then
    echo "   ✓ next.config.js exists"
    if grep -q "distDir" next.config.js; then
        echo "   ✓ distDir is configured"
    else
        echo "   ⚠️  distDir not found in config (this is okay, using default)"
    fi
else
    echo "   ⚠️  next.config.js not found (using defaults)"
fi

echo ""

# Check port availability
echo "6. Checking port 3000..."
if command -v lsof &> /dev/null; then
    if lsof -i :3000 &> /dev/null; then
        echo "   ⚠️  Port 3000 is in use"
        echo "   Kill the process or use a different port"
    else
        echo "   ✓ Port 3000 is available"
    fi
else
    echo "   ⚠️  Cannot check port (lsof not installed)"
fi

echo ""
echo "✅ Setup check complete!"
echo ""
echo "Next steps:"
echo "1. Make sure you're running from WSL (Ubuntu) terminal"
echo "2. Run: cd $FRONTEND_DIR"
echo "3. Run: npm run dev"
echo ""
echo "If you still see errors, check FIX_NEXTJS_ISSUES.md for detailed troubleshooting"

