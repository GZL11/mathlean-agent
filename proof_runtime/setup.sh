#!/bin/bash

# Lean + Claude Code Setup Script
# Usage: ./setup.sh [project_name]

# If you need a proxy, export ALL_PROXY / http_proxy / https_proxy before running.
set -e

PROJECT_NAME="${1:-project}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")" 
MCP_DIR="$PARENT_DIR/third_party/lean-lsp-mcp"
PROJECT_DIR="$SCRIPT_DIR/$PROJECT_NAME"

echo "========================================"
echo "Lean + Claude Code Quick Setup"
echo "Project name: $PROJECT_NAME"
echo "========================================"

# 1. Install elan (Lean version manager)
echo ""
echo "[1/5] Installing elan (Lean version manager)..."
if command -v elan &> /dev/null; then
    echo "elan is already installed, skipping..."
else
    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y
    source "$HOME/.elan/env"
fi

# 2. Create Lean project
echo ""
echo "[2/5] Creating Lean project: $PROJECT_NAME..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory $PROJECT_DIR already exists, skipping project creation..."
else
    cd "$SCRIPT_DIR"
    lake new "$PROJECT_NAME" math
    cd "$PROJECT_DIR"
    echo "Building Lean Project (this may take a while)"
    lake update
    lake exe cache get
    lake build
fi
# cd "$PROJECT_NAME"
# echo "Updating dependencies and building (this may take a while)..."
# === Don't do lake exe cache get everytime, which is every slow ===
# lake update
# lake exe cache get
# lake build

# export http_proxy=http://<your-proxy>:<port>
# export https_proxy=http://<your-proxy>:<port>
# # 3. Install Claude Code
# echo ""
# echo "[3/5] Installing Claude Code..."
# if command -v claude &> /dev/null; then
#     echo "Claude Code is already installed, skipping..."
# else
#     bash ./install.sh
#     # curl -fsSL -k --proxy-insecure --http1.1 https://claude.ai/install.sh | bash
#     # curl -fsSL https://claude.ai/install.sh | bash
# fi
# unset http_proxy
# unset https_proxy

# 4. Install uv
echo ""
echo "[4/5] Installing uv..."
if command -v uv &> /dev/null; then
    echo "uv is already installed, skipping..."
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# 5. Setup lean-lsp-mcp
echo ""
echo "[5/5] Setting up lean-lsp-mcp..."
# MCP_DIR="$HOME/lean-lsp-mcp"
if [ -d "$MCP_DIR" ]; then
    # Currently the lean-lsp-mcp is integrated into our repo.
    # echo "lean-lsp-mcp already exists at $MCP_DIR, updating..."
    echo "lean-lsp-mcp already exists at $MCP_DIR, skipping without update..."
    # cd "$MCP_DIR"
    # git pull
else
    git clone https://github.com/project-numina/lean-lsp-mcp "$MCP_DIR"
fi
chmod +x "$MCP_DIR/numina-lean-mcp.sh"

# Add MCP to Claude Code (from repo root so example entrypoints can see the config)
cd "$PARENT_DIR"
LEAN_PROJECT_PATH="$PROJECT_DIR"
echo "Adding MCP server to Claude Code..."
claude mcp remove lean-lsp 2>/dev/null || true
claude mcp add lean-lsp -- env "LEAN_PROJECT_PATH=$LEAN_PROJECT_PATH" "$MCP_DIR/numina-lean-mcp.sh"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To verify installation:"
echo "  claude mcp list"
echo ""
echo "To start working:"
echo "  cd $PARENT_DIR"
echo "  claude"
echo ""
echo "For more details, see: tutorial/setup.md"
