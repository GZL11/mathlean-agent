#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LOG_NAME="${MCP_LOG_NAME:-mcp_lean_lsp}"
LOG_DIR="${MCP_LOG_DIR:-$PROJECT_DIR}"

# check if log_dir exists
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# --- 新增调试信息：记录到日志 ---
{
    echo "------------------------------------------"
    echo "Timestamp: $(date)"
    echo "Project Directory: $PROJECT_DIR"
    
    # 查找 lean_lsp_mcp 这个包在 uv 环境中的真实物理路径
    echo -n "Actual Package Path: "
    uvx --with-editable "$PROJECT_DIR" python -c "import leanclient; import os; print(os.path.dirname(leanclient.__file__))"
    
    # 查找 python 解释器位置
    echo -n "Python Executable: "
    uvx --with-editable "$PROJECT_DIR" which python
    echo ""
    echo "------------------------------------------"
} >> "$LOG_DIR/$LOG_NAME.log"

#exec uvx --with-editable "$PROJECT_DIR" python -m lean_lsp_mcp.server 2>> "$PROJECT_DIR/mcp_lean_lsp.log"
exec uvx --with-editable "$PROJECT_DIR" python -m lean_lsp_mcp.server 2>> "$LOG_DIR/$LOG_NAME.log"

