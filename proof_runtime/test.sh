#!/bin/bash

# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PARENT_DIR="$(dirname "$SCRIPT_DIR")" 
# MCP_DIR="$PARENT_DIR/third_party"

# echo $MCP_DIR
# echo $PARENT_DIR

claude --print 'hello'
# claude --verbose --model sonnet --permission-mode bypassPermissions 'hello'