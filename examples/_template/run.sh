#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXAMPLE_REL_DIR="${SCRIPT_DIR#"$ROOT_DIR"/}"
cd "$ROOT_DIR"

source "$HOME/.elan/env" 2>/dev/null || true
export FAMOU_LEAN_CONFIG="$SCRIPT_DIR/config.yaml"

python run_famou.py \
    --config "$EXAMPLE_REL_DIR/config.yaml" \
    --programs "$EXAMPLE_REL_DIR/init.scratch" \
    --evaluator proof_runtime/evaluator_lean.py
