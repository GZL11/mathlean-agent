#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python -m famou \
  -c examples/lean_and_left/config.yaml \
  -p examples/lean_and_left/init.scratch \
  -e examples/lean_and_left/evaluator.py
