#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python -m famou \
  -c examples/lean_identity_namespace/config.yaml \
  -p examples/lean_identity_namespace/init.scratch \
  -e examples/lean_identity_namespace/evaluator.py
