#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

source "$HOME/.elan/env" 2>/dev/null || true

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT_DIR}/third_party/benchmarks/miniF2F}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/benchmark_runs/minif2f_test_smoke}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/config.yaml}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-1}"
PER_TASK_TIMEOUT="${PER_TASK_TIMEOUT:-1200}"
LEAN_VERIFY_TIMEOUT="${LEAN_VERIFY_TIMEOUT:-300}"
RUN_MODE="${RUN_MODE:-prepare}"
PROBLEM_ID="${PROBLEM_ID:-}"
SUITE_ROOT="${SUITE_ROOT:-}"

export FAMOU_LEAN_VERIFY_TIMEOUT="${LEAN_VERIFY_TIMEOUT}"

CMD=(
  python -B -m famou.utils.mathproof_benchmark_runner
  --benchmark minif2f
  --benchmark-root "${BENCHMARK_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --split "${SPLIT}"
  --limit "${LIMIT}"
  --config-template "${CONFIG_TEMPLATE}"
  --storage-base-path ./suite_data
)

if [[ -n "${PROBLEM_ID}" ]]; then
  CMD+=(--problem-id "${PROBLEM_ID}")
fi

case "${RUN_MODE}" in
  prepare)
    ;;
  full)
    CMD+=(--run --summarize --per-task-timeout "${PER_TASK_TIMEOUT}")
    ;;
  summarize)
    if [[ -z "${SUITE_ROOT}" ]]; then
      SUITE_ROOT="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    fi
    if [[ -z "${SUITE_ROOT}" ]]; then
      echo "Could not infer SUITE_ROOT under ${OUTPUT_ROOT}. Set SUITE_ROOT explicitly." >&2
      exit 1
    fi
    CMD=(
      python -B -m famou.utils.mathproof_benchmark_runner
      --summarize-suite-root "${SUITE_ROOT}"
    )
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}. Expected prepare, full, or summarize." >&2
    exit 1
    ;;
esac

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
