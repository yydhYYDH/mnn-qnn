#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-reck}"
REMOTE_USER="${REMOTE_USER:-reck}"
REMOTE_REPO_PATH="${REMOTE_REPO_PATH:-/home/${REMOTE_USER}/llama.cpp-test}"

LOCAL_SCRIPT="${ROOT_DIR}/transformers/llm/export/compare_llama_prefill_attn_layer0.py"
REMOTE_SCRIPT_PATH="${REMOTE_REPO_PATH}/$(basename "${LOCAL_SCRIPT}")"
DEFAULT_DUMP_ROOT="${REMOTE_REPO_PATH}/qnn-prefill-attn-debug"

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [push|compare]

Commands:
  push     Push compare_llama_prefill_attn_layer0.py to ${REMOTE_HOST}
  compare  Run the layer-0 CPU-attention-vs-graph compare on ${REMOTE_HOST}

Environment overrides:
  REMOTE_HOST         default: ${REMOTE_HOST}
  REMOTE_USER         default: ${REMOTE_USER}
  REMOTE_REPO_PATH    default: ${REMOTE_REPO_PATH}
  DUMP_ROOT           default: ${DEFAULT_DUMP_ROOT}
  TOKEN_DIR           default: decode-token-0000
  CPU_ATTN_DIR        default: auto-detect cpu-attn-layer0-run-*
  LAYER_INDEX         default: 0
EOF
}

push_script() {
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO_PATH}'"
    rsync -avz --chmod=+x "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO_PATH}/"
}

run_compare() {
    local dump_root="${DUMP_ROOT:-${DEFAULT_DUMP_ROOT}}"
    local token_dir="${TOKEN_DIR:-decode-token-0000}"
    local cpu_attn_dir="${CPU_ATTN_DIR:-}"
    local layer_index="${LAYER_INDEX:-0}"

    push_script

    ssh "${REMOTE_USER}@${REMOTE_HOST}" "
        set -euo pipefail
        if ! test -d '${dump_root}'; then
            echo '[compare] missing dump root: ${dump_root}' >&2
            exit 1
        fi
        /home/reck/Utils/anaconda3/bin/python '${REMOTE_SCRIPT_PATH}' \
            --dump-root '${dump_root}' \
            --token-dir '${token_dir}' \
            --layer '${layer_index}' \
            ${cpu_attn_dir:+--cpu-attn-dir '${cpu_attn_dir}'}
    "
}

COMMAND="${1:-compare}"

case "${COMMAND}" in
    push)
        push_script
        ;;
    compare)
        run_compare
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo "unknown command: ${COMMAND}" >&2
        print_usage >&2
        exit 1
        ;;
esac
