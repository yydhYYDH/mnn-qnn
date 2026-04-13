#!/bin/bash

set -e

# ===== 配置区 =====
REMOTE_HOST="reck"
REMOTE_USER="${REMOTE_USER:-reck}"
REMOTE_REPO_PATH="/home/${REMOTE_USER}/llama.cpp-test"
LOCAL_SCRIPT="transformers/llm/export/compare_qnn_decode_replay.py"
REMOTE_SCRIPT_PATH="${REMOTE_REPO_PATH}/$(basename "${LOCAL_SCRIPT}")"

# 默认参数（可被命令行覆盖）
DEFAULT_MNN_ROOT="/home/${REMOTE_USER}/mnn-prefill-compare/prefill/"
DEFAULT_LLAMA_DIR="/home/${REMOTE_USER}/llama.cpp-test/llama-prefill-compare/decode-token-0000"

# ===== 函数定义 =====
print_usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  push     Push '${LOCAL_SCRIPT}' to ${REMOTE_HOST}:${REMOTE_REPO_PATH}
  compare  Run comparison script on ${REMOTE_HOST} with default or provided paths

Options for 'compare':
  --mnn-root PATH      Path to MNN root (default: ${DEFAULT_MNN_ROOT})
  --llama-dir PATH     Path to LLaMA decode dir (default: ${DEFAULT_LLAMA_DIR})

Example:
  $0 push
  $0 compare
  $0 compare --mnn-root /custom/mnn --llama-dir /custom/llama
EOF
}

push_script() {
    if [[ ! -f "${LOCAL_SCRIPT}" ]]; then
        echo "Error: Local script '${LOCAL_SCRIPT}' not found!" >&2
        exit 1
    fi

    echo "[+] Pushing '${LOCAL_SCRIPT}' to ${REMOTE_HOST}:${REMOTE_REPO_PATH}"

    # 确保远程目录存在
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO_PATH}'"

    # 推送脚本（保留权限）
    rsync -avz --chmod=+x "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO_PATH}/"

    echo "[+] Push completed."
}

ensure_remote_script() {
    push_script
}

run_compare() {
    local mnn_root="${DEFAULT_MNN_ROOT}"
    local llama_dir="${DEFAULT_LLAMA_DIR}"

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mnn-root)
                mnn_root="$2"
                shift 2
                ;;
            --llama-dir)
                llama_dir="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                print_usage
                exit 1
                ;;
        esac
    done

    ensure_remote_script

    echo "[+] Running comparison:"
    echo "    Host: ${REMOTE_USER}@${REMOTE_HOST}"
    echo "    Script: ${REMOTE_SCRIPT_PATH}"
    echo "    --mnn-root: ${mnn_root}"
    echo "    --llama-dir: ${llama_dir}"

    ssh "${REMOTE_USER}@${REMOTE_HOST}" "
        test -d '${mnn_root}' &&
        test -d '${llama_dir}' &&
        python3 '${REMOTE_SCRIPT_PATH}' \
            --mnn-root '${mnn_root}' \
            --llama-dir '${llama_dir}'
    "
}

# ===== 主逻辑 =====
if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift

case "${COMMAND}" in
    push)
        push_script
        ;;
    compare)
        run_compare "$@"
        ;;
    *)
        echo "Invalid command: ${COMMAND}" >&2
        print_usage
        exit 1
        ;;
esac
