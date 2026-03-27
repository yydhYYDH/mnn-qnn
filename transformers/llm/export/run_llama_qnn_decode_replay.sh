#!/usr/bin/env bash
set -euo pipefail

show_usage() {
    cat <<'EOF'
Usage: run_llama_qnn_decode_replay.sh [build|push|run|pull|all]

Environment overrides:
  HOST                 default: oneplus13
  PHONE_HOME           default: /data/data/com.termux/files/home
  REMOTE_ROOT          default: ${PHONE_HOME}/llama.cpp-test
  LOCAL_MODEL_PATH     default: models/Qwen3-4B-fp16.gguf
  PROMPT               default: hello
  N_PREDICT            default: 2
  DUMP_NAME            default: qnn-replay-hello-full
  LOCAL_PULL_DIR       default: /tmp/${DUMP_NAME}
  PUSH_GRAPH_START     default: 0
  PUSH_GRAPH_END       default: 37

Examples:
  PROMPT=hello ./run_llama_qnn_decode_replay.sh all
  DUMP_NAME=qnn-replay-once HOST=oneplus13 ./run_llama_qnn_decode_replay.sh run
EOF
}

retry() {
    local -a success_check=()
    if [ "${1:-}" = "--success-check" ]; then
        shift
        while [ "$#" -gt 0 ] && [ "${1}" != "--" ]; do
            success_check+=("$1")
            shift
        done
        if [ "$#" -eq 0 ]; then
            echo "retry: missing '--' after success check command" >&2
            return 1
        fi
        shift
    fi

    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi

        if [ "${#success_check[@]}" -gt 0 ] && "${success_check[@]}"; then
            echo "command reported failure but success check passed; not retrying: $*" >&2
            return 0
        fi

        if [ "${attempt}" -ge 5 ]; then
            echo "command failed after ${attempt} attempts: $*" >&2
            return 1
        fi

        attempt=$((attempt + 1))
        sleep 2
    done
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LLAMA_ROOT="${ROOT_DIR}/llama.cpp"

HOST=${HOST:-oneplus13}
PHONE_HOME=${PHONE_HOME:-/data/data/com.termux/files/home}
REMOTE_ROOT=${REMOTE_ROOT:-${PHONE_HOME}/llama.cpp-test}
REMOTE_BIN_DIR="${REMOTE_ROOT}/bin"
REMOTE_LIB_DIR="${REMOTE_ROOT}/build-android/lib"
REMOTE_OPS_DIR="${REMOTE_ROOT}/ops-bin"
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-models/Qwen3-4B-fp16.gguf}
PROMPT=${PROMPT:-hello}
N_PREDICT=${N_PREDICT:-2}
DUMP_NAME=${DUMP_NAME:-qnn-replay-hello-full}
LOCAL_PULL_DIR=${LOCAL_PULL_DIR:-/tmp/${DUMP_NAME}}
PUSH_GRAPH_START=${PUSH_GRAPH_START:-0}
PUSH_GRAPH_END=${PUSH_GRAPH_END:-37}

LOCAL_BIN="${LLAMA_ROOT}/build-android/bin/llama-cli"
LOCAL_GRAPH_DIR="${ROOT_DIR}/transformers/llm/export/model/qnn"
REMOTE_DUMP_DIR="${REMOTE_ROOT}/${DUMP_NAME}"
REMOTE_LOG_PATH="${REMOTE_ROOT}/${DUMP_NAME}.log"

remote_run_has_results() {
    ssh "${HOST}" "
        test -s '${REMOTE_LOG_PATH}' || \
        { test -d '${REMOTE_DUMP_DIR}' && find '${REMOTE_DUMP_DIR}' -mindepth 1 -print -quit | grep -q .; }
    "
}

build_local() {
    cmake --build "${LLAMA_ROOT}/build-android" --target ggml-qnn llama-cli -j 4
}

push_bin() {
    retry ssh "${HOST}" "mkdir -p '${REMOTE_BIN_DIR}' '${REMOTE_OPS_DIR}'"
    retry rsync -avhP "${LOCAL_BIN}" "${HOST}:${REMOTE_BIN_DIR}/llama-cli"

    local graph_paths=()
    local i
    for i in $(seq "${PUSH_GRAPH_START}" "${PUSH_GRAPH_END}"); do
        local graph_path="${LOCAL_GRAPH_DIR}/graph${i}.bin"
        if [ -f "${graph_path}" ]; then
            graph_paths+=("${graph_path}")
        fi
    done

    if [ "${#graph_paths[@]}" -gt 0 ]; then
        retry rsync -avhP "${graph_paths[@]}" "${HOST}:${REMOTE_OPS_DIR}/"
    fi
}

run_remote() {
    local prompt_escaped
    prompt_escaped=$(printf '%q' "${PROMPT}")

    retry --success-check remote_run_has_results -- ssh "${HOST}" "
        mkdir -p '${REMOTE_ROOT}' && \
        rm -rf '${REMOTE_DUMP_DIR}' && \
        cd '${REMOTE_ROOT}' && \
        env \
          LD_LIBRARY_PATH='${REMOTE_LIB_DIR}:/system/lib64:/vendor/lib64' \
          ADSP_LIBRARY_PATH='${REMOTE_LIB_DIR}' \
          GGML_QNN_BIN_PATH='${REMOTE_OPS_DIR}/graph0.bin' \
          GGML_QNN_REPLAY_MNN_DECODE=1 \
          GGML_QNN_REPLAY_DUMP_DIR='${REMOTE_DUMP_DIR}' \
          GGML_QNN_DECODE_LOG_TOKENS=0 \
          ./bin/llama-cli \
            -m '${LOCAL_MODEL_PATH}' \
            -p ${prompt_escaped} \
            -n '${N_PREDICT}' \
            --device QNN \
            --no-warmup \
            -no-cnv \
            2>&1 | tee '${REMOTE_LOG_PATH}'
    "
}

pull_remote() {
    rm -rf "${LOCAL_PULL_DIR}"
    mkdir -p "${LOCAL_PULL_DIR}"
    retry rsync -avhP "${HOST}:${REMOTE_DUMP_DIR}/" "${LOCAL_PULL_DIR}/"
    retry rsync -avhP "${HOST}:${REMOTE_LOG_PATH}" "${LOCAL_PULL_DIR}/llama.log"
}

case "${1:-all}" in
    build)
        build_local
        ;;
    push)
        push_bin
        ;;
    run)
        run_remote
        ;;
    pull)
        pull_remote
        ;;
    all)
        build_local
        push_bin
        run_remote
        pull_remote
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "unknown command: $1"
        show_usage
        exit 1
        ;;
esac
