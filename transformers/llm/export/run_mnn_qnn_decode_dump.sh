#!/usr/bin/env bash
set -euo pipefail

show_usage() {
    cat <<'EOF'
Usage: run_mnn_qnn_decode_dump.sh [build|push|run|run-prof|stage-remote|pull|all|all-prof]

Environment overrides:
  HOST                 default: reck
  RECK_WORKING_DIR     default: /home/reck/mnn_qwen3
  DEVICE_ROOT          default: /data/local/tmp/MNN
    LOCAL_PROMPT_FILE    default: transformers/llm/export/propmt_256.txt
  DUMP_NAME            default: qnn-dump-hello-full
  LOCAL_PULL_DIR       default: /tmp/${DUMP_NAME}
  N_PREDICT            default: 2
  PROFILE_STAGE        default: 0
    MNN_CPU_ATTN_DUMP    default: 0
    PULL_MODE            default: minimal

Examples:
  ./run_mnn_qnn_decode_dump.sh all
  DUMP_NAME=qnn-dump-once ./run_mnn_qnn_decode_dump.sh run
  DUMP_NAME=qnn-dump-once ./run_mnn_qnn_decode_dump.sh stage-remote
EOF
}

retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        if [ "$n" -ge 5 ]; then
            echo "command failed after ${n} attempts: $*"
            return 1
        fi
        sleep 2
    done
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

HOST=${HOST:-reck}
RECK_WORKING_DIR=${RECK_WORKING_DIR:-/home/reck/mnn_qwen3}
DEVICE_ROOT=${DEVICE_ROOT:-/data/local/tmp/MNN}
LOCAL_PROMPT_FILE=${LOCAL_PROMPT_FILE:-${ROOT_DIR}/transformers/llm/export/propmt_256.txt}
DUMP_NAME=${DUMP_NAME:-qnn-dump-hello-full}
LOCAL_PULL_DIR=${LOCAL_PULL_DIR:-/tmp/${DUMP_NAME}}
N_PREDICT=${N_PREDICT:-128}
PROFILE_STAGE=${PROFILE_STAGE:-0}
MNN_CPU_ATTN_DUMP=${MNN_CPU_ATTN_DUMP:-0}
PULL_MODE=${PULL_MODE:-minimal}
MNN_LLM_PROMPT_WHOLE_FILE=${MNN_LLM_PROMPT_WHOLE_FILE:-0}

LOCAL_LIB_MNN="${ROOT_DIR}/project/android/build_64/libMNN.so"
LOCAL_LIB_MNN_EXPRESS="${ROOT_DIR}/project/android/build_64/libMNN_Express.so"
LOCAL_LIB_LLM="${ROOT_DIR}/project/android/build_64/libllm.so"
LOCAL_LLM_DEMO="${ROOT_DIR}/project/android/build_64/llm_demo"
REMOTE_PROMPT_FILE="${RECK_WORKING_DIR}/$(basename "${LOCAL_PROMPT_FILE}")"
DEVICE_PROMPT_FILE="${DEVICE_ROOT}/$(basename "${LOCAL_PROMPT_FILE}")"
DEVICE_DUMP_DIR="${DEVICE_ROOT}/${DUMP_NAME}"
DEVICE_LOG_FILE="${DEVICE_ROOT}/${DUMP_NAME}.log"
DEVICE_PROFILE_FILE="${DEVICE_ROOT}/${DUMP_NAME}.stage.tsv"

build_local() {
    cmake --build "${ROOT_DIR}/project/android/build_64" --target MNN llm_demo -j 4
}

push_remote() {
    local runtime_files=(
        "${LOCAL_LIB_MNN}"
        "${LOCAL_LIB_MNN_EXPRESS}"
        "${LOCAL_LIB_LLM}"
        "${LOCAL_LLM_DEMO}"
    )

    local local_file
    for local_file in "${runtime_files[@]}"; do
        if [ ! -f "${local_file}" ]; then
            echo "missing runtime artifact: ${local_file}"
            return 1
        fi
    done

    retry ssh "${HOST}" "mkdir -p '${RECK_WORKING_DIR}'"
    retry rsync -avhP "${LOCAL_LIB_MNN}" "${HOST}:${RECK_WORKING_DIR}/libMNN.so"
    retry rsync -avhP "${LOCAL_LIB_MNN_EXPRESS}" "${HOST}:${RECK_WORKING_DIR}/libMNN_Express.so"
    retry rsync -avhP "${LOCAL_LIB_LLM}" "${HOST}:${RECK_WORKING_DIR}/libllm.so"
    retry rsync -avhP "${LOCAL_LLM_DEMO}" "${HOST}:${RECK_WORKING_DIR}/llm_demo"
    retry rsync -avhP "${LOCAL_PROMPT_FILE}" "${HOST}:${REMOTE_PROMPT_FILE}"

    retry ssh "${HOST}" "
        adb push '${RECK_WORKING_DIR}/libMNN.so' '${DEVICE_ROOT}/libMNN.so' && \
        adb push '${RECK_WORKING_DIR}/libMNN_Express.so' '${DEVICE_ROOT}/libMNN_Express.so' && \
        adb push '${RECK_WORKING_DIR}/libllm.so' '${DEVICE_ROOT}/libllm.so' && \
        adb push '${RECK_WORKING_DIR}/llm_demo' '${DEVICE_ROOT}/llm_demo' && \
        adb push '${REMOTE_PROMPT_FILE}' '${DEVICE_PROMPT_FILE}'
    "
}

run_remote() {
    retry ssh "${HOST}" "
        adb shell '
            rm -rf \"${DEVICE_DUMP_DIR}\" \"${DEVICE_LOG_FILE}\" \"${DEVICE_PROFILE_FILE}\" && \
            cd \"${DEVICE_ROOT}\" && \
            env \
              LD_LIBRARY_PATH=\"${DEVICE_ROOT}\" \
              MNN_QNN_DUMP_DIR=\"${DEVICE_DUMP_DIR}\" \
              MNN_LLM_DUMP_PROMPT_DIR=\"${DEVICE_DUMP_DIR}\" \
              MNN_LLM_PROMPT_WHOLE_FILE=\"${MNN_LLM_PROMPT_WHOLE_FILE}\" \
              MNN_CPU_ATTN_DUMP=\"${MNN_CPU_ATTN_DUMP}\" \
              $( [ "${PROFILE_STAGE}" = "1" ] && printf "MNN_QNN_STAGE_PROFILE=\\\"%s\\\" " "${DEVICE_PROFILE_FILE}" ) \
              ./llm_demo model/config_qnn_raw.json \"$(basename "${DEVICE_PROMPT_FILE}")\" ${N_PREDICT} \
              2>&1 | tee \"${DEVICE_LOG_FILE}\" 
              # && 
            # tail -n 80 \"${DEVICE_LOG_FILE}\" && \
            # find \"${DEVICE_DUMP_DIR}\" -maxdepth 2 -type f | sort
        '
    "
}

run_remote_profile() {
    retry ssh "${HOST}" "
        adb shell '
            rm -rf \"${DEVICE_DUMP_DIR}\" \"${DEVICE_LOG_FILE}\" \"${DEVICE_PROFILE_FILE}\" && \
            cd \"${DEVICE_ROOT}\" && \
            env \
              LD_LIBRARY_PATH=\"${DEVICE_ROOT}\" \
              MNN_LLM_PROMPT_WHOLE_FILE=\"${MNN_LLM_PROMPT_WHOLE_FILE}\" \
              $( [ "${PROFILE_STAGE}" = "1" ] && printf "MNN_QNN_STAGE_PROFILE=\\\"%s\\\" " "${DEVICE_PROFILE_FILE}" ) \
              ./llm_demo model/config_qnn_raw.json \"$(basename "${DEVICE_PROMPT_FILE}")\" ${N_PREDICT} \
              > \"${DEVICE_LOG_FILE}\" 2>&1 && \
            tail -n 80 \"${DEVICE_LOG_FILE}\"
        '
    "
}

stage_remote() {
    retry ssh "${HOST}" "
        mkdir -p '${RECK_WORKING_DIR}' && \
        rm -rf '${RECK_WORKING_DIR}/${DUMP_NAME}' '${RECK_WORKING_DIR}/${DUMP_NAME}.log' '${RECK_WORKING_DIR}/${DUMP_NAME}.stage.tsv' && \
        if adb shell test -d '${DEVICE_DUMP_DIR}'; then adb pull '${DEVICE_DUMP_DIR}' '${RECK_WORKING_DIR}/'; fi && \
        if adb shell test -f '${DEVICE_LOG_FILE}'; then adb pull '${DEVICE_LOG_FILE}' '${RECK_WORKING_DIR}/${DUMP_NAME}.log'; fi && \
        if adb shell test -f '${DEVICE_PROFILE_FILE}'; then adb pull '${DEVICE_PROFILE_FILE}' '${RECK_WORKING_DIR}/${DUMP_NAME}.stage.tsv'; fi
    "
}

pull_remote() {
    rm -rf "${LOCAL_PULL_DIR}"
    mkdir -p "${LOCAL_PULL_DIR}"

    retry ssh "${HOST}" "
        rm -rf '${RECK_WORKING_DIR}/${DUMP_NAME}' '${RECK_WORKING_DIR}/${DUMP_NAME}.stage.tsv' && \
        if adb shell test -d '${DEVICE_DUMP_DIR}'; then adb pull '${DEVICE_DUMP_DIR}' '${RECK_WORKING_DIR}/'; fi && \
        adb pull '${DEVICE_LOG_FILE}' '${RECK_WORKING_DIR}/${DUMP_NAME}.log' && \
        if adb shell test -f '${DEVICE_PROFILE_FILE}'; then adb pull '${DEVICE_PROFILE_FILE}' '${RECK_WORKING_DIR}/${DUMP_NAME}.stage.tsv'; fi
    "

    if ssh "${HOST}" "test -d '${RECK_WORKING_DIR}/${DUMP_NAME}'"; then
        if [ "${PULL_MODE}" = "full" ]; then
            retry rsync -avhP "${HOST}:${RECK_WORKING_DIR}/${DUMP_NAME}/" "${LOCAL_PULL_DIR}/"
        else
            retry rsync -avhP --prune-empty-dirs \
                --include='*/' \
                --include='graph*-run-*/**' \
                --include='prefill/graph*-run-*/**' \
                --include='prefill/chunk-prefill-*/**' \
                --include='prefill/prompt-*' \
                --include='decode/graph*-run-*/**' \
                --include='decode/prompt-*' \
                --exclude='*' \
                "${HOST}:${RECK_WORKING_DIR}/${DUMP_NAME}/" "${LOCAL_PULL_DIR}/"
        fi
    fi
    retry rsync -avhP "${HOST}:${RECK_WORKING_DIR}/${DUMP_NAME}.log" "${LOCAL_PULL_DIR}/mnn.log"
    if ssh "${HOST}" "test -f '${RECK_WORKING_DIR}/${DUMP_NAME}.stage.tsv'"; then
        retry rsync -avhP "${HOST}:${RECK_WORKING_DIR}/${DUMP_NAME}.stage.tsv" "${LOCAL_PULL_DIR}/mnn.stage.tsv"
    fi

    echo "Data pulled from: ${HOST}:${RECK_WORKING_DIR}/${DUMP_NAME}/"
    echo "Log pulled from:  ${HOST}:${RECK_WORKING_DIR}/${DUMP_NAME}.log"
    echo "Saved to local dir: ${LOCAL_PULL_DIR}/"
}

case "${1:-all}" in
    build)
        build_local
        ;;
    push)
        push_remote
        ;;
    run)
        run_remote
        ;;
    run-prof)
        run_remote_profile
        ;;
    stage-remote)
        stage_remote
        ;;
    pull)
        pull_remote
        ;;
    all)
        build_local
        push_remote
        run_remote
        pull_remote
        ;;
    all-prof)
        build_local
        push_remote
        run_remote_profile
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
