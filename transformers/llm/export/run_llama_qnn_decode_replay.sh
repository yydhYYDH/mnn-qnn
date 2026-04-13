#!/usr/bin/env bash
set -euo pipefail

show_usage() {
    cat <<'EOF'
Usage: run_llama_qnn_decode_replay.sh [build|push|run|run-debug|pull|all|all-debug]

Environment overrides:
  HOST                 default: oneplus13
  REMOTE_EXEC_MODE     default: auto; auto uses adb when HOST=reck, otherwise termux ssh
  PHONE_HOME           default: /data/data/com.termux/files/home
  REMOTE_ROOT          default: ${PHONE_HOME}/llama.cpp-test
  REMOTE_STAGE_ROOT    default: /home/reck/llama.cpp-test when adb mode is active
  DEVICE_ROOT          default: /data/local/tmp/MNN when adb mode is active
                      device layout in adb mode defaults to:
                        ${DEVICE_ROOT}/llama-cli
                        ${DEVICE_ROOT}/*.so
                        ${DEVICE_ROOT}/model/qnn/graph0.bin
  LOCAL_MODEL_PATH     default: models/Qwen3-4B-fp16.gguf
  DEVICE_MODEL_PATH    default: ${DEVICE_ROOT}/$(basename "${LOCAL_MODEL_PATH}") in adb mode
  PUSH_MODEL           default: 0; set to 1 to adb push the gguf into DEVICE_MODEL_PATH
  PROMPT               default: hello
    PROMPT_FILE          default: transformers/llm/export/propmt_256.txt; set PROMPT_FILE= to force raw PROMPT
  N_PREDICT            default: 2
    SEED                 default: 123
    TEMP                 default: 0
  UBATCH_SIZE          default: 128
  CLEAN_OUTPUT         default: 0
  DUMP_NAME            default: qnn-replay-hello-full
  LOCAL_PULL_DIR       default: /tmp/${DUMP_NAME}
  REMOTE_PULL_DIR      default: /home/reck/llama.cpp-test/${DUMP_NAME} in adb mode
  PROFILE_STAGE        default: 0
    PROFILE_SUMMARY      default: 1
        PREFILL_SUMMARY      default: 1; analyze pulled llama.stage.tsv for prefill stage share
    PROFILE_LIMIT        default: 12
        PROFILE_PHASE        default: auto; defaults to prefill for run/all
        GGML_QNN_PROFILE     default: off; set to basic/detailed for QNN SDK profile
        GGML_QNN_DEBUG_DECODE default: 0; set to 1 to print decode/prefill dispatch logs
        GGML_QNN_DECODE_LOG_TOKENS default: 0; when DEBUG_DECODE=0, log first N decode tokens
    LIVE_RECURRENT       default: 1
    LIVE_RECURRENT_DECODE default: ${LIVE_RECURRENT}
    LIVE_RECURRENT_PREFILL default: ${LIVE_RECURRENT}
    AUX_CACHE_POLICY     default: tail
    AUX_CACHE_MAX_GRAPH_INDEX default: empty, use backend default
    FAMILY_EXECUTOR      default: auto; unset follows backend policy, set to 0/1 to force disable/enable
    FAMILY_PRELOAD       default: auto; when unset, backend decides whether to preload family bins
    PREFILL_FAMILY_EXECUTOR default: 1; run fallback when FAMILY_EXECUTOR is unset
    PREFILL_FAMILY_PRELOAD  default: 0; run fallback when FAMILY_PRELOAD is unset
    LOAD_EXTRA_MNN_BINS  default: 0, set to 1 to eager-preload auxiliary decode bins for stress testing
  PUSH_GRAPHS          default: 0; set to 1 to sync graph*.bin to reck staging/device
  PUSH_GRAPH_START     default: 0
  PUSH_GRAPH_END       default: 37
    GGML_QNN_DUMP_ATTENTION default: 0
    PULL_MODE            default: minimal
    PULL_METHOD          default: tar; set to 'rsync' for legacy behavior, 'tar' for faster compression-based transfer

Examples:
  PROMPT=hello ./run_llama_qnn_decode_replay.sh all
    GGML_QNN_DUMP_ATTENTION=1 ./run_llama_qnn_decode_replay.sh run-debug
  DUMP_NAME=qnn-replay-once HOST=oneplus13 ./run_llama_qnn_decode_replay.sh run
  HOST=reck REMOTE_EXEC_MODE=adb DEVICE_ROOT=/data/local/tmp/MNN ./run_llama_qnn_decode_replay.sh run
  PROMPT_FILE=/tmp/long.txt N_PREDICT=1 PROFILE_STAGE=1 ./run_llama_qnn_decode_replay.sh run
    PROMPT_FILE=/tmp/long.txt N_PREDICT=1 PROFILE_STAGE=1 ./run_llama_qnn_decode_replay.sh run-debug
  PULL_METHOD=rsync ./run_llama_qnn_decode_replay.sh pull  # Use legacy rsync method
  PULL_METHOD=tar ./run_llama_qnn_decode_replay.sh pull    # Use fast tar method (default)

Notes:
    run is the prefill performance path and does not set GGML_QNN_REPLAY_DUMP_DIR.
    run always forces GGML_QNN_LIVE_RECURRENT_DECODE=1 and GGML_QNN_LIVE_RECURRENT_PREFILL=1.
    run defaults to FAMILY_EXECUTOR=1 and FAMILY_PRELOAD=0 to avoid eager-loading the full graph family.
    run-debug preserves the old dump-producing behavior for prefill investigation.
    PULL_METHOD=tar is faster for large dumps with many files (recommended).
    PULL_METHOD=rsync provides more granular control but slower for many files.
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

is_adb_mode() {
    [ "${REMOTE_EXEC_MODE}" = "adb" ]
}

runtime_model_path() {
    if is_adb_mode; then
        printf "%s" "${DEVICE_MODEL_PATH}"
    else
        printf "%s" "${LOCAL_MODEL_PATH}"
    fi
}

runtime_bin_path() {
    printf "%s" "${REMOTE_BIN_PATH}"
}

verify_remote_runtime_layout() {
    if ! is_adb_mode; then
        return 0
    fi

    local missing_output
    missing_output=$(
        ssh "${HOST}" "
            test -f '${REMOTE_STAGE_BIN_PATH}' || echo 'host_stage_bin ${REMOTE_STAGE_BIN_PATH}'
            adb shell $(device_shell_escape "
                test -f '${REMOTE_BIN_PATH}' || echo 'device_bin ${REMOTE_BIN_PATH}'
                test -d '${REMOTE_LIB_DIR}' || echo 'device_lib_dir ${REMOTE_LIB_DIR}'
                test -f '${REMOTE_OPS_DIR}/graph0.bin' || echo 'device_graph0 ${REMOTE_OPS_DIR}/graph0.bin'
                test -f '${DEVICE_MODEL_PATH}' || echo 'device_model ${DEVICE_MODEL_PATH}'
            ")
        " 2>/dev/null || true
    )

    if [ -z "${missing_output}" ]; then
        return 0
    fi

    cat >&2 <<EOF
remote runtime files are missing for adb mode (HOST=${HOST}):
  ${missing_output}

Path mapping in adb mode:
  host stage root: ${REMOTE_STAGE_ROOT}
  device root: ${REMOTE_ROOT}
  device bin: ${REMOTE_BIN_PATH}
  device libs: ${REMOTE_LIB_DIR}
  device graph0: ${REMOTE_OPS_DIR}/graph0.bin
  device model: ${DEVICE_MODEL_PATH}

Run one of these first:
  ./run_llama_qnn_decode_replay.sh push
  PUSH_MODEL=1 ./run_llama_qnn_decode_replay.sh push
  ./run_llama_qnn_decode_replay.sh all
EOF
    return 1
}

device_shell_escape() {
    printf "%q" "set -e; $1"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LLAMA_ROOT="${ROOT_DIR}/llama.cpp"

HOST=${HOST:-reck}
REMOTE_EXEC_MODE=${REMOTE_EXEC_MODE:-auto}
PHONE_HOME=${PHONE_HOME:-/data/data/com.termux/files/home}
REMOTE_STAGE_ROOT=${REMOTE_STAGE_ROOT:-/home/reck/llama.cpp-test}
DEVICE_ROOT=${DEVICE_ROOT:-/data/local/tmp/MNN}
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-models/Qwen3-4B-fp16.gguf}
DEVICE_MODEL_PATH=${DEVICE_MODEL_PATH:-${DEVICE_ROOT}/$(basename "${LOCAL_MODEL_PATH}")}
PUSH_MODEL=${PUSH_MODEL:-0}
if [ "${REMOTE_EXEC_MODE}" = "auto" ]; then
    if [ "${HOST}" = "reck" ]; then
        REMOTE_EXEC_MODE=adb
    else
        REMOTE_EXEC_MODE=termux
    fi
fi
if [ "${REMOTE_EXEC_MODE}" = "adb" ]; then
    REMOTE_ROOT=${REMOTE_ROOT:-${DEVICE_ROOT}}
else
    REMOTE_ROOT=${REMOTE_ROOT:-${PHONE_HOME}/llama.cpp-test}
fi
if [ "${REMOTE_EXEC_MODE}" = "adb" ]; then
    REMOTE_BIN_PATH=${REMOTE_BIN_PATH:-${REMOTE_ROOT}/llama-cli}
    REMOTE_LIB_DIR=${REMOTE_LIB_DIR:-${REMOTE_ROOT}}
    REMOTE_OPS_DIR=${REMOTE_OPS_DIR:-${REMOTE_ROOT}/model/qnn}
else
    REMOTE_BIN_PATH=${REMOTE_BIN_PATH:-${REMOTE_ROOT}/bin/llama-cli}
    REMOTE_LIB_DIR=${REMOTE_LIB_DIR:-${REMOTE_ROOT}/build-android/lib}
    REMOTE_OPS_DIR=${REMOTE_OPS_DIR:-${REMOTE_ROOT}/ops-bin}
fi
REMOTE_BIN_DIR="$(dirname "${REMOTE_BIN_PATH}")"
REMOTE_STAGE_BIN_PATH="${REMOTE_STAGE_ROOT}/bin/llama-cli"
PROMPT=${PROMPT:-hello}
PROMPT_FILE=${PROMPT_FILE-${SCRIPT_DIR}/propmt_256_0.txt}
N_PREDICT=${N_PREDICT:-128}
SEED=${SEED:-123}
TEMP=${TEMP:-0}
UBATCH_SIZE=${UBATCH_SIZE:-128}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-0}
DUMP_NAME=${DUMP_NAME:-qnn-replay-hello-full}
LOCAL_PULL_DIR=${LOCAL_PULL_DIR:-/tmp/${DUMP_NAME}}
if is_adb_mode; then
    REMOTE_PULL_DIR=${REMOTE_PULL_DIR:-${REMOTE_STAGE_ROOT}/${DUMP_NAME}}
else
    REMOTE_PULL_DIR=${REMOTE_PULL_DIR:-${REMOTE_ROOT}/${DUMP_NAME}}
fi
PROFILE_STAGE=${PROFILE_STAGE:-0}
PROFILE_SUMMARY=${PROFILE_SUMMARY:-1}
PREFILL_SUMMARY=${PREFILL_SUMMARY:-1}
PROFILE_LIMIT=${PROFILE_LIMIT:-12}
PROFILE_PHASE=${PROFILE_PHASE:-}
GGML_QNN_PROFILE=${GGML_QNN_PROFILE:-off}
GGML_QNN_DEBUG_DECODE=${GGML_QNN_DEBUG_DECODE:-0}
GGML_QNN_DECODE_LOG_TOKENS=${GGML_QNN_DECODE_LOG_TOKENS:-0}
LIVE_RECURRENT=${LIVE_RECURRENT:-1}
LIVE_RECURRENT_DECODE=${LIVE_RECURRENT_DECODE:-${LIVE_RECURRENT}}
LIVE_RECURRENT_PREFILL=${LIVE_RECURRENT_PREFILL:-${LIVE_RECURRENT}}
AUX_CACHE_POLICY=${AUX_CACHE_POLICY:-tail}
AUX_CACHE_MAX_GRAPH_INDEX=${AUX_CACHE_MAX_GRAPH_INDEX:-}
FAMILY_EXECUTOR=${FAMILY_EXECUTOR-}
FAMILY_PRELOAD=${FAMILY_PRELOAD-}
PREFILL_FAMILY_EXECUTOR=${PREFILL_FAMILY_EXECUTOR:-1}
PREFILL_FAMILY_PRELOAD=${PREFILL_FAMILY_PRELOAD:-0}
LOAD_EXTRA_MNN_BINS=${LOAD_EXTRA_MNN_BINS:-0}
PUSH_GRAPH_START=${PUSH_GRAPH_START:-0}
PUSH_GRAPH_END=${PUSH_GRAPH_END:-37}
PUSH_GRAPHS=${PUSH_GRAPHS:-0}
GGML_QNN_DUMP_ATTENTION=${GGML_QNN_DUMP_ATTENTION:-0}
PULL_MODE=${PULL_MODE:-minimal}
PULL_METHOD=${PULL_METHOD:-tar}

LOCAL_BIN="${LLAMA_ROOT}/build-android/bin/llama-cli"
LOCAL_GRAPH_DIR="${ROOT_DIR}/transformers/llm/export/model/qnn"
REMOTE_DUMP_DIR="${REMOTE_ROOT}/${DUMP_NAME}"
REMOTE_LOG_PATH="${REMOTE_ROOT}/${DUMP_NAME}.log"
REMOTE_STDOUT_PATH="${REMOTE_ROOT}/${DUMP_NAME}.out"
REMOTE_PROFILE_PATH="${REMOTE_ROOT}/${DUMP_NAME}.stage.tsv"
REMOTE_PROMPT_PATH="${REMOTE_ROOT}/${DUMP_NAME}.prompt.txt"
REMOTE_STAGE_PROMPT_PATH="${REMOTE_STAGE_ROOT}/${DUMP_NAME}.prompt.txt"

minimal_pull_rsync_args() {
    printf -- "--include=*/ "
    printf -- "--include=decode-token-*/graph* "
    if [ "${GGML_QNN_DUMP_ATTENTION}" = "1" ]; then
        printf -- "--include=decode-token-*/attn-* "
    fi
    printf -- "--exclude=*"
}

# Generate tar include patterns for minimal mode
minimal_pull_tar_includes() {
    echo "*/graph*"
    if [ "${GGML_QNN_DUMP_ATTENTION}" = "1" ]; then
        echo "*/attn-*"
    fi
}

stage_profile_env_args() {
    if [ "${PROFILE_STAGE}" = "1" ]; then
        printf "GGML_QNN_STAGE_PROFILE='%s' " "${REMOTE_PROFILE_PATH}"
    fi
}

live_recurrent_env_args() {
    printf "GGML_QNN_LIVE_RECURRENT='%s' GGML_QNN_LIVE_RECURRENT_DECODE='%s' GGML_QNN_LIVE_RECURRENT_PREFILL='%s' " \
        "${LIVE_RECURRENT}" "${LIVE_RECURRENT_DECODE}" "${LIVE_RECURRENT_PREFILL}"
}

live_recurrent_env_args_prefill_dual() {
    printf "GGML_QNN_LIVE_RECURRENT='%s' GGML_QNN_LIVE_RECURRENT_DECODE='1' GGML_QNN_LIVE_RECURRENT_PREFILL='1' " \
        "${LIVE_RECURRENT}"
}

aux_cache_env_args() {
    printf "GGML_QNN_MNN_AUX_CACHE_POLICY='%s' " "${AUX_CACHE_POLICY}"
    if [ -n "${AUX_CACHE_MAX_GRAPH_INDEX}" ]; then
        printf "GGML_QNN_MNN_AUX_CACHE_MAX_GRAPH_INDEX='%s' " "${AUX_CACHE_MAX_GRAPH_INDEX}"
    fi
    if [ -n "${FAMILY_EXECUTOR}" ]; then
        printf "GGML_QNN_MNN_FAMILY_EXECUTOR='%s' " "${FAMILY_EXECUTOR}"
    fi
    if [ -n "${FAMILY_PRELOAD}" ]; then
        printf "GGML_QNN_MNN_FAMILY_PRELOAD='%s' " "${FAMILY_PRELOAD}"
    fi
}

family_executor_env_args() {
    local default_executor="${1:-0}"
    local default_preload="${2:-0}"
    local family_executor="${FAMILY_EXECUTOR-}"
    local family_preload="${FAMILY_PRELOAD-}"

    if [ -z "${family_executor}" ]; then
        if [ -n "${family_preload}" ]; then
            family_executor="${family_preload}"
        else
            family_executor="${default_executor}"
        fi
    fi

    if [ -z "${family_preload}" ]; then
        family_preload="${default_preload}"
    fi

    printf "GGML_QNN_MNN_FAMILY_EXECUTOR='%s' GGML_QNN_MNN_FAMILY_PRELOAD='%s' " \
        "${family_executor}" "${family_preload}"
}

replay_dump_env_args() {
    if [ "${1:-0}" = "1" ]; then
        printf "GGML_QNN_REPLAY_DUMP_DIR='%s' " "${REMOTE_DUMP_DIR}"
    fi
}

remote_cleanup_cmd() {
    local remove_dump="${1:-1}"
    printf "mkdir -p '%s' && " "${REMOTE_ROOT}"
    if [ "${remove_dump}" = "1" ]; then
        printf "rm -rf '%s' && " "${REMOTE_DUMP_DIR}"
    fi
    printf "rm -f '%s' '%s' '%s' && cd '%s' && " \
        "${REMOTE_LOG_PATH}" "${REMOTE_STDOUT_PATH}" "${REMOTE_PROFILE_PATH}" "${REMOTE_ROOT}"
}

llama_common_args() {
    local prompt_args
    prompt_args=$(llama_prompt_args)
    printf "'%s' -m '%s' %s -n '%s' --seed '%s' --temp '%s' --device QNN --no-warmup -no-cnv" \
        "$(runtime_bin_path)" \
        "$(runtime_model_path)" \
        "${prompt_args}" \
        "${N_PREDICT}" \
        "${SEED}" \
        "${TEMP}"
}

run_remote_cmd() {
    local remove_dump="${1}"
    local env_args="${2}"
    local cli_args="${3}"
    local redirect_mode="${4}"
    local flat_env_args="${env_args//$'\n'/ }"

    if [ "${redirect_mode}" = "split" ]; then
        cat <<EOF
$(remote_cleanup_cmd "${remove_dump}")env ${flat_env_args} $(llama_common_args) ${cli_args} 2> '${REMOTE_LOG_PATH}' | tee '${REMOTE_STDOUT_PATH}'
EOF
    else
        cat <<EOF
$(remote_cleanup_cmd "${remove_dump}")env ${flat_env_args} $(llama_common_args) ${cli_args} 2>&1 | tee '${REMOTE_LOG_PATH}'
EOF
    fi
}

print_remote_command() {
    local label="${1}"
    local remote_cmd="${2}"

    echo "[${label}] ssh ${HOST}"
    if is_adb_mode; then
        echo "[${label}] adb shell command:"
    else
        echo "[${label}] remote command:"
    fi
    printf '%s\n' "${remote_cmd}"
}

run_remote_command() {
    local label="${1}"
    local remote_cmd="${2}"
    verify_remote_runtime_layout
    print_remote_command "${label}" "${remote_cmd}"
    if is_adb_mode; then
        retry --success-check remote_run_has_results -- ssh "${HOST}" "adb shell $(device_shell_escape "${remote_cmd}")"
    else
        retry --success-check remote_run_has_results -- ssh "${HOST}" "${remote_cmd}"
    fi
}

default_profile_phase() {
    if [ -n "${PROFILE_PHASE}" ]; then
        printf "%s" "${PROFILE_PHASE}"
    else
        printf "decode"
    fi
}

print_preload_summary_local() {
    local local_profile_path="${1}"

    python3 - "${local_profile_path}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
bin_load_us = 0
family_preload_us = 0
family_graph_hit_us = 0

for line in path.read_text().splitlines():
    if not line.strip():
        continue
    parts = [field.strip() for field in line.split("\t")]
    if len(parts) != 10:
        continue
    if parts[0] == "ts_us":
        continue
    _, _, phase, stage, _, _, _, duration_us, _, _ = parts
    duration = int(duration_us)
    if phase == "init" and stage == "bin_load":
        bin_load_us += duration
    elif phase == "runtime" and stage == "family_bin_preload":
        family_preload_us += duration
    elif phase == "runtime" and stage == "family_graph_hit":
        family_graph_hit_us += duration

total_preload_us = bin_load_us + family_preload_us
print(f"qnn preload summary ({path})")
print(f"bin_load_us={bin_load_us}")
print(f"family_bin_preload_us={family_preload_us}")
print(f"family_graph_hit_us={family_graph_hit_us}")
print(f"total_preload_us={total_preload_us}")
print("")
PY
}

summarize_profile_local() {
    if [ "${PROFILE_STAGE}" != "1" ] || [ "${PROFILE_SUMMARY}" != "1" ]; then
        return 0
    fi

    local local_profile_path="${LOCAL_PULL_DIR}/llama.stage.tsv"
    if [ ! -f "${local_profile_path}" ]; then
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "profile summary skipped: python3 not found" >&2
        return 0
    fi

    print_preload_summary_local "${local_profile_path}"

    local summary_phase
    summary_phase=$(default_profile_phase)

    echo "${summary_phase} stage profile summary (${local_profile_path})"
    python3 "${SCRIPT_DIR}/analyze_stage_profile.py" \
        "${local_profile_path}" \
        --phase "${summary_phase}" \
        --limit "${PROFILE_LIMIT}" \
        --show-token-totals
}

summarize_prefill_profile_local() {
    if [ "${PREFILL_SUMMARY}" != "1" ]; then
        return 0
    fi

    local local_profile_path="${LOCAL_PULL_DIR}/llama.stage.tsv"
    if [ ! -f "${local_profile_path}" ]; then
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "prefill summary skipped: python3 not found" >&2
        return 0
    fi

    echo "prefill stage share summary (${local_profile_path})"
    python3 "${SCRIPT_DIR}/analyze_prefill_profile.py" \
        "${local_profile_path}" \
        --phase prefill \
        --limit "${PROFILE_LIMIT}"
}

remote_run_has_results() {
    if is_adb_mode; then
        ssh "${HOST}" "adb shell $(device_shell_escape "test -s '${REMOTE_LOG_PATH}' || { test -d '${REMOTE_DUMP_DIR}' && find '${REMOTE_DUMP_DIR}' -mindepth 1 -print -quit | grep -q .; }")"
    else
        ssh "${HOST}" "
            test -s '${REMOTE_LOG_PATH}' || \
            { test -d '${REMOTE_DUMP_DIR}' && find '${REMOTE_DUMP_DIR}' -mindepth 1 -print -quit | grep -q .; }
        "
    fi
}

build_local() {
    cmake --build "${LLAMA_ROOT}/build-android" --target ggml-qnn llama-cli -j 4
}

prepare_remote_prompt() {
    if [ -z "${PROMPT_FILE}" ]; then
        return 0
    fi

    if [ ! -f "${PROMPT_FILE}" ]; then
        echo "prompt file not found: ${PROMPT_FILE}" >&2
        return 1
    fi

    if is_adb_mode; then
        retry ssh "${HOST}" "mkdir -p '${REMOTE_STAGE_ROOT}'"
        retry rsync -avhP "${PROMPT_FILE}" "${HOST}:${REMOTE_STAGE_PROMPT_PATH}"
        retry ssh "${HOST}" "adb shell $(device_shell_escape "mkdir -p '${REMOTE_ROOT}'") && adb push '${REMOTE_STAGE_PROMPT_PATH}' '${REMOTE_PROMPT_PATH}'"
    else
        retry ssh "${HOST}" "mkdir -p '${REMOTE_ROOT}'"
        retry rsync -avhP "${PROMPT_FILE}" "${HOST}:${REMOTE_PROMPT_PATH}"
    fi
}

llama_prompt_args() {
    if [ -n "${PROMPT_FILE}" ]; then
        printf -- "-f '%s'" "${REMOTE_PROMPT_PATH}"
    else
        local prompt_escaped
        prompt_escaped=$(printf '%q' "${PROMPT}")
        printf -- "-p %s" "${prompt_escaped}"
    fi
}

require_prefill_eval() {
    if [ "${N_PREDICT}" = "0" ]; then
        cat >&2 <<'EOF'
run-prefill requires N_PREDICT>=1.
This llama-cli build does not enter the eval loop when -n 0, so the prompt is tokenized but prefill is not executed.
Use N_PREDICT=1 and read prompt eval time / stage profile for real prefill numbers.
EOF
        return 1
    fi
}

push_bin() {
    local graph_paths=()
    local i
    for i in $(seq "${PUSH_GRAPH_START}" "${PUSH_GRAPH_END}"); do
        local graph_path="${LOCAL_GRAPH_DIR}/graph${i}.bin"
        if [ -f "${graph_path}" ]; then
            graph_paths+=("${graph_path}")
        fi
        local prefill_graph_path="${LOCAL_GRAPH_DIR}/graph1_${i}.bin"
        if [ -f "${prefill_graph_path}" ]; then
            graph_paths+=("${prefill_graph_path}")
        fi
    done

    if is_adb_mode; then
        retry ssh "${HOST}" "mkdir -p '${REMOTE_STAGE_ROOT}/bin' '${REMOTE_STAGE_ROOT}/build-android' '${REMOTE_STAGE_ROOT}/ops-bin'"
        retry rsync -avhP "${LOCAL_BIN}" "${HOST}:${REMOTE_STAGE_BIN_PATH}"
        # retry rsync -avhP "${LLAMA_ROOT}/build-android/lib/" "${HOST}:${REMOTE_STAGE_ROOT}/build-android/lib/"
        if [ "${PUSH_GRAPHS}" = "1" ] && [ "${#graph_paths[@]}" -gt 0 ]; then
            retry rsync -avhP "${graph_paths[@]}" "${HOST}:${REMOTE_STAGE_ROOT}/ops-bin/"
        fi
        retry ssh "${HOST}" "adb shell $(device_shell_escape "mkdir -p '${REMOTE_BIN_DIR}' '${REMOTE_LIB_DIR}' '${REMOTE_OPS_DIR}'")"
        echo "adb push '${REMOTE_STAGE_BIN_PATH}' '${REMOTE_BIN_PATH}'"
        retry ssh "${HOST}" "adb push '${REMOTE_STAGE_BIN_PATH}' '${REMOTE_BIN_PATH}'"
        retry ssh "${HOST}" "adb push '${REMOTE_STAGE_ROOT}/build-android/lib/.' '${REMOTE_LIB_DIR}/'"
        if [ "${PUSH_GRAPHS}" = "1" ] && [ "${#graph_paths[@]}" -gt 0 ]; then
            retry ssh "${HOST}" "adb push '${REMOTE_STAGE_ROOT}/ops-bin/.' '${REMOTE_OPS_DIR}/'"
        fi
        if [ "${PUSH_MODEL}" = "1" ]; then
            retry rsync -avhP "${LOCAL_MODEL_PATH}" "${HOST}:${REMOTE_STAGE_ROOT}/$(basename "${LOCAL_MODEL_PATH}")"
            retry ssh "${HOST}" "adb shell $(device_shell_escape "mkdir -p \"$(dirname "${DEVICE_MODEL_PATH}")\"") && adb push '${REMOTE_STAGE_ROOT}/$(basename "${LOCAL_MODEL_PATH}")' '${DEVICE_MODEL_PATH}'"
        fi
    else
        retry ssh "${HOST}" "mkdir -p '${REMOTE_BIN_DIR}' '${REMOTE_OPS_DIR}'"
        retry rsync -avhP "${LOCAL_BIN}" "${HOST}:${REMOTE_BIN_DIR}/llama-cli"
        if [ "${PUSH_GRAPHS}" = "1" ] && [ "${#graph_paths[@]}" -gt 0 ]; then
            retry rsync -avhP "${graph_paths[@]}" "${HOST}:${REMOTE_OPS_DIR}/"
        fi
    fi
}

run_remote() {
    PROFILE_PHASE=${PROFILE_PHASE:-prefill}
    run_remote_prefill_common 0
}

run_remote_debug() {
    PROFILE_PHASE=${PROFILE_PHASE:-prefill}
    run_remote_prefill_common 1
}

run_remote_prefill_common() {
    local enable_dump="${1:-0}"
    local family_env_args
    local env_args
    local remote_cmd
    local redirect_mode
    require_prefill_eval
    prepare_remote_prompt
    family_env_args=$(family_executor_env_args "${PREFILL_FAMILY_EXECUTOR}" "${PREFILL_FAMILY_PRELOAD}")

    redirect_mode="combined"
    if [ "${CLEAN_OUTPUT}" = "1" ]; then
        redirect_mode="split"
    fi
    env_args="
LD_LIBRARY_PATH='${REMOTE_LIB_DIR}:/system/lib64:/vendor/lib64'
ADSP_LIBRARY_PATH='${REMOTE_LIB_DIR}'
GGML_QNN_BIN_PATH='${REMOTE_OPS_DIR}/graph0.bin'
GGML_QNN_RUN_MNN_PREFILL=1
GGML_QNN_RUN_MNN_DECODE='${GGML_QNN_RUN_MNN_DECODE:-1}'
${family_env_args}
$(live_recurrent_env_args_prefill_dual)
$(aux_cache_env_args)
$(replay_dump_env_args "${enable_dump}")
GGML_QNN_DUMP_ATTENTION='${GGML_QNN_DUMP_ATTENTION}'
GGML_QNN_PROFILE='${GGML_QNN_PROFILE}'
GGML_QNN_DEBUG_DECODE='${GGML_QNN_DEBUG_DECODE}'
$(stage_profile_env_args)
GGML_QNN_DECODE_LOG_TOKENS='${GGML_QNN_DECODE_LOG_TOKENS}'"
    remote_cmd=$(run_remote_cmd 1 "${env_args}" "--ubatch-size '${UBATCH_SIZE}' -fa on " "${redirect_mode}")
    run_remote_command "run" "${remote_cmd}"
}

# TAR-based optimized pull function
pull_remote_tar() {
    echo "[pull] Using TAR method and extracting on remote host"

    if is_adb_mode; then
        local remote_tar_path="${REMOTE_STAGE_ROOT}/${DUMP_NAME}.tar.gz"

        retry ssh "${HOST}" "mkdir -p '${REMOTE_STAGE_ROOT}'"

        # Step 1: Pull files from device to staging area
        echo "[pull] Pulling files from device to staging area..."
        retry ssh "${HOST}" "
            rm -rf '${REMOTE_STAGE_ROOT}/${DUMP_NAME}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.log' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.out' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.stage.tsv' && \
            if adb shell test -d '${REMOTE_DUMP_DIR}'; then adb pull '${REMOTE_DUMP_DIR}' '${REMOTE_STAGE_ROOT}/'; fi && \
            if adb shell test -f '${REMOTE_LOG_PATH}'; then adb pull '${REMOTE_LOG_PATH}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.log'; fi && \
            if adb shell test -f '${REMOTE_STDOUT_PATH}'; then adb pull '${REMOTE_STDOUT_PATH}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.out'; fi && \
            if adb shell test -f '${REMOTE_PROFILE_PATH}'; then adb pull '${REMOTE_PROFILE_PATH}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.stage.tsv'; fi
        "

        # Step 1.5: Check what files were actually pulled
        echo "[pull] Checking available files on staging host..."
        local available_files
        available_files=$(ssh "${HOST}" "
            cd '${REMOTE_STAGE_ROOT}' && \
            find . -maxdepth 2 -name '${DUMP_NAME}*' -o -name '${DUMP_NAME}' -type d 2>/dev/null || true
        ")

        if [ -z "${available_files}" ]; then
            echo "[pull] No files found to transfer. Make sure to run 'run' or 'run-debug' first." >&2
            return 1
        fi

        echo "[pull] Available files for transfer: ${available_files}"

        # Step 2: Create archive on staging host
        echo "[pull] Creating archive on staging host..."
        if [ "${PULL_MODE}" = "full" ]; then
            retry ssh "${HOST}" "
                cd '${REMOTE_STAGE_ROOT}' && \
                files_to_archive='' && \
                [ -d '${DUMP_NAME}' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}\" && \
                [ -f '${DUMP_NAME}.log' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}.log\" && \
                [ -f '${DUMP_NAME}.out' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}.out\" && \
                [ -f '${DUMP_NAME}.stage.tsv' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}.stage.tsv\" && \
                if [ -n \"\${files_to_archive}\" ]; then \
                    tar -czf '${remote_tar_path}' \${files_to_archive} && \
                    echo 'TAR archive created successfully'; \
                else \
                    echo 'No files to archive' >&2 && false; \
                fi
            "
        else
            # Minimal mode with selective file inclusion - using simpler approach
            echo "[pull] Debug: Checking files on staging host..."
            ssh "${HOST}" "cd '${REMOTE_STAGE_ROOT}' && ls -la ${DUMP_NAME}* 2>/dev/null | head -10 || echo 'No dump files found'"

            retry ssh "${HOST}" "
                cd '${REMOTE_STAGE_ROOT}' || exit 1
                echo 'Working directory:' \$(pwd)
                echo 'Available files:'
                ls -la ${DUMP_NAME}* 2>/dev/null | head -5

                # Create comprehensive file list for minimal mode
                {
                    # Always include log files
                    [ -f '${DUMP_NAME}.log' ] && echo '${DUMP_NAME}.log'
                    [ -f '${DUMP_NAME}.out' ] && echo '${DUMP_NAME}.out'
                    [ -f '${DUMP_NAME}.stage.tsv' ] && echo '${DUMP_NAME}.stage.tsv'

                    # Include all dump directory content
                    if [ -d '${DUMP_NAME}' ]; then
                        # Include all directories first
                        find '${DUMP_NAME}' -type d 2>/dev/null
                        # Include all files
                        find '${DUMP_NAME}' -type f 2>/dev/null
                    fi
                } | sort | uniq > filelist.txt

                echo 'Files to archive (first 20):'
                head -20 filelist.txt
                echo \"Total files: \$(wc -l < filelist.txt)\"

                if [ -s filelist.txt ]; then
                    tar -czf '${remote_tar_path}' -T filelist.txt &&
                    echo 'TAR archive created successfully (minimal mode)' &&
                    ls -lh '${remote_tar_path}'
                else
                    echo 'No files to archive' >&2
                    exit 1
                fi

                rm -f filelist.txt
            "
        fi

        # Step 2.5: Verify archive was created
        if ! ssh "${HOST}" "test -f '${remote_tar_path}'"; then
            echo "[pull] ERROR: Failed to create archive on staging host" >&2
            return 1
        fi

        # Step 3: Extract archive on remote host and keep a stable compare directory
        echo "[pull] Extracting archive on remote host..."
        retry ssh "${HOST}" "
            cd '${REMOTE_STAGE_ROOT}' && \
            rm -rf '${REMOTE_PULL_DIR}' && \
            tar -xzf '${remote_tar_path}' && \
            mkdir -p '${REMOTE_PULL_DIR}' && \
            if [ -f '${DUMP_NAME}.log' ]; then cp -f '${DUMP_NAME}.log' '${REMOTE_PULL_DIR}/llama.log'; fi && \
            if [ -f '${DUMP_NAME}.out' ]; then cp -f '${DUMP_NAME}.out' '${REMOTE_PULL_DIR}/llama.out'; fi && \
            if [ -f '${DUMP_NAME}.stage.tsv' ]; then cp -f '${DUMP_NAME}.stage.tsv' '${REMOTE_PULL_DIR}/llama.stage.tsv'; fi && \
            rm -f '${remote_tar_path}'
        "

    else
        # Non-ADB mode
        local remote_tar_path="${REMOTE_ROOT}/${DUMP_NAME}.tar.gz"

        # Check what files are available on remote
        echo "[pull] Checking available files on remote..."
        local available_files
        available_files=$(ssh "${HOST}" "
            cd '${REMOTE_ROOT}' && \
            find . -maxdepth 2 -name '${DUMP_NAME}' -o -name '${DUMP_NAME}.log' -o -name '${DUMP_NAME}.out' -o -name '${DUMP_NAME}.stage.tsv' 2>/dev/null | sed 's|^\./||' || true
        ")

        if [ -z "${available_files}" ]; then
            echo "[pull] No files found to transfer. Make sure to run 'run' or 'run-debug' first." >&2
            return 1
        fi

        echo "[pull] Available files for transfer: ${available_files}"

        # Create archive on remote
        echo "[pull] Creating archive on remote..."
        if [ "${PULL_MODE}" = "full" ]; then
            retry ssh "${HOST}" "
                cd '${REMOTE_ROOT}' && \
                files_to_archive='' && \
                [ -d '${DUMP_NAME}' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}\" && \
                [ -f '${DUMP_NAME}.log' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}.log\" && \
                [ -f '${DUMP_NAME}.out' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}.out\" && \
                [ -f '${DUMP_NAME}.stage.tsv' ] && files_to_archive=\"\${files_to_archive} ${DUMP_NAME}.stage.tsv\" && \
                if [ -n \"\${files_to_archive}\" ]; then \
                    tar -czf '${DUMP_NAME}.tar.gz' \${files_to_archive} && \
                    echo 'TAR archive created successfully'; \
                else \
                    echo 'No files to archive' >&2 && false; \
                fi
            "
        else
            # Minimal mode with selective file inclusion - using simpler approach
            echo "[pull] Debug: Checking files on remote..."
            ssh "${HOST}" "cd '${REMOTE_ROOT}' && ls -la ${DUMP_NAME}* 2>/dev/null | head -10 || echo 'No dump files found'"

            retry ssh "${HOST}" "
                cd '${REMOTE_ROOT}' || exit 1
                echo 'Working directory:' \$(pwd)
                echo 'Available files:'
                ls -la ${DUMP_NAME}* 2>/dev/null | head -5

                # Create comprehensive file list for minimal mode
                {
                    # Always include log files
                    [ -f '${DUMP_NAME}.log' ] && echo '${DUMP_NAME}.log'
                    [ -f '${DUMP_NAME}.out' ] && echo '${DUMP_NAME}.out'
                    [ -f '${DUMP_NAME}.stage.tsv' ] && echo '${DUMP_NAME}.stage.tsv'

                    # Include all dump directory content
                    if [ -d '${DUMP_NAME}' ]; then
                        # Include all directories first
                        find '${DUMP_NAME}' -type d 2>/dev/null
                        # Include all files
                        find '${DUMP_NAME}' -type f 2>/dev/null
                    fi
                } | sort | uniq > filelist.txt

                echo 'Files to archive (first 20):'
                head -20 filelist.txt
                echo \"Total files: \$(wc -l < filelist.txt)\"

                if [ -s filelist.txt ]; then
                    tar -czf '${DUMP_NAME}.tar.gz' -T filelist.txt &&
                    echo 'TAR archive created successfully (minimal mode)' &&
                    ls -lh '${DUMP_NAME}.tar.gz'
                else
                    echo 'No files to archive' >&2
                    exit 1
                fi

                rm -f filelist.txt
            "
        fi

        # Verify archive was created
        if ! ssh "${HOST}" "test -f '${REMOTE_ROOT}/${DUMP_NAME}.tar.gz'"; then
            echo "[pull] ERROR: Failed to create archive on remote" >&2
            return 1
        fi

        # Extract archive on remote and keep a stable compare directory
        echo "[pull] Extracting archive on remote host..."
        retry ssh "${HOST}" "
            cd '${REMOTE_ROOT}' && \
            rm -rf '${REMOTE_PULL_DIR}' && \
            tar -xzf '${remote_tar_path}' && \
            mkdir -p '${REMOTE_PULL_DIR}' && \
            if [ -f '${DUMP_NAME}.log' ]; then cp -f '${DUMP_NAME}.log' '${REMOTE_PULL_DIR}/llama.log'; fi && \
            if [ -f '${DUMP_NAME}.out' ]; then cp -f '${DUMP_NAME}.out' '${REMOTE_PULL_DIR}/llama.out'; fi && \
            if [ -f '${DUMP_NAME}.stage.tsv' ]; then cp -f '${DUMP_NAME}.stage.tsv' '${REMOTE_PULL_DIR}/llama.stage.tsv'; fi && \
            rm -f '${remote_tar_path}'
        "
    fi

    echo "[pull] Remote extract completed: ${HOST}:${REMOTE_PULL_DIR}"
}

# Legacy RSYNC-based pull function
pull_remote_rsync() {
    rm -rf "${LOCAL_PULL_DIR}"
    mkdir -p "${LOCAL_PULL_DIR}"

    echo "[pull] Using RSYNC method (legacy)"

    if is_adb_mode; then
        retry ssh "${HOST}" "mkdir -p '${REMOTE_STAGE_ROOT}'"
        retry ssh "${HOST}" "
            rm -rf '${REMOTE_STAGE_ROOT}/${DUMP_NAME}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.log' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.out' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.stage.tsv' && \
            if adb shell test -d '${REMOTE_DUMP_DIR}'; then adb pull '${REMOTE_DUMP_DIR}' '${REMOTE_STAGE_ROOT}/'; fi && \
            adb pull '${REMOTE_LOG_PATH}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.log' && \
            if adb shell test -f '${REMOTE_STDOUT_PATH}'; then adb pull '${REMOTE_STDOUT_PATH}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.out'; fi && \
            if adb shell test -f '${REMOTE_PROFILE_PATH}'; then adb pull '${REMOTE_PROFILE_PATH}' '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.stage.tsv'; fi
        "
        if ssh "${HOST}" "test -d '${REMOTE_STAGE_ROOT}/${DUMP_NAME}'"; then
            if [ "${PULL_MODE}" = "full" ]; then
                retry rsync -avhP "${HOST}:${REMOTE_STAGE_ROOT}/${DUMP_NAME}/" "${LOCAL_PULL_DIR}/"
            else
                local -a minimal_args=()
                read -r -a minimal_args <<< "$(minimal_pull_rsync_args)"
                retry rsync -avhP --prune-empty-dirs \
                    "${minimal_args[@]}" \
                    "${HOST}:${REMOTE_STAGE_ROOT}/${DUMP_NAME}/" "${LOCAL_PULL_DIR}/"
            fi
        fi
        retry rsync -avhP "${HOST}:${REMOTE_STAGE_ROOT}/${DUMP_NAME}.log" "${LOCAL_PULL_DIR}/llama.log"
        if ssh "${HOST}" "test -f '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.out'"; then
            retry rsync -avhP "${HOST}:${REMOTE_STAGE_ROOT}/${DUMP_NAME}.out" "${LOCAL_PULL_DIR}/llama.out"
        fi
        if ssh "${HOST}" "test -f '${REMOTE_STAGE_ROOT}/${DUMP_NAME}.stage.tsv'"; then
            retry rsync -avhP "${HOST}:${REMOTE_STAGE_ROOT}/${DUMP_NAME}.stage.tsv" "${LOCAL_PULL_DIR}/llama.stage.tsv"
        fi
    else
        if ssh "${HOST}" "test -d '${REMOTE_DUMP_DIR}'"; then
            if [ "${PULL_MODE}" = "full" ]; then
                retry rsync -avhP "${HOST}:${REMOTE_DUMP_DIR}/" "${LOCAL_PULL_DIR}/"
            else
                local -a minimal_args=()
                read -r -a minimal_args <<< "$(minimal_pull_rsync_args)"
                retry rsync -avhP --prune-empty-dirs \
                    "${minimal_args[@]}" \
                    "${HOST}:${REMOTE_DUMP_DIR}/" "${LOCAL_PULL_DIR}/"
            fi
        fi
        retry rsync -avhP "${HOST}:${REMOTE_LOG_PATH}" "${LOCAL_PULL_DIR}/llama.log"
        if ssh "${HOST}" "test -f '${REMOTE_STDOUT_PATH}'"; then
            retry rsync -avhP "${HOST}:${REMOTE_STDOUT_PATH}" "${LOCAL_PULL_DIR}/llama.out"
        fi
        if ssh "${HOST}" "test -f '${REMOTE_PROFILE_PATH}'"; then
            retry rsync -avhP "${HOST}:${REMOTE_PROFILE_PATH}" "${LOCAL_PULL_DIR}/llama.stage.tsv"
        fi
    fi
}

# Main pull function that delegates to the appropriate method
pull_remote() {
    case "${PULL_METHOD}" in
        tar)
            if ! pull_remote_tar; then
                echo "[pull] TAR method failed, falling back to RSYNC method..." >&2
                pull_remote_rsync
            fi
            ;;
        rsync)
            pull_remote_rsync
            ;;
        *)
            echo "Unknown PULL_METHOD: ${PULL_METHOD}. Using default (tar)." >&2
            if ! pull_remote_tar; then
                echo "[pull] TAR method failed, falling back to RSYNC method..." >&2
                pull_remote_rsync
            fi
            ;;
    esac

    summarize_profile_local
    summarize_prefill_profile_local
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
    run-debug)
        run_remote_debug
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
    all-debug)
        build_local
        push_bin
        run_remote_debug
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
