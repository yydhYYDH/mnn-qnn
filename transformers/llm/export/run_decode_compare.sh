#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompt_128.txt}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompt_chinese_hello.txt}"
HOST="${HOST:-reck}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
MNN_DUMP_NAME="${MNN_DUMP_NAME:-mnn-decode-compare}"
LLAMA_DUMP_NAME="${LLAMA_DUMP_NAME:-llama-decode-compare}"

# Compare decode by ordinal step: graph0-run-0000 <-> decode-token-0000,
# graph0-run-0001 <-> decode-token-0001, etc.
# Keep N_PREDICT >= 2 so we can compare more than one decode step when needed.
N_PREDICT="${N_PREDICT:-2}"
TOKEN_INDEX="${TOKEN_INDEX:-0}"
RUN_ID="${RUN_ID:-${TOKEN_INDEX}}"
GGML_QNN_DUMP_ATTENTION="${GGML_QNN_DUMP_ATTENTION:-0}"
GGML_QNN_DEBUG_DECODE="${GGML_QNN_DEBUG_DECODE:-0}"

if [ ! -f "${PROMPT_FILE}" ]; then
    echo "prompt file not found: ${PROMPT_FILE}" >&2
    exit 1
fi

if [ "${N_PREDICT}" -lt 1 ]; then
    cat >&2 <<EOF
run_decode_compare.sh requires N_PREDICT>=1.

Use N_PREDICT=2 or larger if you want to compare multiple decode steps.
EOF
    exit 1
fi

printf -v TOKEN_DIR "decode-token-%04d" "${TOKEN_INDEX+1}"

MNN_REMOTE_ROOT="/home/reck/mnn_qwen3/${MNN_DUMP_NAME}"
MNN_REMOTE_DECODE_ROOT="${MNN_REMOTE_ROOT}/decode"
LLAMA_REMOTE_ROOT="/home/reck/llama.cpp-test/${LLAMA_DUMP_NAME}"
LLAMA_REMOTE_TOKEN_DIR="${LLAMA_REMOTE_ROOT}/${TOKEN_DIR}"

cat <<EOF
[decode-compare] Configuration
  prompt file: ${PROMPT_FILE}
  host: ${HOST}
  device root: ${DEVICE_ROOT}
  mnn dump: ${MNN_DUMP_NAME}
  llama dump: ${LLAMA_DUMP_NAME}
  n_predict: ${N_PREDICT}
  token index: ${TOKEN_INDEX}
  mnn run id: ${RUN_ID}
  llama token dir: ${LLAMA_REMOTE_TOKEN_DIR}
EOF

cat <<EOF
[0/7] Remove existing decode compare dumps on reck
EOF
ssh "${HOST}" "rm -rf '${MNN_REMOTE_ROOT}' '${MNN_REMOTE_ROOT}.log' '${MNN_REMOTE_ROOT}.stage.tsv' '${LLAMA_REMOTE_ROOT}' '${LLAMA_REMOTE_ROOT}.log' '${LLAMA_REMOTE_ROOT}.out' '${LLAMA_REMOTE_ROOT}.stage.tsv'"

cat <<EOF
[1/7] Build MNN android binaries
EOF
cmake --build "${ROOT_DIR}/project/android/build_64" --target MNN llm_demo -j 4

cat <<EOF
[2/7] Build llama.cpp android binaries
EOF
"${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" build

cat <<EOF
[3/7] Run MNN dump with at least one decode step
EOF
LOCAL_PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
"${SCRIPT_DIR}/run_mnn_qnn_decode_dump.sh" push
LOCAL_PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
MNN_LLM_PROMPT_WHOLE_FILE=1 \
N_PREDICT="${N_PREDICT}" \
PROFILE_STAGE=1 \
DUMP_NAME="${MNN_DUMP_NAME}" \
"${SCRIPT_DIR}/run_mnn_qnn_decode_dump.sh" run

cat <<EOF
[4/7] Run llama.cpp replay dump with decode recording enabled
EOF
PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
"${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" push
PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
N_PREDICT="${N_PREDICT}" \
PROFILE_STAGE=1 \
DUMP_NAME="${LLAMA_DUMP_NAME}" \
GGML_QNN_DUMP_ATTENTION="${GGML_QNN_DUMP_ATTENTION}" \
GGML_QNN_DEBUG_DECODE="${GGML_QNN_DEBUG_DECODE}" \
"${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" run-debug

cat <<EOF
[5/7] Stage remote dumps on reck
EOF
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
DUMP_NAME="${MNN_DUMP_NAME}" \
"${SCRIPT_DIR}/run_mnn_qnn_decode_dump.sh" stage-remote
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
DUMP_NAME="${LLAMA_DUMP_NAME}" \
"${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" stage-remote

cat <<EOF
[6/7] Verify remote decode directories
EOF
ssh "${HOST}" "
    test -d '${MNN_REMOTE_DECODE_ROOT}' || { echo '[decode-compare] missing MNN decode root: ${MNN_REMOTE_DECODE_ROOT}' >&2; exit 1; }
    test -d '${LLAMA_REMOTE_TOKEN_DIR}' || { echo '[decode-compare] missing llama token dir: ${LLAMA_REMOTE_TOKEN_DIR}' >&2; exit 1; }
    echo '[decode-compare] MNN decode root:' '${MNN_REMOTE_DECODE_ROOT}'
    find '${MNN_REMOTE_DECODE_ROOT}' -maxdepth 1 -type d | sort | sed -n '1,10p'
    echo '[decode-compare] llama token dir:' '${LLAMA_REMOTE_TOKEN_DIR}'
    find '${LLAMA_REMOTE_TOKEN_DIR}' -maxdepth 1 -type f | sort | sed -n '1,10p'
"

cat <<EOF
[7/7] Compare first decode token on reck
EOF
"${SCRIPT_DIR}/run_compare.sh" compare \
    --mnn-root "${MNN_REMOTE_DECODE_ROOT}" \
    --llama-dir "${LLAMA_REMOTE_TOKEN_DIR}" \
    --run-id "${RUN_ID}"
echo "${SCRIPT_DIR}/run_compare.sh" compare \
    --mnn-root "${MNN_REMOTE_DECODE_ROOT}" \
    --llama-dir "${LLAMA_REMOTE_TOKEN_DIR}" \
    --run-id "${RUN_ID}"
cat <<EOF

[decode-compare] Done.

Useful knobs:
  TOKEN_INDEX=0 RUN_ID=0       compare the first decode step (default)
  TOKEN_INDEX=1 RUN_ID=1       compare the second decode step
  N_PREDICT=3 TOKEN_INDEX=2 RUN_ID=2
                               compare the third decode step
  GGML_QNN_DUMP_ATTENTION=1    also dump attn-layer*/kv_cache*/Qcur/Kcur/Vcur

Minimal llama.cpp env for true decode dump:
  GGML_QNN_REPLAY_DUMP_DIR=<dump dir>
  GGML_QNN_RUN_MNN_DECODE=1
  N_PREDICT>=2
EOF
