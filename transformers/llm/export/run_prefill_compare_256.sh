#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-/home/chensm22/MNN/transformers/llm/export/propmt_80.txt}"
HOST="${HOST:-reck}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
MNN_DUMP_NAME="${MNN_DUMP_NAME:-mnn-prefill-compare}"
LLAMA_DUMP_NAME="${LLAMA_DUMP_NAME:-llama-prefill-compare}"

cat <<EOF
[0/6] Remove existing prefill dumps
EOF
ssh reck 'rm -rf /home/reck/mnn_qwen3/mnn-prefill-compare /home/reck/llama.cpp-test/llama-prefill-compare'

cat <<EOF
[1/6] Build MNN android binaries
EOF
cmake --build /home/chensm22/MNN/project/android/build_64 --target MNN llm_demo -j 4

cat <<EOF
[2/6] Build llama.cpp android binaries
EOF
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh build

cat <<EOF
[3/6] Run MNN prefill dump
EOF
LOCAL_PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
/home/chensm22/MNN/transformers/llm/export/run_mnn_qnn_decode_dump.sh push
LOCAL_PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
MNN_LLM_PROMPT_WHOLE_FILE=1 \
N_PREDICT=1 \
PROFILE_STAGE=1 \
DUMP_NAME="${MNN_DUMP_NAME}" \
/home/chensm22/MNN/transformers/llm/export/run_mnn_qnn_decode_dump.sh run

cat <<EOF
[4/6] Run llama.cpp prefill dump
EOF
PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh push
PROMPT_FILE="${PROMPT_FILE}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
N_PREDICT=1 \
PROFILE_STAGE=1 \
DUMP_NAME="${LLAMA_DUMP_NAME}" \
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh run-debug

cat <<EOF
[5/7] Stage remote dumps on reck
EOF
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
DUMP_NAME="${MNN_DUMP_NAME}" \
/home/chensm22/MNN/transformers/llm/export/run_mnn_qnn_decode_dump.sh stage-remote
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
DUMP_NAME="${LLAMA_DUMP_NAME}" \
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh stage-remote

cat <<EOF
[6/7] Compare prefill chunk 0
EOF
/home/chensm22/MNN/transformers/llm/export/run_compare.sh compare --chunk-index 0

cat <<EOF
[7/7] Compare prefill chunk 1
EOF
/home/chensm22/MNN/transformers/llm/export/run_compare.sh compare --chunk-index 1

if [ "${RUN_ATTN_UNIT_COMPARE_FROM_PREFILL:-0}" = "1" ]; then
cat <<EOF
[8/8] Replay prefill q/k/v through attention unit compare
EOF
INPUT_MODE="prefill" \
ATTN_INPUT_SOURCE_KIND="${ATTN_INPUT_SOURCE_KIND:-llama}" \
ATTN_INPUT_SOURCE_ROOT="${ATTN_INPUT_SOURCE_ROOT:-}" \
ATTN_INPUT_GRAPH="${ATTN_INPUT_GRAPH:-}" \
ATTN_INPUT_RUN_ID="${ATTN_INPUT_RUN_ID:-0}" \
HOST="${HOST}" \
DEVICE_ROOT="${DEVICE_ROOT}" \
/home/chensm22/MNN/transformers/llm/export/run_attention_unit_compare.sh
fi
