#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-/home/chensm22/MNN/transformers/llm/export/propmt_256_0.txt}"
HOST="${HOST:-reck}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
MNN_DUMP_NAME="${MNN_DUMP_NAME:-mnn-prefill-compare}"
LLAMA_DUMP_NAME="${LLAMA_DUMP_NAME:-llama-prefill-compare}"

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
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh pull

cat <<EOF
[6/7] Compare prefill chunk 0
EOF
/home/chensm22/MNN/transformers/llm/export/run_compare.sh compare --chunk-index 0

cat <<EOF
[7/7] Compare prefill chunk 1
EOF
/home/chensm22/MNN/transformers/llm/export/run_compare.sh compare --chunk-index 1
