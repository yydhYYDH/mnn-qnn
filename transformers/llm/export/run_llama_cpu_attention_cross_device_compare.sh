#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

HOST="${HOST:-reck}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
RECK_LLAMA_DIR="${RECK_LLAMA_DIR:-/home/reck/llama.cpp-test}"

COMPARE_SCRIPT_LOCAL="${COMPARE_SCRIPT_LOCAL:-${ROOT_DIR}/transformers/llm/export/compare_tensor_dump.py}"

SEQ_LEN="${SEQ_LEN:-32}"
KV_LEN="${KV_LEN:-32}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
CAUSAL="${CAUSAL:-1}"
LLAMA_ATTN_KERNEL="${LLAMA_ATTN_KERNEL:-ggml}"
LLAMA_NUM_THREADS="${LLAMA_NUM_THREADS:-1}"

X86_DUMP_NAME="${X86_DUMP_NAME:-llama-cpu-attention-x86}"
PHONE_DUMP_NAME="${PHONE_DUMP_NAME:-llama-cpu-attention-phone}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-${ROOT_DIR}/.tmp/llama-attention-cross-device}"

LLAMA_ROOT="${LLAMA_ROOT:-${ROOT_DIR}/llama.cpp}"
LLAMA_X86_BUILD_DIR="${LLAMA_X86_BUILD_DIR:-${LLAMA_ROOT}/build-x86}"
LLAMA_ANDROID_BUILD_DIR="${LLAMA_ANDROID_BUILD_DIR:-${LLAMA_ROOT}/build-android}"

X86_BIN="${X86_BIN:-${LLAMA_X86_BUILD_DIR}/bin/test-attention-unit}"
ANDROID_BIN="${ANDROID_BIN:-${LLAMA_ANDROID_BUILD_DIR}/bin/test-attention-unit}"

LOCAL_X86_DIR="${LOCAL_STAGE_ROOT}/${X86_DUMP_NAME}"
LOCAL_PHONE_DIR="${LOCAL_STAGE_ROOT}/${PHONE_DUMP_NAME}"
REMOTE_STAGE_BIN="${RECK_LLAMA_DIR}/test-attention-unit"
PHONE_DUMP_DIR="${DEVICE_ROOT}/${PHONE_DUMP_NAME}"
REMOTE_PHONE_STAGE_DIR="${RECK_LLAMA_DIR}/${PHONE_DUMP_NAME}"

retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        if [ "${n}" -ge 5 ]; then
            echo "command failed after ${n} attempts: $*" >&2
            return 1
        fi
        sleep 2
    done
}

ensure_x86_cmake() {
    if [ -f "${LLAMA_X86_BUILD_DIR}/CMakeCache.txt" ]; then
        return 0
    fi

    cat <<EOF
[1/8] Configure llama.cpp x86 build directory
EOF
    cmake \
      -S "${LLAMA_ROOT}" \
      -B "${LLAMA_X86_BUILD_DIR}" \
      -DGGML_QNN=OFF \
      -DLLAMA_CURL=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=Release
}

ensure_android_cmake() {
    if [ -f "${LLAMA_ANDROID_BUILD_DIR}/CMakeCache.txt" ]; then
        return 0
    fi

    cat <<EOF
[3/8] Configure llama.cpp android build directory
EOF
    "${LLAMA_ROOT}/build-android.sh" cmake
}

mkdir -p "${LOCAL_STAGE_ROOT}"
rm -rf "${LOCAL_X86_DIR}" "${LOCAL_PHONE_DIR}"

cat <<EOF
[0/8] Clean staged dumps
EOF
retry ssh "${HOST}" "rm -rf '${REMOTE_PHONE_STAGE_DIR}' '${PHONE_DUMP_DIR}' && mkdir -p '${RECK_LLAMA_DIR}'"

ensure_x86_cmake

cat <<EOF
[2/8] Build llama.cpp x86 attention unit test
EOF
cmake --build "${LLAMA_X86_BUILD_DIR}" --target test-attention-unit -j 4

ensure_android_cmake

cat <<EOF
[4/8] Build llama.cpp android attention unit test
EOF
cmake --build "${LLAMA_ANDROID_BUILD_DIR}" --target test-attention-unit -j 4

cat <<EOF
[5/8] Run llama.cpp CPU attention on x86
EOF
"${X86_BIN}" \
  --seq-len "${SEQ_LEN}" \
  --kv-len "${KV_LEN}" \
  --num-heads "${NUM_HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --causal "${CAUSAL}" \
  --num-threads "${LLAMA_NUM_THREADS}" \
  --kernel "${LLAMA_ATTN_KERNEL}" \
  --dump-dir "${LOCAL_X86_DIR}"

cat <<EOF
[6/8] Push android binary and run llama.cpp CPU attention on phone
EOF
retry rsync -avhP "${ANDROID_BIN}" "${HOST}:${REMOTE_STAGE_BIN}"
retry ssh "${HOST}" "
    adb shell 'mkdir -p \"${DEVICE_ROOT}\"' &&
    adb push '${REMOTE_STAGE_BIN}' '${DEVICE_ROOT}/test-attention-unit' &&
    adb shell '
        rm -rf \"${PHONE_DUMP_DIR}\" &&
        cd \"${DEVICE_ROOT}\" &&
        env \
          LD_LIBRARY_PATH=\"${DEVICE_ROOT}\" \
          ./test-attention-unit \
          --seq-len ${SEQ_LEN} \
          --kv-len ${KV_LEN} \
          --num-heads ${NUM_HEADS} \
          --head-dim ${HEAD_DIM} \
          --causal ${CAUSAL} \
          --num-threads ${LLAMA_NUM_THREADS} \
          --kernel ${LLAMA_ATTN_KERNEL} \
          --dump-dir \"${PHONE_DUMP_DIR}\"
    '
"

cat <<EOF
[7/8] Pull phone dumps back to local machine
EOF
retry ssh "${HOST}" "
    rm -rf '${REMOTE_PHONE_STAGE_DIR}' &&
    mkdir -p '${RECK_LLAMA_DIR}' &&
    adb pull '${PHONE_DUMP_DIR}' '${RECK_LLAMA_DIR}/'
"
retry rsync -avhP "${HOST}:${REMOTE_PHONE_STAGE_DIR}/" "${LOCAL_PHONE_DIR}/"

cat <<EOF
[8/8] Compare x86 vs phone attention dumps
EOF
python3 "${COMPARE_SCRIPT_LOCAL}" \
  --lhs "${LOCAL_X86_DIR}/attn-query.bin" \
  --rhs "${LOCAL_PHONE_DIR}/attn-query.bin" \
  --dtype f32 \
  --shape "1,${SEQ_LEN},${NUM_HEADS},${HEAD_DIM}"
python3 "${COMPARE_SCRIPT_LOCAL}" \
  --lhs "${LOCAL_X86_DIR}/attn-key.bin" \
  --rhs "${LOCAL_PHONE_DIR}/attn-key.bin" \
  --dtype f32 \
  --shape "1,${KV_LEN},${NUM_HEADS},${HEAD_DIM}"
python3 "${COMPARE_SCRIPT_LOCAL}" \
  --lhs "${LOCAL_X86_DIR}/attn-value.bin" \
  --rhs "${LOCAL_PHONE_DIR}/attn-value.bin" \
  --dtype f32 \
  --shape "1,${KV_LEN},${NUM_HEADS},${HEAD_DIM}"
python3 "${COMPARE_SCRIPT_LOCAL}" \
  --lhs "${LOCAL_X86_DIR}/attn-mask.bin" \
  --rhs "${LOCAL_PHONE_DIR}/attn-mask.bin" \
  --dtype f32 \
  --shape "1,1,${SEQ_LEN},${KV_LEN}"
python3 "${COMPARE_SCRIPT_LOCAL}" \
  --lhs "${LOCAL_X86_DIR}/attn-output.bin" \
  --rhs "${LOCAL_PHONE_DIR}/attn-output.bin" \
  --dtype f32 \
  --shape "1,${SEQ_LEN},${NUM_HEADS},${HEAD_DIM}"
