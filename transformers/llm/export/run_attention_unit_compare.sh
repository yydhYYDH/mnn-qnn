#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-reck}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
RECK_MNN_DIR="${RECK_MNN_DIR:-/home/reck/mnn_qwen3}"
RECK_LLAMA_DIR="${RECK_LLAMA_DIR:-/home/reck/llama.cpp-test}"
COMPARE_SCRIPT_LOCAL="${COMPARE_SCRIPT_LOCAL:-/home/chensm22/MNN/transformers/llm/export/compare_tensor_dump.py}"
COMPARE_SCRIPT_REMOTE="${COMPARE_SCRIPT_REMOTE:-${RECK_LLAMA_DIR}/compare_tensor_dump.py}"
PREPARE_INPUT_LOCAL="${PREPARE_INPUT_LOCAL:-/home/chensm22/MNN/transformers/llm/export/prepare_attention_unit_input.py}"
PREPARE_INPUT_REMOTE="${PREPARE_INPUT_REMOTE:-${RECK_LLAMA_DIR}/prepare_attention_unit_input.py}"

SEQ_LEN="${SEQ_LEN:-32}"
KV_LEN="${KV_LEN:-32}"
NUM_HEADS="${NUM_HEADS:-32}"
NUM_KV_HEADS="${NUM_KV_HEADS:-${NUM_HEADS}}"
HEAD_DIM="${HEAD_DIM:-128}"
CAUSAL="${CAUSAL:-1}"
ATTENTION_OPTION="${ATTENTION_OPTION:-8}"
LLAMA_ATTN_KERNEL="${LLAMA_ATTN_KERNEL:-mnn}"
LLAMA_NUM_THREADS="${LLAMA_NUM_THREADS:-1}"
INPUT_MODE="${INPUT_MODE:-generated}"
ATTN_INPUT_SOURCE_KIND="${ATTN_INPUT_SOURCE_KIND:-llama}"
ATTN_INPUT_SOURCE_ROOT="${ATTN_INPUT_SOURCE_ROOT:-}"
ATTN_INPUT_GRAPH="${ATTN_INPUT_GRAPH:-}"
ATTN_INPUT_RUN_ID="${ATTN_INPUT_RUN_ID:-0}"
ATTN_INPUT_NAME="${ATTN_INPUT_NAME:-attention-unit-input}"
DEFAULT_LLAMA_PREFILL_DIR="${DEFAULT_LLAMA_PREFILL_DIR:-${RECK_LLAMA_DIR}/llama-prefill-compare/decode-token-0000}"
DEFAULT_MNN_PREFILL_ROOT="${DEFAULT_MNN_PREFILL_ROOT:-${RECK_MNN_DIR}/mnn-prefill-compare/prefill}"

MNN_DUMP_NAME="${MNN_DUMP_NAME:-mnn-attention-unit}"
LLAMA_DUMP_NAME="${LLAMA_DUMP_NAME:-llama-attention-unit}"

MNN_BUILD_DIR="${MNN_BUILD_DIR:-/home/chensm22/MNN/project/android/build_64}"
LLAMA_ROOT="${LLAMA_ROOT:-/home/chensm22/MNN/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-/home/chensm22/MNN/llama.cpp/build-android}"
ANDROID_NDK="${ANDROID_NDK:-/home/chensm22/android-ndk-r28c}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/chensm22/qairt/2.41.0.251128}"
GGML_MNN_REPO_ROOT="${GGML_MNN_REPO_ROOT:-/home/chensm22/MNN}"
GGML_MNN_LIBRARY="${GGML_MNN_LIBRARY:-${MNN_BUILD_DIR}/libMNN.so}"
GGML_MNN_EXPRESS_LIBRARY="${GGML_MNN_EXPRESS_LIBRARY:-${MNN_BUILD_DIR}/libMNN_Express.so}"
FORCE_LLAMA_CMAKE="${FORCE_LLAMA_CMAKE:-0}"

MNN_BIN="${MNN_BUILD_DIR}/run_test.out"
LLAMA_BIN="${LLAMA_BUILD_DIR}/bin/test-attention-unit"

MNN_DEVICE_DIR="${DEVICE_ROOT}/${MNN_DUMP_NAME}"
LLAMA_DEVICE_DIR="${DEVICE_ROOT}/${LLAMA_DUMP_NAME}"
ATTN_INPUT_REMOTE_DIR="${RECK_LLAMA_DIR}/${ATTN_INPUT_NAME}"
ATTN_INPUT_DEVICE_DIR="${DEVICE_ROOT}/${ATTN_INPUT_NAME}"

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

if [ "${INPUT_MODE}" = "prefill" ]; then
    if [ -z "${ATTN_INPUT_SOURCE_ROOT}" ]; then
        if [ "${ATTN_INPUT_SOURCE_KIND}" = "llama" ]; then
            ATTN_INPUT_SOURCE_ROOT="${DEFAULT_LLAMA_PREFILL_DIR}"
        else
            ATTN_INPUT_SOURCE_ROOT="${DEFAULT_MNN_PREFILL_ROOT}"
        fi
    fi
    if [ -z "${ATTN_INPUT_GRAPH}" ]; then
        ATTN_INPUT_GRAPH="graph1_0"
    fi
fi

MNN_INPUT_ENV_ARGS=()
LLAMA_INPUT_ARGS=()
if [ "${INPUT_MODE}" = "prefill" ]; then
    MNN_INPUT_ENV_ARGS=("MNN_ATTN_UNIT_INPUT_DIR=\"${ATTN_INPUT_DEVICE_DIR}\"")
    LLAMA_INPUT_ARGS=("--input-dir" "\"${ATTN_INPUT_DEVICE_DIR}\"")
fi

cat <<EOF
[0/7] Clean old staged results on reck
EOF
retry ssh "${HOST}" "rm -rf '${RECK_MNN_DIR}/${MNN_DUMP_NAME}' '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}'"
retry ssh "${HOST}" "mkdir -p '${RECK_LLAMA_DIR}'"
retry rsync -avhP "${COMPARE_SCRIPT_LOCAL}" "${HOST}:${COMPARE_SCRIPT_REMOTE}"
if [ "${INPUT_MODE}" = "prefill" ]; then
    retry rsync -avhP "${PREPARE_INPUT_LOCAL}" "${HOST}:${PREPARE_INPUT_REMOTE}"
    retry ssh "${HOST}" "rm -rf '${ATTN_INPUT_REMOTE_DIR}'"
fi

cat <<EOF
[1/7] Build MNN attention unit test
EOF
cmake --build "${MNN_BUILD_DIR}" --target run_test.out -j 4

cat <<EOF
[2/7] Build llama.cpp attention unit test
EOF
if [ "${FORCE_LLAMA_CMAKE}" = "1" ] || [ ! -f "${LLAMA_BUILD_DIR}/CMakeCache.txt" ]; then
    cmake \
      -S "${LLAMA_ROOT}" \
      -B "${LLAMA_BUILD_DIR}" \
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DCMAKE_C_FLAGS="-march=armv8.7a" \
      -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
      -DGGML_OPENMP=OFF \
      -DGGML_LLAMAFILE=OFF \
      -DGGML_MNN_ATTENTION=ON \
      -DGGML_MNN_REPO_ROOT="${GGML_MNN_REPO_ROOT}" \
      -DGGML_MNN_LIBRARY="${GGML_MNN_LIBRARY}" \
      -DGGML_MNN_EXPRESS_LIBRARY="${GGML_MNN_EXPRESS_LIBRARY}" \
      -DGGML_QNN=ON \
      -DQNN_SDK_ROOT="${QNN_SDK_ROOT}" \
      -DLLAMA_CURL=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "${LLAMA_BUILD_DIR}" --target test-attention-unit -j 4

cat <<EOF
[3/7] Push MNN attention unit test to phone via reck
EOF
retry ssh "${HOST}" "mkdir -p '${RECK_MNN_DIR}'"
retry rsync -avhP "${MNN_BIN}" "${HOST}:${RECK_MNN_DIR}/run_test.out"
retry rsync -avhP "${MNN_BUILD_DIR}/libMNN.so" "${HOST}:${RECK_MNN_DIR}/libMNN.so"
retry rsync -avhP "${MNN_BUILD_DIR}/libMNN_Express.so" "${HOST}:${RECK_MNN_DIR}/libMNN_Express.so"
if [ -f "${MNN_BUILD_DIR}/libplugin_matmul.so" ]; then
    retry rsync -avhP "${MNN_BUILD_DIR}/libplugin_matmul.so" "${HOST}:${RECK_MNN_DIR}/libplugin_matmul.so"
fi
retry ssh "${HOST}" "
    adb shell 'mkdir -p \"${DEVICE_ROOT}\"' &&
    adb push '${RECK_MNN_DIR}/run_test.out' '${DEVICE_ROOT}/run_test.out' &&
    adb push '${RECK_MNN_DIR}/libMNN.so' '${DEVICE_ROOT}/libMNN.so' &&
    adb push '${RECK_MNN_DIR}/libMNN_Express.so' '${DEVICE_ROOT}/libMNN_Express.so' &&
    if [ -f '${RECK_MNN_DIR}/libplugin_matmul.so' ]; then adb push '${RECK_MNN_DIR}/libplugin_matmul.so' '${DEVICE_ROOT}/libplugin_matmul.so'; fi
"

cat <<EOF
[4/7] Push llama.cpp attention unit test to phone via reck
EOF
retry ssh "${HOST}" "mkdir -p '${RECK_LLAMA_DIR}'"
retry rsync -avhP "${LLAMA_BIN}" "${HOST}:${RECK_LLAMA_DIR}/test-attention-unit"
retry ssh "${HOST}" "
    adb shell 'mkdir -p \"${DEVICE_ROOT}\"' &&
    adb push '${RECK_LLAMA_DIR}/test-attention-unit' '${DEVICE_ROOT}/test-attention-unit'
"

if [ "${INPUT_MODE}" = "prefill" ]; then
cat <<EOF
[4.5/7] Prepare prefill q/k/v input bundle on reck
EOF
retry ssh "${HOST}" "
    python3 '${PREPARE_INPUT_REMOTE}' \
      --source-kind '${ATTN_INPUT_SOURCE_KIND}' \
      --source-root '${ATTN_INPUT_SOURCE_ROOT}' \
      --graph '${ATTN_INPUT_GRAPH}' \
      --run-id '${ATTN_INPUT_RUN_ID}' \
      --output-dir '${ATTN_INPUT_REMOTE_DIR}'
"
retry ssh "${HOST}" "
    adb shell 'rm -rf \"${ATTN_INPUT_DEVICE_DIR}\" && mkdir -p \"${DEVICE_ROOT}\"' &&
    adb push '${ATTN_INPUT_REMOTE_DIR}' '${DEVICE_ROOT}/'
"
fi

cat <<EOF
[5/7] Run both attention unit tests on phone
EOF
retry ssh "${HOST}" "
    adb shell '
        rm -rf \"${MNN_DEVICE_DIR}\" \"${LLAMA_DEVICE_DIR}\" &&
        cd \"${DEVICE_ROOT}\" &&
        env \
          LD_LIBRARY_PATH=\"${DEVICE_ROOT}\" \
          MNN_ATTN_UNIT_DUMP_DIR=\"${MNN_DEVICE_DIR}\" \
          MNN_ATTN_UNIT_SEQ_LEN=\"${SEQ_LEN}\" \
          MNN_ATTN_UNIT_KV_LEN=\"${KV_LEN}\" \
          MNN_ATTN_UNIT_NUM_HEADS=\"${NUM_HEADS}\" \
          MNN_ATTN_UNIT_NUM_KV_HEADS=\"${NUM_KV_HEADS}\" \
          MNN_ATTN_UNIT_HEAD_DIM=\"${HEAD_DIM}\" \
          MNN_ATTN_UNIT_ATTENTION_OPTION=\"${ATTENTION_OPTION}\" \
          ${MNN_INPUT_ENV_ARGS[*]} \
          ./run_test.out op/attention_unit_dump 0 0 1
    '
"
retry ssh "${HOST}" "
    adb shell '
        rm -rf \"${LLAMA_DEVICE_DIR}\" &&
        cd \"${DEVICE_ROOT}\" &&
        env \
          LD_LIBRARY_PATH=\"${DEVICE_ROOT}\" \
          ./test-attention-unit \
          --seq-len ${SEQ_LEN} \
          --kv-len ${KV_LEN} \
          --num-heads ${NUM_HEADS} \
          --num-kv-heads ${NUM_KV_HEADS} \
          --head-dim ${HEAD_DIM} \
          --causal ${CAUSAL} \
          --num-threads ${LLAMA_NUM_THREADS} \
          --kernel ${LLAMA_ATTN_KERNEL} \
          --dump-dir \"${LLAMA_DEVICE_DIR}\" \
          ${LLAMA_INPUT_ARGS[*]}
    '
"

cat <<EOF
[6/7] Pull attention dumps from phone to reck
EOF
retry ssh "${HOST}" "
    mkdir -p '${RECK_MNN_DIR}' '${RECK_LLAMA_DIR}' &&
    rm -rf '${RECK_MNN_DIR}/${MNN_DUMP_NAME}' '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}' &&
    adb pull '${MNN_DEVICE_DIR}' '${RECK_MNN_DIR}/' &&
    adb pull '${LLAMA_DEVICE_DIR}' '${RECK_LLAMA_DIR}/'
"

cat <<EOF
[7/7] Compare q/k/v/mask/output on reck
EOF
ssh "${HOST}" "
    if [ '${INPUT_MODE}' = 'prefill' ] && test -f '${ATTN_INPUT_REMOTE_DIR}/attention_input.env'; then
        . '${ATTN_INPUT_REMOTE_DIR}/attention_input.env'
    fi &&
    q_shape=\$(grep '^shape=' '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-query.meta.txt' | cut -d= -f2) &&
    k_shape=\$(grep '^shape=' '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-key.meta.txt' | cut -d= -f2) &&
    v_shape=\$(grep '^shape=' '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-value.meta.txt' | cut -d= -f2) &&
    mask_shape=\$(grep '^shape=' '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-mask.meta.txt' | cut -d= -f2) &&
    out_shape=\$(grep '^shape=' '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-output.meta.txt' | cut -d= -f2) &&
    python3 '${COMPARE_SCRIPT_REMOTE}' \
      --lhs '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-query.bin' \
      --rhs '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}/attn-query.bin' \
      --dtype f32 \
      --shape \"\${q_shape}\" &&
    python3 '${COMPARE_SCRIPT_REMOTE}' \
      --lhs '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-key.bin' \
      --rhs '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}/attn-key.bin' \
      --dtype f32 \
      --shape \"\${k_shape}\" &&
    python3 '${COMPARE_SCRIPT_REMOTE}' \
      --lhs '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-value.bin' \
      --rhs '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}/attn-value.bin' \
      --dtype f32 \
      --shape \"\${v_shape}\" &&
    python3 '${COMPARE_SCRIPT_REMOTE}' \
      --lhs '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-mask.bin' \
      --rhs '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}/attn-mask.bin' \
      --dtype f32 \
      --shape \"\${mask_shape}\" &&
    python3 '${COMPARE_SCRIPT_REMOTE}' \
      --lhs '${RECK_MNN_DIR}/${MNN_DUMP_NAME}/attn-output.bin' \
      --rhs '${RECK_LLAMA_DIR}/${LLAMA_DUMP_NAME}/attn-output.bin' \
      --dtype f32 \
      --shape \"\${out_shape}\"
"
