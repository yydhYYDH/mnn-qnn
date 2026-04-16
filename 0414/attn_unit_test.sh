#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-reck}"
LOCAL_LLAMA_DIR="${LOCAL_LLAMA_DIR:-/home/chensm22/llama.cpp}"
LOCAL_COMPARE_PY="${LOCAL_COMPARE_PY:-/home/chensm22/MNN/transformers/llm/export/compare_llama_prefill_attn_layer0.py}"
LOCAL_MNN_REPO_ROOT="${LOCAL_MNN_REPO_ROOT:-/home/chensm22/MNN}"
LOCAL_MNN_BUILD_DIR="${LOCAL_MNN_BUILD_DIR:-/home/chensm22/MNN/project/android/build_64}"
LOCAL_MNN_LIB="${LOCAL_MNN_LIB:-${LOCAL_MNN_BUILD_DIR}/libMNN.so}"
LOCAL_MNN_EXPRESS_LIB="${LOCAL_MNN_EXPRESS_LIB:-${LOCAL_MNN_BUILD_DIR}/libMNN_Express.so}"
LOCAL_MNN_PLUGIN_LIB="${LOCAL_MNN_PLUGIN_LIB:-${LOCAL_MNN_BUILD_DIR}/libplugin_matmul.so}"

REMOTE_DUMP_ROOT="${REMOTE_DUMP_ROOT:-/home/reck/llama.cpp-test/llama-prefill-compare}"
REMOTE_TOKEN_DIR="${REMOTE_TOKEN_DIR:-${REMOTE_DUMP_ROOT}/decode-token-0000}"
REMOTE_STAGE_DIR="${REMOTE_STAGE_DIR:-/home/reck/llama.cpp-test/attn-unit-layer0}"
REMOTE_TEST_BIN="${REMOTE_TEST_BIN:-${REMOTE_STAGE_DIR}/test-attention-unit}"
REMOTE_COMPARE_PY="${REMOTE_COMPARE_PY:-${REMOTE_STAGE_DIR}/compare_llama_prefill_attn_layer0.py}"
REMOTE_MNN_LIB="${REMOTE_MNN_LIB:-${REMOTE_STAGE_DIR}/libMNN.so}"
REMOTE_MNN_EXPRESS_LIB="${REMOTE_MNN_EXPRESS_LIB:-${REMOTE_STAGE_DIR}/libMNN_Express.so}"
REMOTE_MNN_PLUGIN_LIB="${REMOTE_MNN_PLUGIN_LIB:-${REMOTE_STAGE_DIR}/libplugin_matmul.so}"
REMOTE_INPUT_DIR="${REMOTE_INPUT_DIR:-${REMOTE_STAGE_DIR}/input}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_STAGE_DIR}/cpu-attn-layer0-run-unit}"
REMOTE_COMPARE_DIR="${REMOTE_COMPARE_DIR:-${REMOTE_STAGE_DIR}/compare-token-0000}"

DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
DEVICE_INPUT_DIR="${DEVICE_INPUT_DIR:-${DEVICE_ROOT}/input}"
DEVICE_OUTPUT_DIR="${DEVICE_OUTPUT_DIR:-${DEVICE_ROOT}/cpu-attn-layer0-run-unit}"
DEVICE_BIN="${DEVICE_BIN:-${DEVICE_ROOT}/test-attention-unit}"

LOCAL_ANDROID_BIN="${LOCAL_ANDROID_BIN:-${LOCAL_LLAMA_DIR}/build-android/bin/test-attention-unit}"

cat <<EOF
[1/7] Configure/build Android attention unit test
EOF
(
  cd "${LOCAL_LLAMA_DIR}"
  MNN_REPO_ROOT="${LOCAL_MNN_REPO_ROOT}" ./build-android.sh cmake
  cmake --build build-android --target test-attention-unit -j4
)

cat <<EOF
[2/7] Push Android binary, MNN libs, and compare script to ${HOST}
EOF
ssh "${HOST}" "mkdir -p '${REMOTE_STAGE_DIR}' '${REMOTE_INPUT_DIR}' '${REMOTE_OUTPUT_DIR}' '${REMOTE_COMPARE_DIR}'"
scp "${LOCAL_ANDROID_BIN}" "${HOST}:${REMOTE_TEST_BIN}"
scp "${LOCAL_COMPARE_PY}" "${HOST}:${REMOTE_COMPARE_PY}"
scp "${LOCAL_MNN_LIB}" "${HOST}:${REMOTE_MNN_LIB}"
scp "${LOCAL_MNN_EXPRESS_LIB}" "${HOST}:${REMOTE_MNN_EXPRESS_LIB}"
if [ -f "${LOCAL_MNN_PLUGIN_LIB}" ]; then
  scp "${LOCAL_MNN_PLUGIN_LIB}" "${HOST}:${REMOTE_MNN_PLUGIN_LIB}"
fi

cat <<EOF
[3/7] Prepare layer0 q/k/v input staging on ${HOST}
EOF
ssh "${HOST}" "
rm -rf '${REMOTE_INPUT_DIR}' '${REMOTE_OUTPUT_DIR}' &&
mkdir -p '${REMOTE_INPUT_DIR}' '${REMOTE_OUTPUT_DIR}' &&
cp '${REMOTE_TOKEN_DIR}/graph1_0-output__t92.bin' '${REMOTE_INPUT_DIR}/attn-query.bin' &&
cp '${REMOTE_TOKEN_DIR}/graph1_0-output__t92.meta.txt' '${REMOTE_INPUT_DIR}/attn-query.meta.txt' &&
cp '${REMOTE_TOKEN_DIR}/graph1_0-output__t122.bin' '${REMOTE_INPUT_DIR}/attn-key.bin' &&
cp '${REMOTE_TOKEN_DIR}/graph1_0-output__t122.meta.txt' '${REMOTE_INPUT_DIR}/attn-key.meta.txt' &&
cp '${REMOTE_TOKEN_DIR}/graph1_0-output__t127.bin' '${REMOTE_INPUT_DIR}/attn-value.bin' &&
cp '${REMOTE_TOKEN_DIR}/graph1_0-output__t127.meta.txt' '${REMOTE_INPUT_DIR}/attn-value.meta.txt'
"

cat <<EOF
[4/7] Push binary and inputs from ${HOST} to phone via adb
EOF
ssh "${HOST}" "
adb shell 'rm -rf \"${DEVICE_ROOT}\" && mkdir -p \"${DEVICE_ROOT}\"' &&
adb push '${REMOTE_TEST_BIN}' '${DEVICE_BIN}' &&
adb push '${REMOTE_MNN_LIB}' '${DEVICE_ROOT}/libMNN.so' &&
adb push '${REMOTE_MNN_EXPRESS_LIB}' '${DEVICE_ROOT}/libMNN_Express.so' &&
if [ -f '${REMOTE_MNN_PLUGIN_LIB}' ]; then adb push '${REMOTE_MNN_PLUGIN_LIB}' '${DEVICE_ROOT}/libplugin_matmul.so'; fi &&
adb push '${REMOTE_INPUT_DIR}' '${DEVICE_ROOT}/'
"

cat <<EOF
[5/7] Run attention unit on phone
EOF
ssh "${HOST}" "adb shell '
set -e
cd \"${DEVICE_ROOT}\"
chmod +x \"${DEVICE_BIN}\"
export LD_LIBRARY_PATH=.
./test-attention-unit \
  --graph-kind boundary \
  --kernel ggml \
  --input-dir \"${DEVICE_INPUT_DIR}\" \
  --dump-dir \"${DEVICE_OUTPUT_DIR}\"
'"

cat <<EOF
[6/7] Pull phone outputs back to ${HOST} into an isolated compare directory
EOF
ssh "${HOST}" "
rm -rf '${REMOTE_OUTPUT_DIR}' '${REMOTE_COMPARE_DIR}' &&
mkdir -p '${REMOTE_COMPARE_DIR}' &&
adb pull '${DEVICE_OUTPUT_DIR}' '${REMOTE_STAGE_DIR}/' &&
'/home/reck/Utils/anaconda3/bin/python' - <<'PY'
from pathlib import Path
import shutil

unit_dir = Path('${REMOTE_OUTPUT_DIR}')
compare_dir = Path('${REMOTE_COMPARE_DIR}')

pairs = [
    ('Qcur-0 (view) (permuted)', 'attn-layer0-boundary-src0-src0-contig__Qcur-0__unit'),
    ('kv_cache_k-0 (permuted)', 'attn-layer0-boundary-src0-src1-contig__kv_cache_k-0__unit'),
    ('kv_cache_v-0 (permuted)', 'attn-layer0-boundary-src0-src2-contig__kv_cache_v-0__unit'),
    ('kqv_out-0', 'attn-layer0-boundary-contig__kqv_out-0__unit'),
]

for src_stem, dst_stem in pairs:
    for suffix in ('.bin', '.meta.txt'):
        src = unit_dir / f'{src_stem}{suffix}'
        dst = compare_dir / f'{dst_stem}{suffix}'
        if not src.exists():
            raise FileNotFoundError(f'missing unit output: {src}')
        shutil.copyfile(src, dst)
        print(f'copied {src} -> {dst}')

for stem in (
    'graph1_0-output__t92',
    'graph1_0-output__t122',
    'graph1_0-output__t127',
    'graph1_1-input__t129',
):
    for suffix in ('.bin', '.meta.txt'):
        src = Path('${REMOTE_TOKEN_DIR}') / f'{stem}{suffix}'
        dst = compare_dir / f'{stem}{suffix}'
        shutil.copyfile(src, dst)
        print(f'copied {src} -> {dst}')
PY
"

cat <<EOF
[7/7] Compare layer0 q/k/v inputs and final kqv_out against MNN graph dumps
EOF
ssh "${HOST}" "'/home/reck/Utils/anaconda3/bin/python' '${REMOTE_COMPARE_PY}' \
  --dump-root '${REMOTE_STAGE_DIR}' \
  --token-dir 'compare-token-0000' \
  --layer 0"

cat <<EOF
Done.
Compared only the layer0 chain:
  graph1_0-output__t92   -> Q
  graph1_0-output__t122  -> K
  graph1_0-output__t127  -> V
  kqv_out                -> graph1_1-input__t129
Remote dump root:
  ${REMOTE_DUMP_ROOT}
Remote stage dir:
  ${REMOTE_STAGE_DIR}
Isolated compare dir:
  ${REMOTE_COMPARE_DIR}
Device root:
  ${DEVICE_ROOT}
EOF
