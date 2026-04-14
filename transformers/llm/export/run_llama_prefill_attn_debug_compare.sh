#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

HOST="${HOST:-reck}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/MNN}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/propmt_256_0.txt}"
DUMP_NAME="${DUMP_NAME:-qnn-prefill-attn-debug}"
REMOTE_STAGE_ROOT="${REMOTE_STAGE_ROOT:-/home/reck/llama.cpp-test}"
REMOTE_DUMP_ROOT="${REMOTE_DUMP_ROOT:-${REMOTE_STAGE_ROOT}/${DUMP_NAME}}"
RUN_REMOTE_STAGE="${RUN_REMOTE_STAGE:-1}"
RUN_REMOTE_PULL="${RUN_REMOTE_PULL:-0}"
TOKEN_GLOB="${TOKEN_GLOB:-decode-token-*}"
CPU_ATTN_DUMP_LAYER="${CPU_ATTN_DUMP_LAYER:-0}"

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [run|compare|all]

Commands:
  run      Run llama QNN prefill attention dump, then stage results on ${HOST}
  compare  Compare the key attention boundary files on ${HOST}
  all      Run + compare

Environment overrides:
  HOST                default: ${HOST}
  DEVICE_ROOT         default: ${DEVICE_ROOT}
  PROMPT_FILE         default: ${PROMPT_FILE}
  DUMP_NAME           default: ${DUMP_NAME}
  REMOTE_STAGE_ROOT   default: ${REMOTE_STAGE_ROOT}
  REMOTE_DUMP_ROOT    default: ${REMOTE_DUMP_ROOT}
  RUN_REMOTE_STAGE    default: ${RUN_REMOTE_STAGE}; set 0 to skip stage-remote after run
  RUN_REMOTE_PULL     default: ${RUN_REMOTE_PULL}; set 1 to also pull dump locally
  TOKEN_GLOB          default: ${TOKEN_GLOB}
  CPU_ATTN_DUMP_LAYER default: ${CPU_ATTN_DUMP_LAYER}; set empty to dump every CPU attention layer

This script answers three questions for each decode-token directory:
  1. Is graph1_1-prepared-input__t129.bin identical to graph1_1-input__t129.bin?
  2. Is attn-layer0-boundary-contig__kqv_out-0.bin identical to graph1_1-prepared-input__t129.bin?
  3. Is attn-layer0-boundary-src0-contig__*.bin identical to graph1_1-prepared-input__t129.bin?

Interpretation:
  - graph1_1-prepared-input__t129.bin is the actual QNN input_attn buffer after prepare_inputs
  - graph1_1-input__t129.bin is the graph-input dump taken from the same execution buffer set
  - attn-layer0-boundary-src0-contig__*.bin is the contiguous copy source before prepare_inputs
  - attn-layer0-boundary-contig__kqv_out-0.bin is the contiguous dump of the boundary node itself
EOF
}

run_dump() {
    env \
        GGML_QNN_DUMP_ATTENTION=1 \
        GGML_QNN_RUN_MNN_DECODE=1 \
        GGML_QNN_LOG_LEVEL=5 \
        GGML_QNN_CPU_ATTN_DUMP_LAYER="${CPU_ATTN_DUMP_LAYER}" \
        HOST="${HOST}" \
        DEVICE_ROOT="${DEVICE_ROOT}" \
        PROMPT_FILE="${PROMPT_FILE}" \
        N_PREDICT=1 \
        PROFILE_STAGE=1 \
        DUMP_NAME="${DUMP_NAME}" \
        "${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" run-debug

    if [ "${RUN_REMOTE_STAGE}" = "1" ]; then
        env \
            HOST="${HOST}" \
            DEVICE_ROOT="${DEVICE_ROOT}" \
            DUMP_NAME="${DUMP_NAME}" \
            "${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" stage-remote
    fi

    if [ "${RUN_REMOTE_PULL}" = "1" ]; then
        env \
            HOST="${HOST}" \
            DEVICE_ROOT="${DEVICE_ROOT}" \
            DUMP_NAME="${DUMP_NAME}" \
            "${SCRIPT_DIR}/run_llama_qnn_decode_replay.sh" pull
    fi
}

compare_remote() {
    ssh "${HOST}" "
set -euo pipefail

dump_root='${REMOTE_DUMP_ROOT}'
token_glob='${TOKEN_GLOB}'

if [ ! -d \"\${dump_root}\" ]; then
    echo \"[compare] missing remote dump root: \${dump_root}\" >&2
    exit 1
fi

found_any=0
for d in \$(find \"\${dump_root}\" -mindepth 1 -maxdepth 1 -type d -name \"\${token_glob}\" | sort); do
    found_any=1

    prepared=\"\${d}/graph1_1-prepared-input__t129.bin\"
    graph_input=\"\${d}/graph1_1-input__t129.bin\"
    boundary_contig=\"\${d}/attn-layer0-boundary-contig__kqv_out-0.bin\"

    echo \"== \$(basename \"\${d}\") ==\"

    if [ ! -f \"\${prepared}\" ] || [ ! -f \"\${graph_input}\" ]; then
        echo \"prepared_vs_input: MISSING\"
    elif cmp -s \"\${prepared}\" \"\${graph_input}\"; then
        echo \"prepared_vs_input: IDENTICAL\"
    else
        echo \"prepared_vs_input: DIFFERENT\"
        wc -c \"\${prepared}\" \"\${graph_input}\"
        sha256sum \"\${prepared}\" \"\${graph_input}\"
    fi

    if [ ! -f \"\${boundary_contig}\" ] || [ ! -f \"\${prepared}\" ]; then
        echo \"boundary_contig_vs_prepared: MISSING\"
    elif cmp -s \"\${boundary_contig}\" \"\${prepared}\"; then
        echo \"boundary_contig_vs_prepared: IDENTICAL\"
    else
        echo \"boundary_contig_vs_prepared: DIFFERENT\"
        wc -c \"\${boundary_contig}\" \"\${prepared}\"
        sha256sum \"\${boundary_contig}\" \"\${prepared}\"
    fi

    src0_found=0
    src0_identical=0
    for src0 in \"\${d}\"/attn-layer0-boundary-src0-contig__*.bin; do
        if [ ! -f \"\${src0}\" ]; then
            continue
        fi
        src0_found=1
        if cmp -s \"\${src0}\" \"\${prepared}\"; then
            src0_identical=1
            echo \"src0_contig_vs_prepared: IDENTICAL \$(basename \"\${src0}\")\"
        else
            echo \"src0_contig_vs_prepared: DIFFERENT \$(basename \"\${src0}\")\"
            wc -c \"\${src0}\" \"\${prepared}\"
            sha256sum \"\${src0}\" \"\${prepared}\"
        fi
    done

    if [ \"\${src0_found}\" = \"0\" ]; then
        echo \"src0_contig_vs_prepared: MISSING\"
    fi

    if [ -f \"\${prepared}\" ] && [ -f \"\${graph_input}\" ] && cmp -s \"\${prepared}\" \"\${graph_input}\"; then
        if [ \"\${src0_identical}\" = \"1\" ]; then
            echo \"conclusion: input_attn matches prepared-input, graph-input, and src0-contig\"
            echo \"meaning: QNN input_attn comes from boundary src0 after contiguous copy, not from boundary-contig\"
        else
            echo \"conclusion: input_attn matches prepared-input and graph-input, but not this token's src0-contig dump\"
            echo \"meaning: inspect padding, truncation, or shape differences for src0-contig in this token\"
        fi
    else
        echo \"conclusion: prepared-input and graph-input diverged; investigate prepare_inputs path first\"
    fi

    echo
done

if [ \"\${found_any}\" = \"0\" ]; then
    echo \"[compare] no token directories matched \${dump_root}/\${token_glob}\" >&2
    exit 1
fi
"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    run)
        run_dump
        ;;
    compare)
        compare_remote
        ;;
    all)
        run_dump
        compare_remote
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo "unknown command: ${COMMAND}" >&2
        print_usage >&2
        exit 1
        ;;
esac
