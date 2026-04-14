ssh reck 'rm -rf /home/reck/mnn_qwen3/mnn-prefill-compare /home/reck/llama.cpp-test/llama-prefill-compare'

PROMPT_FILE="/home/chensm22/MNN/transformers/llm/export/propmt_256_0.txt" \
 HOST=reck DEVICE_ROOT=/data/local/tmp/MNN LOCAL_PROMPT_FILE="${PROMPT_FILE}" \
 N_PREDICT=1 PROFILE_STAGE=1 DUMP_NAME=mnn-prefill-compare ./transformers/llm/export/run_mnn_qnn_decode_dump.sh all

./transformers/llm/export/run_compare.sh compare --chunk-index 0
./transformers/llm/export/run_compare.sh compare --chunk-index 1