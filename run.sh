PROMPT_FILE="/home/chensm22/MNN/transformers/llm/export/propmt_256_0.txt" \
 HOST=reck DEVICE_ROOT=/data/local/tmp/MNN LOCAL_PROMPT_FILE="${PROMPT_FILE}" \
 N_PREDICT=1 PROFILE_STAGE=1 DUMP_NAME=mnn-prefill-compare ./transformers/llm/export/run_mnn_qnn_decode_dump.sh all
