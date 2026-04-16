GGML_QNN_DUMP_ATTENTION=1 GGML_QNN_DEBUG_DECODE=1 HOST=reck \
 DEVICE_ROOT=/data/local/tmp/MNN PROMPT_FILE=/home/chensm22/MNN/transformers/llm/export/propmt_256_0.txt DUMP_NAME=qnn-prefill-attn-debug transformers/llm/export/run_llama_prefill_attn_debug_compare.sh run

ssh reck '/home/reck/Utils/anaconda3/bin/python \
/tmp/compare_graph_kv_to_attn_cache.py \
  --llama-dir /home/reck/llama.cpp-test/qnn-prefill-attn-debug/decode-token-0000 \
  --layer 0 \
  --slots 0,1,2,3,4,5,6,7 \
  --limit 8
'