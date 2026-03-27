# MNN QNN / `llama.cpp` Parity README

本文档记录 2026-03-27 这一轮迁移的当前状态，包括：

- 已经完成并验证过的任务
- 这轮实际使用过的关键命令
- 当前可用的 dump / 对比手段
- 下一步准备执行的命令

相关补充文档：

- [mnn-qnn-llama-migration-notes.md](/home/chensm22/MNN/transformers/llm/export/mnn-qnn-llama-migration-notes.md)
- [mnn-qnn-export-pipeline-report.md](/home/chensm22/MNN/transformers/llm/export/mnn-qnn-export-pipeline-report.md)
- [compare_tensor_dump.py](/home/chensm22/MNN/transformers/llm/export/compare_tensor_dump.py)
- [compare_qnn_decode_replay.py](/home/chensm22/MNN/transformers/llm/export/compare_qnn_decode_replay.py)
- [run_llama_qnn_decode_replay.sh](/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh)
- [run_mnn_qnn_decode_dump.sh](/home/chensm22/MNN/transformers/llm/export/run_mnn_qnn_decode_dump.sh)

## 当前结论

- `graph0 .. graph37` 是 decode 图，`seq = 1`。
- `graph1_0 .. graph1_37` 是 prefill 图，当前导出里 `seq = 128`。
- `llama.cpp` 当前 decode 已经确认使用的是 `graph0`，不是 `graph1_0`。
- `llama.cpp` 里 `graph0` 的输入 embedding，已经和 MNN 对同一个 raw prompt `hello` 的 embedding 完全一致。
- `llama.cpp` 里 `graph0` 的输出 `t4/t59/t90/t92/t122/t127`，已经和 MNN 完全一致。
- `llama.cpp` 里 `graph1` 的输入输出链路已经打通，`graph1` 的关键张量也已经和 MNN 对齐。
- `llama.cpp` 当前已经能稳定 replay `graph0 -> attention -> graph1`。

## 已完成任务

- 确认了导出图的范式不是“单独 QKV 图”，而是 MNN host 管理的递推状态机。
- 确认了 `graph0` 的真实输入输出契约，并整理到了迁移笔记里。
- 在 `llama.cpp` 侧接入了 `graph0` replay 和边界 dump。
- 在 `llama.cpp` 侧修掉了 `graph1` 无法进入 QNN 的 blocker。
- 在 `llama.cpp` 侧加入了 sibling binary 自动加载，因此 `graph1.bin` 可以和 `graph0.bin` 一起被复用。
- 在 MNN 侧加入了 `graph0` dump。
- 在 MNN 侧把 dump 能力扩展到了 `graph1`。
- 用同一个 raw prompt `hello` 实测比对了 MNN 和 `llama.cpp` 的输入 embedding。
- 用同一个 raw prompt `hello` 实测比对了 MNN 和 `llama.cpp` 的 `graph0` 输出。
- 用同一个 raw prompt `hello` 实测比对了 MNN 和 `llama.cpp` 的 `graph1` 关键输入输出。

## 当前已经验证通过的点

- `graph0` 输入 embedding：`max_abs_diff = 0.0`
- `graph0` 输出 `t4`：`max_abs_diff = 0.0`
- `graph0` 输出 `t59`：`max_abs_diff = 0.0`
- `graph0` 输出 `t90`：`max_abs_diff = 0.0`
- `graph0` 输出 `t92`：`max_abs_diff = 0.0`
- `graph0` 输出 `t122`：`max_abs_diff = 0.0`
- `graph0` 输出 `t127`：`max_abs_diff = 0.0`
- `graph1` 输入 `t129`：`max_abs_diff = 0.0`
- `graph1` 输出 `t139`：`max_abs_diff = 0.0`
- `graph1` 输出 `t187`：`max_abs_diff = 0.0`
- `graph1` 输出 `t216`：`max_abs_diff = 0.0`
- `graph1` 输出 `t221`：`max_abs_diff = 0.0`

说明：

- MNN 自身在 host roundtrip 时，`t4` 和 `t59` 有极小中间态误差。
- 目前观察到的是：
- `t4 graph0->graph1 handoff max_abs_diff = 1.5040859580039978e-06`
- `t59 graph0->graph1 handoff max_abs_diff = 0.0001519918441772461`
- `t90 graph0->graph1 handoff max_abs_diff = 0.0`
- 尽管如此，`graph1` 的最终关键输出仍然和 `llama.cpp` 完全一致。

## 关键代码位置

- `llama.cpp` QNN 主逻辑：[ggml-qnn.cpp](/home/chensm22/MNN/llama.cpp/ggml/src/ggml-qnn/ggml-qnn.cpp)
- `llama.cpp` QNN 工具定义：[qnn-utils.h](/home/chensm22/MNN/llama.cpp/ggml/src/ggml-qnn/qnn-utils.h)
- MNN QNN 执行入口：[QNNBackend.cpp](/home/chensm22/MNN/source/backend/qnn/backend/QNNBackend.cpp)
- MNN demo 入口：[llm_demo.cpp](/home/chensm22/MNN/transformers/llm/engine/demo/llm_demo.cpp)
- 导出脚本参考：[export.sh](/home/chensm22/MNN/export.sh)
- `llama.cpp` Android 构建脚本：[build-android.sh](/home/chensm22/MNN/llama.cpp/build-android.sh)

## 当前 dump 产物

- `llama.cpp` replay dump：
  - [/tmp/qnn-replay-hello-20260327d](/tmp/qnn-replay-hello-20260327d)
- MNN `graph0 + graph1` dump：
  - [/tmp/mnn_qnn_graph01_dump_hello](/tmp/mnn_qnn_graph01_dump_hello)
- MNN prompt / embedding dump：
  - [/home/chensm22/MNN/tmp/mnn-hello-dump](/home/chensm22/MNN/tmp/mnn-hello-dump)

## 推荐脚本

- `llama.cpp` 侧 replay / push / pull：
  - [run_llama_qnn_decode_replay.sh](/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh)
- MNN 侧 dump / push / pull：
  - [run_mnn_qnn_decode_dump.sh](/home/chensm22/MNN/transformers/llm/export/run_mnn_qnn_decode_dump.sh)
- 自动比较 MNN 与 `llama.cpp` dump：
  - [compare_qnn_decode_replay.py](/home/chensm22/MNN/transformers/llm/export/compare_qnn_decode_replay.py)

这三个脚本已经能覆盖：

- 本地构建
- 增量 push 关键文件
- 设备侧运行
- dump 回传
- 按 `graphN-input/output__tensor_name` 自动逐文件比较

## 建议先定义 retry helper

`oneplus13` 和 `reck` 的连接偶尔不稳定，建议先在本机 shell 里定义：

```bash
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
      echo "command failed after ${n} attempts: $*"
      return 1
    fi
    sleep 2
  done
}
```

## 这轮实际使用过的关键命令

### 1. 本机构建

```bash
cmake --build /home/chensm22/MNN/llama.cpp/build-android --target ggml-qnn llama-cli -j 4
```

```bash
cmake --build /home/chensm22/MNN/project/android/build_64 --target MNN llm_demo -j 4
```

### 2. 只增量推送关键文件

`llama.cpp` 侧只推新的可执行文件和新增的 QNN bin：

```bash
retry rsync -avhP \
  /home/chensm22/MNN/llama.cpp/build-android/bin/llama-cli \
  oneplus13:/data/data/com.termux/files/home/llama.cpp-test/bin/llama-cli
```

```bash
retry rsync -avhP \
  /home/chensm22/MNN/transformers/llm/export/model/qnn/graph1.bin \
  oneplus13:/data/data/com.termux/files/home/llama.cpp-test/ops-bin/graph1.bin
```

MNN 侧只推新的 `libMNN.so` 和 prompt：

```bash
retry rsync -avhP \
  /home/chensm22/MNN/project/android/build_64/libMNN.so \
  reck:/home/reck/mnn_qwen3/libMNN.so
```

```bash
retry rsync -avhP \
  /home/chensm22/MNN/transformers/llm/export/prompt-hello.txt \
  reck:/home/reck/mnn_qwen3/prompt-hello.txt
```

```bash
retry ssh reck \
  "adb push /home/reck/mnn_qwen3/libMNN.so /data/local/tmp/MNN/libMNN.so"
```

```bash
retry ssh reck \
  "adb push /home/reck/mnn_qwen3/prompt-hello.txt /data/local/tmp/MNN/prompt-hello.txt"
```

### 3. 运行 `llama.cpp` 侧 replay + dump

```bash
retry ssh oneplus13 "
  cd /data/data/com.termux/files/home/llama.cpp-test && \
  env \
    LD_LIBRARY_PATH=/data/data/com.termux/files/home/llama.cpp-test/build-android/lib:/system/lib64:/vendor/lib64 \
    ADSP_LIBRARY_PATH=/data/data/com.termux/files/home/llama.cpp-test/build-android/lib \
    GGML_QNN_BIN_PATH=/data/data/com.termux/files/home/llama.cpp-test/ops-bin/graph0.bin \
    GGML_QNN_REPLAY_GRAPH01=1 \
    GGML_QNN_REPLAY_DUMP_DIR=/data/data/com.termux/files/home/llama.cpp-test/qnn-replay-hello-20260327d \
    GGML_QNN_DECODE_LOG_TOKENS=0 \
    ./bin/llama-cli \
      -m models/Qwen3-4B-f16.gguf \
      -p hello \
      -n 2 \
      --device QNN \
      --no-warmup \
      2>&1 | tee /data/data/com.termux/files/home/llama.cpp-test/llama-graph1-20260327d.log
"
```

### 4. 拉回 `llama.cpp` dump

```bash
retry rsync -avhP \
  oneplus13:/data/data/com.termux/files/home/llama.cpp-test/qnn-replay-hello-20260327d/ \
  /tmp/qnn-replay-hello-20260327d/
```

### 5. 运行 MNN 侧 dump

```bash
retry ssh reck "
  adb shell '
    rm -rf /data/local/tmp/MNN/qnn-dump-hello-graph01 && \
    cd /data/local/tmp/MNN && \
    env \
      LD_LIBRARY_PATH=/data/local/tmp/MNN \
      MNN_QNN_DUMP_DIR=/data/local/tmp/MNN/qnn-dump-hello-graph01 \
      ./llm_demo model/config_qnn_raw.json prompt-hello.txt 2 && \
    find /data/local/tmp/MNN/qnn-dump-hello-graph01 -maxdepth 2 -type f | sort
  '
"
```

### 6. 拉回 MNN dump

```bash
retry ssh reck "
  rm -rf /home/reck/mnn_qwen3/qnn-dump-hello-graph01 && \
  adb pull /data/local/tmp/MNN/qnn-dump-hello-graph01 /home/reck/mnn_qwen3/
"
```

```bash
retry rsync -avhP \
  reck:/home/reck/mnn_qwen3/qnn-dump-hello-graph01/ \
  /tmp/mnn_qnn_graph01_dump_hello/
```

### 7. 对比关键 dump

对比输入 embedding：

```bash
python3 /home/chensm22/MNN/transformers/llm/export/compare_tensor_dump.py \
  --lhs /home/chensm22/MNN/tmp/mnn-hello-dump/prompt-0000.embedding.bin \
  --rhs /tmp/qnn-replay-hello-20260327d/decode-token-0000/graph0-input__inp_embd.bin \
  --dtype f32 \
  --shape 1,1,2560
```

对比 `graph0` 的 `Q` 输出：

```bash
python3 /home/chensm22/MNN/transformers/llm/export/compare_tensor_dump.py \
  --lhs /tmp/mnn_qnn_graph01_dump_hello/graph0-run-0007/graph0-output__t92.bin \
  --rhs /tmp/qnn-replay-hello-20260327d/decode-token-0000/graph0-output__t92.bin \
  --dtype f32 \
  --shape 1,1,32,128
```

对比 `graph1` 的 `V` 输出：

```bash
python3 /home/chensm22/MNN/transformers/llm/export/compare_tensor_dump.py \
  --lhs /tmp/mnn_qnn_graph01_dump_hello/graph1-run-0007/graph1-output__t221.bin \
  --rhs /tmp/qnn-replay-hello-20260327d/decode-token-0000/graph1-output__t221.bin \
  --dtype f32 \
  --shape 1,1,8,128
```

如果已经有整目录 dump，推荐直接用自动对比脚本：

```bash
python3 /home/chensm22/MNN/transformers/llm/export/compare_qnn_decode_replay.py \
  --mnn-root /tmp/mnn_qnn_graph01_dump_hello \
  --llama-dir /tmp/qnn-replay-hello-20260327d/decode-token-0000 \
  --graphs graph0 graph1 \
  --run-id 7 \
  --atol 1e-3
```

说明：

- 如果 MNN dump 目录里累积了多轮运行，自动选择的 `run_id` 可能不是你想要比的那一轮。
- 已知 `graph0/graph1` 对齐时，`run_id=7` 可以得到全通过结果。

## 当前最重要的 blocker

- `graph0 -> attention -> graph1` 这一段已经对齐。
- 还没有把 `graph2 .. graph37` 全部自动接上。
- 现在的 `llama.cpp` replay 逻辑还偏向验证框架，不是完整 decode 状态机。
- MNN 侧 dump 目前已经覆盖 `graph0` 和 `graph1`，但还没有扩到整条 decode 链。

## 下一步准备执行的命令

以下命令属于“接上后续层递推”这一阶段，当前文档先记录计划，等代码支持范围扩展后按这套流程继续跑。

### 1. 继续增量推送 decode 离线图

```bash
for i in $(seq 2 37); do
  retry rsync -avhP \
    /home/chensm22/MNN/transformers/llm/export/model/qnn/graph${i}.bin \
    oneplus13:/data/data/com.termux/files/home/llama.cpp-test/ops-bin/graph${i}.bin
done
```

### 2. 每次代码改动后重新构建并只推关键文件

```bash
cmake --build /home/chensm22/MNN/llama.cpp/build-android --target ggml-qnn llama-cli -j 4
```

```bash
cmake --build /home/chensm22/MNN/project/android/build_64 --target MNN llm_demo -j 4
```

```bash
retry rsync -avhP \
  /home/chensm22/MNN/llama.cpp/build-android/bin/llama-cli \
  oneplus13:/data/data/com.termux/files/home/llama.cpp-test/bin/llama-cli
```

```bash
retry rsync -avhP \
  /home/chensm22/MNN/project/android/build_64/libMNN.so \
  reck:/home/reck/mnn_qwen3/libMNN.so
```

```bash
retry ssh reck \
  "adb push /home/reck/mnn_qwen3/libMNN.so /data/local/tmp/MNN/libMNN.so"
```

### 3. 复用同一个 `hello` prompt 再跑一次全链路 dump

- `llama.cpp` 继续使用 raw prompt `hello`
- MNN 继续使用 [prompt-hello.txt](/home/chensm22/MNN/transformers/llm/export/prompt-hello.txt)
- 这样可以避免 prompt 模板差异干扰定位

### 4. 扩大对比范围

计划继续检查这些边界张量：

- `graph2`：`t222 / t232 / t280 / t309 / t314`
- `graph3`：下一层对应的 attention 输入和 `Q/K/V`
- `graph35`：最后一个规则递推层
- `graph36`：最后一层 block 收尾输出
- `graph37`：最终 logits

### 5. 最终目标

- 在 `llama.cpp` 里把 decode `graph0 .. graph37` 整条状态机接齐
- 用同一个 prompt 在 MNN 和 `llama.cpp` 上逐层 dump
- 逐层比较中间态和最终 logits
- 最终把“整模型与 MNN 完全一致”作为验收标准
