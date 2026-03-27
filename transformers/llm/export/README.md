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

## 当前问题与进展速览

### 当前进展

- `llama.cpp` 侧 decode replay 和 MNN 侧 decode dump 都已经覆盖 `graph0 .. graph37`，这轮工作不再卡在“后续层没接上”。
- 结构性问题已经收敛：recurrent graph IO 映射改成按 tensor 名绑定，replay graph buffer 也改成按需创建，正常推理路径上的 sibling bin 预加载 DSP 报错已经移除。
- 对齐结果已经收敛到很小的范围：`graph0` 和 `graph1` 全 pass，第一处分叉稳定出现在 `graph2-input__t222`，其次才是 `graph2-input__t139` 的轻微 host roundtrip 误差。

### 当前问题

- 当前真正的 blocker 已经不是 `graph2` 的 IO 名字绑定，而是 `llama.cpp` live attention 在 layer-1 实际读到的 past `V` 与 replay 链不一致。
- replay 视角下，MNN 的 `graph1-output__t221` 与 `llama.cpp` replay 的 `graph1-output__t221` 完全一致；但 `llama.cpp` live attention 看到的 `kv_cache_v-1` slot0 仍与它存在明显差异。
- `llama.cpp` replay 的 `graph2-input__t222` 又和 live attention 看到的 `kv_cache_v-1` slot0 展开结果完全一致，因此问题已经缩小到 host attention / KV handoff，而不是 replay dump 后处理。

### 当前下一步

- 复跑已经加入 dense `current_k/current_v` dump、`Qcur/Kcur/Vcur` ancestor dump、以及 KV write -> attention view 显式依赖修正的这版 `llama.cpp`。
- 优先验证 `Vcur-1 / kv_cache_v-1 / graph2-input__t222` 这一条链是否已经对齐；如果这里收敛，`graph2` 之后的层间偏差预计会一起消失。

## 2026-03-27 最新进展

- `llama.cpp` 侧 decode replay 现在已经能从 `graph0` 一直跑到 `graph37`，不再停在 `graph1`。
- `llama.cpp` 侧 recurrent graph 的 IO 映射已经改成按导出 tensor 名绑定，不再依赖固定 index；`graph2` 对应 `t222/t139/t59/t90 -> t232/t280/t309/t314`，`graph3` 对应 `t315/t232/t59/t90 -> t325/t373/t402/t407`。
- `llama.cpp` 侧 replay graph buffer 已改成按需创建，普通推理默认不再预加载所有 sibling decode bin，因此此前那组 `Failed to register memHandles / qnnMemCreateHandle failed / err 8003 / err 5005 / err 1002` 的启动期 DSP 报错，已经从正常推理路径移除。
- `llama.cpp` 侧 prefill 现在也已经能在 NPU 上实际跑完整条 `graph1_0 .. graph1_37` 链，`1 < token < 128` 的 chunk 会按 MNN 同样的语义 pad 到 `128` 再执行。
- `run_llama_qnn_decode_replay.sh` 现在新增了 `CLEAN_OUTPUT=1` 模式：模型文本走 `stdout`，中间 QNN / sched-trace 日志单独落到 `<dump>.log`，不再把中文 token 打断。
- MNN 侧已经成功收集到 `token0` 的 `graph0 .. graph37` 全量 dump，可用脚本 [run_mnn_qnn_decode_dump.sh](/home/chensm22/MNN/transformers/llm/export/run_mnn_qnn_decode_dump.sh) 直接复跑。
- `token0` 的全量对比结论是：
- `graph0` 全 pass。
- `graph1` 全 pass。
- 第一处分叉仍然是 `graph2-input__t222`，其次是 `graph2-input__t139` 的很小 host roundtrip 误差。
- `graph2` 之后的偏差是从这两个输入开始逐层传播，不是 `graph2` 的 IO 名字绑定再次出错。
- 对比结果默认是直接打印到终端；若要落盘，可执行：

```bash
python3 /home/chensm22/MNN/transformers/llm/export/compare_qnn_decode_replay.py \
  --mnn-root /tmp/qnn-dump-hello-20260327-rerun \
  --llama-dir /tmp/qnn-replay-hello-20260327-rerun/decode-token-0000 \
  --run-id 7 \
  --atol 1e-3 | tee /tmp/qnn-compare-hello-20260327-rerun-token0.log
```

### 当前最重要的新发现

- replay 视角下，MNN 的 `graph1-output__t221` 和 `llama.cpp` replay 的 `graph1-output__t221` 已经逐值完全一致。
- 但是 `llama.cpp` live attention 实际读到的 `layer-1` `kv_cache_v` 第一个 slot，与 replay `graph1-output__t221` 之间仍然存在明显差异，这个差异显著大于普通 `f32 -> f16 -> f32` roundtrip 噪声。
- `llama.cpp` replay 的 `graph2-input__t222` 与 `llama.cpp` live attention 看到的 `kv_cache_v-1` slot0 展开结果完全一致，说明 `graph2-input__t222` 的偏差不是 replay dump 自己后处理错了，而是 host attention 真正吃到的 past V 就和 MNN 不同。
- MNN 侧 `graph2-input__t222` 与 `expand(graph1-output__t221)` 只差正常的 `f16` 级别 roundtrip 误差，这进一步说明 MNN 的 `graph1 -> graph2` 递推契约本身是自洽的。

### 当前工作假设

- 现在最可疑的不是 `graph2` 的名字映射，也不是 `flash_attn` 开关，而是 `llama.cpp` live ggml 图里 layer-1 host attention 所看到的 past KV 依赖仍然和 replay 链不同。
- 为了验证这一点，当前本地树已经额外加入了：
- MNN 侧 dense `current_k/current_v` dump。
- `llama.cpp` 侧 `Qcur/Kcur/Vcur` ancestor dump。
- `llama.cpp` 侧 KV write -> attention view 的显式依赖改动。
- 下一步应优先复跑这版 `llama.cpp`，确认 live attention 是否已经能看到正确的 `Vcur-1 / kv_cache_v-1`，再继续比较 `graph2-input__t222`。

## Prefill NPU 跑通记录

### 现象

- 一开始把 prefill family 直接按 sibling bin 全量加载到 DSP 时，会反复出现下面这组资源错误：
- `QnnDsp <E> Skel failed to process context binary.`
- `QnnDsp <E> Context create from binary failed ... err 5005`
- `QnnDsp <E> Failed to map shared weights ... err: 1002`
- 资源错误绕开以后，prefill 仍然会在 `graph1_0` 入口或 `graph1_1` 边界前失败，因为当前导出的 prefill 图族只有 `seq = 128`，而 `llama.cpp` 实际 prompt chunk 可能是 `32`、`24` 这类 `1 < token < 128` 的长度。

### 根因

- 第一层问题不是图本身坏了，而是加载策略不对。MNN 不是“启动时把所有 sibling prefill bin 全塞进 DSP”，而是主 bin 常驻、auxiliary bin 按需 lazy load；我们之前的 eager load 会一次性创建太多 DSP context，重新触发 `5005 / 1002`。
- 第二层问题是 shape 契约不匹配。`graph1_0 .. graph1_37` 的输入输出都固定在 `seq = 128`，但 `llama.cpp` prefill 之前一直把真实长度的 hidden / pos / attention 边界张量直接送进 QNN，所以 `32-token` prompt 会和 `128-token` graph buffer 撞 size mismatch。

### 修法

- 保留 `graph0.bin` 作为主 context，prefill auxiliary bin 改成和 decode 一样按需 lazy load；正常 prefill 不再开启 `GGML_QNN_LOAD_EXTRA_MNN_BINS=1` 这类全量预加载。
- `qnn_compute_forward_mul_mat_qkv()` 里对 prefill `graph1_0` 入口做 pad-to-128：
- 输入 `hidden/pos` 先零填充到 QNN graph buffer 大小，再把真实 token 的连续数据拷进去。
- `graph1_0` 输出的 `Q/K/V` 只切回真实 token 对应的前缀字节给 host attention，保证 host 侧 layer-0 attention 仍按真实 prompt 长度工作。
- `qnn_replay_execute_decode_graph()` 里对 prefill `graph1_1 .. graph1_36` 的 attention boundary 也做同样的 pad-to-128，再把 full-128 hidden/rope 状态继续传给后续 graph。
- 这样就和 MNN 的控制流对齐了：真实 token 数只体现在 host attention 和最终取样，QNN prefill graph family 始终跑 `seq = 128`。

### 验证

- 本轮成功日志在 [/tmp/qnn-prefill-pad-20260327b/llama.log](/tmp/qnn-prefill-pad-20260327b/llama.log)。
- 资源错误已经消失；在该日志里搜索 `5005`、`1002`、`Skel failed`、`Failed to map shared weights` 都没有命中。
- 日志确认 `graph1_1 .. graph1_37` 都已经按需加载并执行，例如：
- `Lazy-loaded auxiliary QNN binary .../graph1.bin for graph graph1_1`
- `[prefill] padded graph1_1 attention input from 524288 to 2097152 bytes`
- `[prefill] graph1_37 finished for token 0`
- 同一轮 prompt eval 已成功完成：
- `prompt eval time = 4681.87 ms / 32 tokens`
- 说明当前 `32-token` prefill 已经真实经过 NPU 上的 `graph1_0 .. graph1_37` 链，不再停在启动期 DSP 资源错误或 graph size assert。

## Clean 输出模式验证

### 修法

- [run_llama_qnn_decode_replay.sh](/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh) 现在支持 `CLEAN_OUTPUT=1`。
- 开启后，设备侧 `llama-cli` 的 `stdout` 通过 `tee` 单独写到 `<dump>.out`，`stderr` 单独写到 `<dump>.log`。
- 这样模型生成的正文不会再和 `[QNN]`、`[sched-trace]` 这类中间日志混在同一条流里，中文输出也不会再被日志插断成乱码。

### 短 prompt 样例

- 命令：

```bash
CLEAN_OUTPUT=1 HOST=oneplus13 \
PROMPT='用一句中文介绍深圳的春天。' \
N_PREDICT=24 \
DUMP_NAME=qnn-clean-output-20260327 \
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh run-prefill
```

- 纯净输出在 [/tmp/qnn-clean-output-20260327/llama.out](/tmp/qnn-clean-output-20260327/llama.out)。
- 对应日志在 [/tmp/qnn-clean-output-20260327/llama.log](/tmp/qnn-clean-output-20260327/llama.log)。
- 这轮日志确认 NPU 正常初始化、prefill 跑到 `graph1_37`、decode 跑到 token 23：
- `HTP device created successfully`
- `[prefill] graph1_37 finished for token 0`
- `[replay] captured graph0 outputs for token 23`

### 长 prompt 样例

- 命令：

```bash
CLEAN_OUTPUT=1 HOST=oneplus13 \
PROMPT='请用中文写一篇面向第一次来深圳旅行的短文，分三段介绍深圳春天的天气、值得去的地方，以及一天的散步路线建议，语言自然，有画面感，不要使用列表，字数控制在400字左右。请直接开始正文。' \
N_PREDICT=160 \
DUMP_NAME=qnn-long-answer-20260327 \
/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh run-prefill
```

- 纯净输出在 [/tmp/qnn-long-answer-20260327/llama.out](/tmp/qnn-long-answer-20260327/llama.out)。
- 对应日志在 [/tmp/qnn-long-answer-20260327/llama.log](/tmp/qnn-long-answer-20260327/llama.log)。
- 这轮长回答验证了长 prompt + 长 decode 也能稳定走 clean 模式；日志里已经能看到：
- `HTP device created successfully`
- `[prefill] graph1_37 finished for token 0`
- `[replay] captured graph0 outputs for token 159`
- `prompt eval time = 9501.42 ms / 60 tokens`
- `eval time = 35965.37 ms / 159 runs`
- 当前这条长样例在 `N_PREDICT=160` 处仍然被截断，说明 clean 模式已经解决“中间日志污染输出”的问题，但若要拿到完整长文，仍应继续提高 `N_PREDICT` 上限。

## 已完成任务

- 确认了导出图的范式不是“单独 QKV 图”，而是 MNN host 管理的递推状态机。
- 确认了 `graph0` 的真实输入输出契约，并整理到了迁移笔记里。
- 在 `llama.cpp` 侧接入了 decode replay 和边界 dump，并把 replay 链路从 `graph0` 扩到了 `graph37`。
- 在 `llama.cpp` 侧修掉了 `graph1` 无法进入 QNN 的 blocker，并把 recurrent graph 的 IO 绑定改成按 tensor 名匹配。
- 在 `llama.cpp` 侧把 replay graph buffer 改成按需创建，普通推理路径不再预加载所有 sibling decode bin。
- 在 MNN 侧把 dump 能力从 `graph0` 扩到了 `graph37`，已经能稳定收集 `token0` 全链路 dump。
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

- `llama.cpp` 最新全链 replay dump：
  - [/tmp/qnn-replay-hello-20260327-rerun](/tmp/qnn-replay-hello-20260327-rerun)
- `llama.cpp` clean stdout 短 prompt 样例：
  - [/tmp/qnn-clean-output-20260327/llama.out](/tmp/qnn-clean-output-20260327/llama.out)
  - [/tmp/qnn-clean-output-20260327/llama.log](/tmp/qnn-clean-output-20260327/llama.log)
- `llama.cpp` clean stdout 长 prompt 样例：
  - [/tmp/qnn-long-answer-20260327/llama.out](/tmp/qnn-long-answer-20260327/llama.out)
  - [/tmp/qnn-long-answer-20260327/llama.log](/tmp/qnn-long-answer-20260327/llama.log)
- MNN 最新全链 decode dump：
  - [/tmp/qnn-dump-hello-20260327-rerun](/tmp/qnn-dump-hello-20260327-rerun)
- 早期 `graph0 + graph1` 快速对比 dump：
  - [/tmp/mnn_qnn_graph01_dump_hello](/tmp/mnn_qnn_graph01_dump_hello)
- MNN prompt / embedding dump：
  - [/home/chensm22/MNN/tmp/mnn-hello-dump](/home/chensm22/MNN/tmp/mnn-hello-dump)

## 推荐脚本

- `llama.cpp` 侧 replay / push / pull：
  - [run_llama_qnn_decode_replay.sh](/home/chensm22/MNN/transformers/llm/export/run_llama_qnn_decode_replay.sh)
- 若只想看模型正文，不想让 QNN 中间日志打断输出，可在同一个脚本上额外加 `CLEAN_OUTPUT=1`。
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

## 当前最重要的问题

- `graph0 .. graph37` 的 replay / dump 链路都已经打通，当前不是“后续层没接上”的问题。
- 第一处分叉稳定落在 `graph2-input__t222`；`graph2-input__t139` 仍有一处较小的 host roundtrip 误差，但不是主矛盾。
- 关键异常是：`graph1-output__t221` 在 replay 里已经和 MNN 完全一致，但 `llama.cpp` live attention 看到的 layer-1 `kv_cache_v` slot0 仍明显不同。
- 因为 `graph2-input__t222` 与 live attention 看到的 `kv_cache_v-1` slot0 展开结果完全一致，所以当前最可疑的是 host attention / KV handoff，而不是 `graph2` 的名字映射或 replay dump 后处理。

## 下一步准备执行的命令

以下命令属于“验证 live attention past KV 依赖”这一阶段；当前代码已经能覆盖 `graph0 .. graph37`，下一轮重点不是再把图接起来，而是确认 `Vcur-1 / kv_cache_v-1 / graph2-input__t222` 这条链是否已经收敛。

### 1. 如需同步最新 bin，重新增量推送 decode 离线图

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

### 4. 优先比较这些边界张量

- `graph1-output__t221` 与 live `kv_cache_v-1` slot0
- `graph2`：`t222 / t139 / t232 / t280 / t309 / t314`
- `graph3`：下一层对应的 attention 输入和 `Q/K/V`
- `graph35`：最后一个规则递推层
- `graph36`：最后一层 block 收尾输出
- `graph37`：最终 logits

### 5. 最终目标

- 在 `llama.cpp` 里把 decode `graph0 .. graph37` 整条状态机接齐
- 用同一个 prompt 在 MNN 和 `llama.cpp` 上逐层 dump
- 逐层比较中间态和最终 logits
- 最终把“整模型与 MNN 完全一致”作为验收标准
