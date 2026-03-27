# MNN QNN Export And Runtime Pipeline Report

## Scope

This note explains what `export.sh` is doing, what files it produces, and how the exported MNN artifacts interact with CPU and QNN/HTP at runtime.

Relevant entry points:

- `export.sh`
- `transformers/llm/export/llmexport.py`
- `transformers/llm/export/npu/generate_llm_qnn.py`
- `transformers/llm/engine/tools/generateLlmIO.cpp`
- `tools/cpp/compilefornpu.cpp`
- `source/backend/qnn/npu_convert.py`
- `transformers/llm/engine/src/llm.cpp`
- `source/backend/cpu/CPUPlugin.cpp`
- `source/backend/qnn/backend/QNNBackend.cpp`

## High-Level Result

`export.sh` does not convert the whole model into a single pure-NPU runtime.

What it actually produces is:

- one MNN model that still runs under the normal MNN runtime: `model/qnn/llm.mnn`
- one or more QNN offline cache binaries: `model/qnn/*.bin`
- a launcher config: `model/config_qnn.json`
- separate host-side embedding data such as `model/embeddings_bf16.bin`

At runtime:

- `llm_demo model/config_qnn.json` starts from the normal MNN LLM engine
- the outer execution is still scheduled by MNN
- some subgraphs inside `qnn/llm.mnn` have been replaced with `Plugin` ops of type `QNN`
- those plugin ops call into QNN and execute offline graphs on HTP/NPU
- tensors crossing plugin boundaries are still managed by the host side, which is why CPU/app-side interaction is still required

So this is a hybrid pipeline:

- CPU / MNN graph orchestration
- QNN / HTP execution for selected subgraphs
- host-managed tensor handoff between normal MNN tensors and QNN graph inputs/outputs

## Step 1: Export The Base MNN Model

This command in `export.sh`:

```bash
python llmexport.py --path /home/chensm22/Models/Qwen3-4B/ --export mnn \
 --quant_block 64 --quant_bit 4 \
 --generate_for_npu --seperate_embed \
 --act_bit=16 --sym --omni --hqq --calib_data prompt.txt
```

produces the normal MNN-side model assets under `transformers/llm/export/model/`.

Important effects:

- `--export mnn` exports an MNN-format LLM
- `--generate_for_npu` enables quant/export choices intended for later QNN conversion
- `--seperate_embed` keeps embedding data outside the main model, so embedding lookup can still happen on the host side

Typical artifacts from this phase include:

- `model/llm.mnn`
- `model/llm.mnn.weight`
- `model/llm_config.json`
- `model/embeddings_bf16.bin`

This phase alone does not yet produce QNN offline binaries.

## Step 2: Generate Representative I/O For Prefill And Decode

`generate_llm_qnn.py` first runs `generateLlmIO`:

```python
exe = os.path.join(os.getcwd(), args.mnn_path, "generateLlmIO")
```

and calls it with:

- `blockSize = 128`
- and an extra decode sample with `seqLen = 1`

The implementation is in [generateLlmIO.cpp](/home/chensm22/MNN/transformers/llm/engine/tools/generateLlmIO.cpp).

It saves:

- `tmp/testdir/128/input.mnn`
- `tmp/testdir/128/output.mnn`
- `tmp/testdir/1/input.mnn`
- `tmp/testdir/1/output.mnn`

Meaning:

- `128` is the prefill-like chunk sample
- `1` is the single-token decode sample

These samples are later used by `compilefornpu` to:

- discover split points
- compile multiple shape variants
- record exact input/output names and shapes

## Step 3: Split `llm.mnn` Into QNN-Candidate Subgraphs

`generate_llm_qnn.py` then writes `tmp/qnn.json` and runs:

```bash
compilefornpu model/llm.mnn qnn/llm.mnn qnn.json
```

The generated `qnn.json` looks conceptually like:

```json
{
  "type": "QNN",
  "testdir": ["testdir/1", "testdir/128"],
  "KVCACHE_SIZE_LIMIT": 0,
  "cache": "qnn"
}
```

Inside [compilefornpu.cpp](/home/chensm22/MNN/tools/cpp/compilefornpu.cpp):

- `type = "QNN"` makes it use `MNN_CONVERT_QNN`
- `testdir` tells it to compile using both decode and prefill example shapes
- `cache = "qnn"` makes all intermediate QNN-export products land under `tmp/qnn`

For each candidate submodule:

1. it runs the submodule with the QNN convert backend
2. QNN convert mode records the graph into a directory as:
   - `<graph>.cpp`
   - parameter `.raw` files
3. `compilefornpu` replaces the original MNN subgraph with a `Plugin` op of type `QNN`

That replacement happens in `_compileSubModule(...)` in [compilefornpu.cpp](/home/chensm22/MNN/tools/cpp/compilefornpu.cpp#L453).

The generated plugin op stores attrs such as:

- `path`
- `inputs`
- `outputs`
- `allGraphName`
- `allInputShape`
- shape-specific output descriptions like `o_0_0`, `o_1_0`
- optional KV state metadata

This is the key handoff point:

- the original pure-MNN subgraph is gone
- in its place is a CPU-visible `Plugin` node that knows how to call a QNN binary

## Step 4: Merge Shape Variants Into Final Offline QNN Binaries

`compilefornpu` writes `tmp/npu_postreat.json`, including a `merge` map.

Then `generate_llm_qnn.py` runs:

```bash
python3 source/backend/qnn/npu_convert.py npu_postreat.json <soc_id> <dsp_arch>
```

In [npu_convert.py](/home/chensm22/MNN/source/backend/qnn/npu_convert.py):

1. each exported graph directory is packed:
   - `.raw` files are tarred into `<graph>.bin`
2. `qnn-model-lib-generator` compiles `<graph>.cpp + <graph>.bin` into `lib<graph>.so`
3. `qnn-context-binary-generator` merges one or more graph libraries into a final context binary under `qnn/`

This is why one final `.bin` can contain multiple graph names.

For example:

- one plugin path like `qnn/graph0.bin`
- may internally contain both a decode graph and a prefill graph variant

The runtime later selects the concrete graph by `allGraphName`, not only by filename.

## Step 5: Final Runtime Config Is Still CPU-Driven

`generate_llm_qnn.py` writes:

```json
{
  "llm_model": "qnn/llm.mnn",
  "backend_type": "cpu",
  "thread_num": 1,
  "precision": "low",
  "chunk_limits": [128, 1],
  "memory": "low"
}
```

This is important:

- `backend_type` is deliberately `cpu`
- it does not mean QNN is unused
- it means the outer MNN execution stays on the CPU backend, and only selected plugin ops jump into QNN

The reason is architectural:

- the model is now a mixed graph
- normal MNN ops still run on CPU
- `Plugin(type="QNN")` ops are dispatched specially

## Step 6: What Happens When `llm_demo model/config_qnn.json` Runs

`llm_demo` just does:

1. `Llm::createLLM(config_path)`
2. `llm->load()`

The main load path is in [llm.cpp](/home/chensm22/MNN/transformers/llm/engine/src/llm.cpp).

Key points:

- `config_qnn.json` points `llm_model` to `qnn/llm.mnn`
- `llm.cpp` sets:
  - `rtg->setExternalFile(mConfig->llm_weight())`
  - `rtg->setExternalPath(mConfig->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR)`

The `npu_model_dir` defaults to the model base directory, so the runtime can resolve relative plugin paths such as:

- `qnn/graph0.bin`

That base directory is propagated into the backend in [StaticModule.cpp](/home/chensm22/MNN/express/module/StaticModule.cpp#L341).

## Step 7: Why CPU Backend Still Executes QNN Plugin Ops

`qnn/llm.mnn` contains `OpType_Plugin` nodes.

On CPU backend, [CPUPlugin.cpp](/home/chensm22/MNN/source/backend/cpu/CPUPlugin.cpp) creates a `CPUKernelContext` for every plugin op and dispatches by plugin type.

For `type = "QNN"`:

- `QNNBackend.cpp` registers:
  - a shape kernel: `PluginShapeRaw`
  - a compute kernel: `PluginExecuteRaw`

So the call chain is:

1. CPU backend sees `OpType_Plugin`
2. `CPUPlugin` looks at plugin type
3. for `QNN`, it instantiates `PluginExecuteRaw`
4. `PluginExecuteRaw` loads and executes the offline QNN graph

This is the central reason the runtime looks like “CPU backend”, but still executes on NPU inside selected nodes.

## Step 8: How Shape Selection Works

Because `compilefornpu` fused multiple shape variants into one plugin description, the plugin stores:

- `allInputShape`
- `allGraphName`

At runtime, `PluginShapeRaw` and `PluginExecuteRaw` use `computeIndex(...)` in [QNNBackend.cpp](/home/chensm22/MNN/source/backend/qnn/backend/QNNBackend.cpp#L421) to match the current input tensor shapes against the recorded shape list.

That decides:

- shape index `0` or `1`
- which output shape attrs `o_<shapeIndex>_<outputIndex>` to use
- which concrete graph name from `allGraphName` should be executed

So:

- prefill shape and decode shape can share one plugin op
- but runtime still picks different QNN graph entries internally

This matches your earlier observation that `graph0.bin` may contain more than one graph.

## Step 9: Where CPU And NPU Actually Exchange Data

There are 3 distinct host/NPU interaction layers.

### 1. MNN tensor to plugin-local tensor copy

In `PluginExecuteRaw::compute(...)`:

- CPU-side input tensors are copied into plugin-owned tensors first
- this is done by `ctx->backend()->onCopyBuffer(...)`

That is ordinary host-side tensor preparation.

### 2. Plugin-local tensor to QNN graph input/output binding

`PluginExecuteRaw` uses `RawExecutorWrapper` to:

- open the QNN binary
- retrieve the requested graph(s)
- bind named inputs and outputs by tensor name
- execute with `graphExecute(...)`

This is where the QNN binary actually runs on HTP/NPU.

### 3. QNN outputs copied back into normal MNN tensors

After the QNN graph finishes:

- outputs are copied back from plugin-owned tensors to the surrounding MNN tensors

So the host side remains responsible for:

- input staging
- output collection
- graph selection
- KV state maintenance
- invoking the next graph

The NPU is only responsible for the compiled subgraph execution itself.

## Step 10: KV Cache And CPU/NPU Cooperation

`compilefornpu` also records KV-related state metadata into the plugin op when attention state exists.

At runtime, `PluginExecuteRaw` allocates:

- host-managed state buffers
- update buffers
- a mask buffer

Then after every execution it updates those buffers on the host side.

That means KV cache handling is not “fully hidden inside the NPU binary”.

Instead:

- QNN subgraphs consume and produce state slices
- the host/plugin layer keeps the full rolling state and decides where the next slice lands

This is another important CPU/NPU interaction point.

## What `t4`, `t59`, `t90` Mean In This Context

For graphs like `graph0.cpp`, outputs such as:

- `t4`
- `t59`
- `t90`

are not useless debug leftovers.

In the original MNN-exported multi-graph pipeline:

- one graph emits them as `APP_READ`
- later graphs consume them as `APP_WRITE`

So they are part of the host-managed inter-graph contract.

That is consistent with this whole architecture:

- NPU executes one offline graph
- host side receives intermediate tensors
- host side feeds them into the next graph invocation

## The Most Important Practical Conclusion

When you run:

```bash
./llm_demo model/config_qnn.json
```

you are not running:

- “all CPU”

and you are also not running:

- “one giant pure-NPU graph with no host interaction”

You are running:

- an MNN-controlled hybrid pipeline
- CPU backend as the outer scheduler and plugin dispatcher
- QNN offline subgraphs on HTP/NPU for selected regions
- host-managed tensor handoff, shape dispatch, and KV/state updates between those regions

## Artifact Map

From the commands in `export.sh`, the important final artifacts are:

- `model/config_qnn.json`
  - startup config for `llm_demo`
- `model/qnn/llm.mnn`
  - mixed MNN model whose selected subgraphs have been replaced by `Plugin(type="QNN")`
- `model/qnn/*.bin`
  - QNN offline context binaries used by those plugin ops
- `model/embeddings_bf16.bin`
  - host-side embedding data
- `libQnnHtp.so`, `libQnnHtpVXXStub.so`, `libQnnHtpVXXSkel.so`, `libQnnSystem.so`
  - required QNN runtime libraries on device

## Short Answer

`export.sh` first exports a normal MNN LLM, then uses representative decode/prefill samples to carve out QNN-capable subgraphs, replaces those subgraphs inside `llm.mnn` with `Plugin(type="QNN")` ops, compiles the real QNN graphs into `.bin` offline caches, and finally runs everything through the normal MNN LLM engine. CPU is still responsible for scheduling, tensor marshaling, graph selection, and state updates; NPU is responsible for executing the compiled QNN subgraphs.
