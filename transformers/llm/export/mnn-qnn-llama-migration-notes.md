# MNN QNN To `llama.cpp` Migration Notes

## Goal

Reuse MNN-exported QNN offline graphs inside `llama.cpp` while preserving MNN-side numerics as closely as possible.

If the target is **full-model parity with MNN**, then the migration boundary must match MNN's host-managed graph recurrence, not `llama.cpp`'s current per-op boundary.

## Key Finding

The MNN-exported QNN graphs are **not** cut at arbitrary `Q/K/V` boundaries.

They form a host-managed recurrence:

- one graph produces `Q/K/V` for the current layer
- host / CPU runs attention
- the next graph consumes the attention result plus preserved hidden / RoPE side inputs
- that next graph finishes the previous layer and prepares `Q/K/V` for the following layer

If `llama.cpp` only reuses `graph0` and then falls back to its own native layer logic, it is **not** following the same numerical boundary as MNN.

That is the main reason why "MNN side is correct, `llama.cpp` side is not" can happen even when `graph0` itself executes successfully.

There are also **two parallel graph families**:

- `graph0 .. graph37`
  - decode path
  - `seq = 1`
- `graph1_0 .. graph1_37`
  - prefill / chunk path
  - `seq = 128` in the current export

So if the goal is "fully match MNN", `llama.cpp` eventually needs **both**:

- a decode state machine for `graph0 .. graph37`
- a prefill state machine for `graph1_0 .. graph1_37`

## `graph0` Contract

Inputs:

- `t2`: current hidden state, shape `{1, 1, 2560}`
- `t0`: current position id, shape `{1, 1}`

Outputs:

- `t4`: shape `{1, 1, 2560}`
  - produced by `_blocks_0_Reshape_output_0_Reshape`
  - this is the host-visible hidden branch used later for residual add
- `t59`: shape `{1, 1, 1, 128}`
  - produced by `_blocks_0_self_attn_Gather_2_output_0_Reshape`
  - one RoPE coefficient branch
- `t90`: shape `{1, 1, 1, 128}`
  - produced by `_blocks_0_self_attn_Gather_3_output_0_Reshape`
  - the paired RoPE coefficient branch
- `t92`: shape `{1, 1, 32, 128}`
  - layer-0 `Q`, after q_norm + RoPE
- `t122`: shape `{1, 1, 8, 128}`
  - layer-0 `K`, after k_norm + RoPE
- `t127`: shape `{1, 1, 8, 128}`
  - layer-0 `V`

So `graph0` is not just "QKV projection".

It is:

- layer-0 pre-attention hidden preparation
- layer-0 q/k RoPE coefficient extraction
- layer-0 Q/K/V projection

## `graph1` Contract

Inputs:

- `t129`: shape `{1, 1, 4096}`
  - layer-0 attention output before `o_proj`
- `t4`: shape `{1, 1, 2560}`
  - residual branch from `graph0`
- `t59`: shape `{1, 1, 1, 128}`
  - RoPE coefficient branch, reused
- `t90`: shape `{1, 1, 1, 128}`
  - RoPE coefficient branch, reused

Outputs:

- `t139`: shape `{1, 1, 2560}`
  - hidden state after finishing layer 0
  - also the input hidden state for layer 1
- `t187`: shape `{1, 1, 32, 128}`
  - layer-1 `Q`
- `t216`: shape `{1, 1, 8, 128}`
  - layer-1 `K`
- `t221`: shape `{1, 1, 8, 128}`
  - layer-1 `V`

So `graph1` does:

- layer-0 `o_proj`
- residual add using `t4`
- layer-0 FFN and residual
- layer-1 attention norm and Q/K/V preparation
- layer-1 q/k RoPE application using reused `t59/t90`

## `graph2` Confirms The Pattern

Inputs:

- `t222`: shape `{1, 1, 4096}`
  - layer-1 attention output before `o_proj`
- `t139`: shape `{1, 1, 2560}`
  - layer-1 hidden / residual branch
- `t59`, `t90`
  - same RoPE coefficients reused again

Outputs:

- next hidden state
- next-layer `Q/K/V`

So for `graphN (N >= 1)` the pattern is stable:

- input: attention output of previous layer + hidden branch + shared RoPE coeffs
- output: next hidden state + next layer `Q/K/V`

## The Tail Is Different From The Middle

The middle recurrence is regular, but the tail is not.

### Decode Tail: `graph35`, `graph36`, `graph37`

For the `seq=1` family:

- `graph35`
  - inputs:
    - `t3291`: previous attention output, shape `{1, 1, 4096}`
    - `t3208`: current hidden branch, shape `{1, 1, 2560}`
    - `t59`, `t90`: shared RoPE tensors, shape `{1, 1, 1, 128}`
  - outputs:
    - `t3301`: next hidden branch, shape `{1, 1, 2560}`
    - `t3349`: layer-35 `Q`, shape `{1, 1, 32, 128}`
    - `t3378`: layer-35 `K`, shape `{1, 1, 8, 128}`
    - `t3383`: layer-35 `V`, shape `{1, 1, 8, 128}`

- `graph36`
  - inputs:
    - `t3384`: layer-35 attention output before `o_proj`, shape `{1, 1, 4096}`
    - `t3301`: hidden branch from `graph35`, shape `{1, 1, 2560}`
  - outputs:
    - `t3393`: final hidden before output head, shape `{1, 1, 2560}`
  - note:
    - no `t59/t90`
    - no `Q/K/V`
    - this graph only finishes the final transformer block

- `graph37`
  - inputs:
    - `t3394`: final hidden, shape `{1, 1, 2560}`
  - outputs:
    - `t3396`: logits, shape `{1, 1, 151936}`
  - note:
    - this graph performs final RMSNorm + `lm_head`

So the decode recurrence is:

1. `graph0` boots layer 0
2. `graph1 .. graph35` follow the regular recurrence
3. `graph36` finishes the last transformer block
4. `graph37` produces logits

### Prefill Tail: `graph1_35`, `graph1_36`, `graph1_37`

The `seq=128` family has the same structure, just with chunk-sized tensors:

- `graph1_35`
  - outputs:
    - `t3301`: `{1, 128, 2560}`
    - `t3349`: `{1, 128, 32, 128}`
    - `t3378`: `{1, 128, 8, 128}`
    - `t3383`: `{1, 128, 8, 128}`
- `graph1_36`
  - output:
    - `t3393`: `{1, 128, 2560}`
- `graph1_37`
  - output:
    - `t3396`: `{1, 128, 151936}`

So prefill is not a separate algorithm. It is the same state machine, but with different tensor shapes and graph names.

## Exact Decode State Machine

If the goal is exact decode parity with MNN, the `llama.cpp` loop should conceptually become:

1. `graph0(resid_0, pos)`
   - outputs:
     - `branch_0 = t4`
     - `rope_a = t59`
     - `rope_b = t90`
     - `q_0 = t92`
     - `k_0 = t122`
     - `v_0 = t127`
2. host attention on `q_0/k_0/v_0`
   - output:
     - `attn_0 = t129`
3. for layer `i = 1 .. 35`:
   - `graphi(attn_{i-1}, branch_{i-1}, rope_a, rope_b)`
   - outputs:
     - `branch_i`
     - `q_i`
     - `k_i`
     - `v_i`
   - host attention on `q_i/k_i/v_i`
   - output:
     - `attn_i`
4. `graph36(attn_35, branch_35)`
   - output:
     - `final_hidden`
5. `graph37(final_hidden)`
   - output:
     - `logits`

This is the decode flow that matches the exported MNN graphs.

## Exact Prefill State Machine

For prefill / chunk execution, `llama.cpp` needs the same control flow with the `graph1_*` family:

1. `graph1_0(chunk_hidden, chunk_pos)` -> `branch_0`, `rope_a`, `rope_b`, `q_0`, `k_0`, `v_0`
2. host chunk attention
3. `graph1_1 .. graph1_35`
4. `graph1_36(attn_35, branch_35)` -> `final_hidden`
5. `graph1_37(final_hidden)` -> `logits`

The important point is that decode and prefill should share one conceptual runner in `llama.cpp`:

- same recurrence
- different graph family
- different tensor shapes

## What This Means For `llama.cpp`

To match MNN numerics, `llama.cpp` should not treat `graph0` as an isolated acceleration for `Q/K/V`.

Instead it should model the same host-side recurrence:

1. Run `graph0(hidden, pos)`:
   - get `hidden_branch = t4`
   - get `rope_a = t59`
   - get `rope_b = t90`
   - get `Q/K/V` for layer 0
2. Run attention on host:
   - produce `attn_out_0` matching MNN `t129`
3. Run `graph1(attn_out_0, hidden_branch, rope_a, rope_b)`:
   - get `hidden_1`
   - get `Q/K/V` for layer 1
4. Run attention on host:
   - produce `attn_out_1` matching MNN `t222`
5. Run `graph2(attn_out_1, hidden_1, rope_a, rope_b)`
6. Repeat

This is the real MNN execution pattern.

For exact parity, this also implies a design choice:

- do **not** keep the current "`ggml` op fusion + partial graph replacement" path as the main execution path
- add a dedicated MNN-style layer runner that executes at transformer-block boundaries

That runner should own:

- graph selection:
  - `graph0 .. graph37` for decode
  - `graph1_0 .. graph1_37` for prefill
- host attention between graph calls
- persistent host-visible state:
  - current hidden branch
  - current attention output
  - shared RoPE side tensors

If `llama.cpp` stays on the current op-level integration, it will remain very hard to prove exact parity with MNN because the execution boundary is different by construction.

## Why The Current `llama.cpp` Integration Is Numerically Mismatched

The current `llama.cpp` QNN bridge only uses `graph0` to replace layer-0 `Q/K/V`, and then:

- copies back only `Q/K/V`
- ignores `t4/t59/t90` semantically
- falls back to native `llama.cpp` math for the rest of the layer stack

That means the runtime boundary is different from MNN:

- MNN continues in quantized/offline graph space using `t4` and shared RoPE coeffs
- `llama.cpp` continues in its own graph using original hidden tensors and its own RoPE / FFN / residual path

Even if both paths are individually "reasonable", they are not numerically identical.

## The Most Likely Precision Risks

### 1. Wrong graph boundary

This is the biggest issue.

Using only `graph0` is not equivalent to the MNN pipeline.

### 2. Wrong semantic source for residual branch

If later graphs are used, they must consume the exported hidden branch tensor (`t4`, then `t139`, ...), not a recomputed or original `llama.cpp` hidden tensor.

### 3. Losing MNN's RoPE side tensors

If later graphs are used, `t59/t90` must be preserved and reused across all later graphs for the same token.

### 4. Attention output boundary mismatch

`graph1` expects `t129`, which is the post-attention, pre-`o_proj` tensor.

If `llama.cpp` feeds a tensor from a different point in its attention pipeline, the next graph will diverge immediately.

### 5. Layout mismatch at graph boundary

For `Q/K/V`, direct memcpy is likely correct because the flat buffer ordering matches the logical `[head_dim, head]` packing expected by ggml.

But for full graph-to-graph migration, every boundary tensor must be checked explicitly, especially:

- attention output tensors (`t129`, `t222`, ...)
- hidden-state tensors (`t4`, `t139`, ...)

## Recommended Migration Strategy

### Option A: Exact MNN-style recurrence

Best for precision consistency.

Implement in `llama.cpp`:

- decode graph runner:
  - `graph0`
  - `graph1 .. graph35`
  - `graph36`
  - `graph37`
- prefill graph runner:
  - `graph1_0`
  - `graph1_1 .. graph1_35`
  - `graph1_36`
  - `graph1_37`
- host-side attention kernel between graph calls
- explicit boundary state object:
  - current hidden branch tensor
  - current attention output tensor
  - shared RoPE tensors `t59/t90`
  - current `Q/K/V`

This is the closest to MNN numerically.

### Option B: Only reuse `graph0`

Best for implementation simplicity, but not for exact parity.

If this path is kept, then precision should be validated only at the `Q/K/V` boundary:

- compare MNN `graph0` outputs vs `llama.cpp` injected outputs for one token
- if they match closely, any remaining full-model divergence comes from different graph boundaries, not from `graph0` execution itself

## Practical Validation Order

To debug `llama.cpp` precision problems, compare in this order:

1. `graph0` input tensors:
   - hidden input
   - position input
2. `graph0` output tensors:
   - `t92`, `t122`, `t127`
   - and optionally `t4`, `t59`, `t90`
3. attention output boundary:
   - `t129` in MNN vs corresponding `llama.cpp` tensor
4. `graph1` outputs:
   - next hidden
   - next-layer `Q/K/V`

If step 2 already differs, the issue is in graph execution or tensor layout.

If step 2 matches but step 3 differs, the issue is in host-side attention / KV / masking / softmax.

If step 2 and step 3 match but full results still differ, the issue is the graph boundary or later recurrence handling.

## Recommended Verification Ladder

If the target is "exactly the same as MNN", verify in four layers, not just one.

### Level 1: Offline Graph Contract Check

Before touching full decode, verify that `llama.cpp` can run the raw exported graph with MNN-identical inputs and produce MNN-identical outputs.

Recommended checks:

- `graph0`
  - inputs: `t2`, `t0`
  - outputs: `t4`, `t59`, `t90`, `t92`, `t122`, `t127`
- `graph1`
  - inputs: `t129`, `t4`, `t59`, `t90`
  - outputs: `t139`, `t187`, `t216`, `t221`
- `graph36`
  - inputs: `t3384`, `t3301`
  - output: `t3393`
- `graph37`
  - input: `t3394`
  - output: `t3396`

If any of these do not match, stop there. Do not debug end-to-end generation yet.

### Level 2: Boundary Tensor Replay

Add one replay mode to `llama.cpp`:

- load boundary tensors dumped from MNN
- execute one exact graph step
- dump the outputs

This makes it possible to isolate:

- graph execution mismatch
- layout mismatch
- wrong tensor selected at the boundary

### Level 3: Per-Layer Decode Replay

Once individual graph calls match, verify the whole decode recurrence for one token:

1. MNN dump:
   - `graph0` inputs / outputs
   - `attn_0`
   - `graph1` outputs
   - `attn_1`
   - ...
   - `graph36` output
   - `graph37` logits
2. `llama.cpp` dump the same checkpoints
3. compare layer by layer

At this stage, the first mismatching checkpoint tells you exactly which subsystem is wrong:

- `graph0` output mismatch:
  - wrong graph call or tensor layout
- attention output mismatch:
  - host attention / KV / mask / softmax mismatch
- `graphN` output mismatch after attention matches:
  - wrong graph selection or wrong hidden branch / RoPE reuse

### Level 4: Final Logits And Token Choice

Only after Levels 1 to 3 pass should you compare:

- final logits
- argmax / sampled token
- generated text

Text-level mismatch is too late as a primary debugging signal.

## What To Dump From MNN

The minimum useful golden dump for decode is:

- `graph0`
  - `t2`, `t0`
  - `t4`, `t59`, `t90`, `t92`, `t122`, `t127`
- each middle layer `i = 0 .. 34`
  - attention output before next graph
  - next graph outputs:
    - hidden branch
    - `Q/K/V`
- tail
  - `graph36` output `t3393`
  - `graph37` output `t3396`

For prefill, use the same rule with the `graph1_*` family.

## What To Dump From `llama.cpp`

To make the comparison fair, `llama.cpp` should dump the same host-visible tensors, with the same names where possible:

- `graph0` inputs / outputs
- attention outputs
- per-layer graph inputs / outputs
- final hidden before `graph37`
- final logits

Each dump should include:

- tensor name
- dtype
- shape
- raw binary file

That is enough to compare numerically offline.

## Practical Implementation Advice For `llama.cpp`

To keep the migration debuggable, split the implementation into these milestones:

1. add a dedicated "MNN graph replay" path
   - can run `graph0`, `graph1`, `graph36`, `graph37` directly from dumped inputs
2. add decode `graph0 -> attention -> graph1` for one token
3. generalize middle recurrence for `graph1 .. graph35`
4. add `graph36`
5. add `graph37`
6. add prefill family `graph1_0 .. graph1_37`
7. switch runtime dispatch from ad-hoc fused ops to the explicit MNN-style state machine

That order gives you a clean checkpoint after every major step.

## Bottom Line

If exact MNN parity is the goal, the migration target is not:

- "reuse some MNN graphs inside `ggml`"

It is:

- "reproduce MNN's graph-level state machine inside `llama.cpp`"

The middle layers are regular:

- previous attention output
- current hidden branch
- shared RoPE tensors
- next hidden + next `Q/K/V`

The tail is special:

- `graph36` finishes the last block
- `graph37` produces logits

That is the structure `llama.cpp` needs to mirror if you want MNN and `llama.cpp` to be numerically identical.
