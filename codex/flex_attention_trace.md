# FlexAttention Implementation Trace (reference, Triton, CuTeDSL/FLASH)

## Scope
This trace follows the exact benchmark call sites in:
- [flex_flash_attention.py](../thirdparty/attention-gym/examples/flex_flash_attention.py#L66)

and traces all requested paths:
1. Reference path (`flex_attention` without explicit backend compile wrapper)
2. Triton backend (`kernel_options={"BACKEND": "TRITON"}`)
3. CuTeDSL FLASH backend (`kernel_options={"BACKEND": "FLASH"}`)

For each backend, both forward and backward are traced down to kernel/template generation and execution.

## Source Map
| Area | Link |
|---|---|
| Benchmark harness / call sites | [flex_flash_attention.py](../thirdparty/attention-gym/examples/flex_flash_attention.py#L66) |
| User API + BlockMask plumbing | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1468) |
| Higher-order op + autograd + trace capture | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L94) |
| Inductor flex lowering (fwd/bwd) | [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107) |
| FLASH/CuTeDSL lowering path | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247) |
| Triton template inlining hooks | [torch/_inductor/select_algorithm.py](../thirdparty/pytorch/torch/_inductor/select_algorithm.py#L881) |
| Triton forward template | [templates/flex_attention.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja#L1) |
| Triton backward template | [templates/flex_backwards.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja#L1) |
| Shared fwd inner loops + mod/mask application | [templates/common.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L3) |
| FLASH forward template | [templates/flash_attention.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention.py.jinja#L1) |
| FLASH backward template | [templates/flash_attention_backward.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja#L1) |
| CuTeDSL template codegen hooks | [cutedsl_kernel.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py#L404) |
| CuTeDSL scheduling/compile handoff | [cutedsl_scheduling.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py#L59) |
| Async compile for CuteDSL | [async_compile.py](../thirdparty/pytorch/torch/_inductor/async_compile.py#L571) |
| flash-attn CuTe interface | [flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94) |
| flash-attn forward kernel object | [flash_attn/cute/flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L52) |
| flash-attn backward kernel object (SM100) | [flash_attn/cute/flash_bwd_sm100.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm100.py#L1835) |
| Shared score_mod/score_mod_bwd inner routines | [flash_attn/cute/softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343) |
| Shared mask_mod application | [flash_attn/cute/mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L127) |
| Callable hashing (compile cache keys) | [flash_attn/cute/utils.py](../thirdparty/flash-attention/flash_attn/cute/utils.py#L42) |

## Frame 0: Exact Benchmark Entry Points
Reference snippet with inline frame comments:

```python
# file: thirdparty/attention-gym/examples/flex_flash_attention.py
# link: ../thirdparty/attention-gym/examples/flex_flash_attention.py#L86

# [R-FWD entry] Reference path: direct flex_attention() call
ref = flex_attention(q_ref, k_ref, v_ref, score_mod=score_mod, block_mask=block_mask)

# [R-BWD entry] Reference backward
grad_out = torch.randn_like(ref)
ref.backward(grad_out)

# [F-FWD/F-BWD entry] FLASH path: torch.compile(partial(...BACKEND="FLASH"))
flash_out = flex_flash(q_flash, k_flash, v_flash, score_mod=score_mod, block_mask=block_mask)
flash_out.backward(grad_out.to(flash_out.dtype))

# [T-FWD/T-BWD entry] TRITON path: torch.compile(partial(...BACKEND="TRITON"))
triton_out = flex_triton(q_triton, k_triton, v_triton, score_mod=score_mod, block_mask=block_mask)
triton_out.backward(grad_out.to(triton_out.dtype))
```

Compiler wrapper setup in the same file:

```python
# file: thirdparty/attention-gym/examples/flex_flash_attention.py
# link: ../thirdparty/attention-gym/examples/flex_flash_attention.py#L36

def compile_flex(backend: Literal["FLASH", "TRITON"], *, dynamic: bool) -> Callable:
    return torch.compile(
        partial(flex_attention, kernel_options={"BACKEND": backend}),
        dynamic=dynamic,
    )
```

## High-Level Execution Diagram
```text
compare_backends()
  -> flex_attention(...) [reference]
     -> HOP dispatch -> math_attention (CompositeExplicitAutograd)
     -> autograd backward -> sdpa_dense_backward (math)

  -> torch.compile(partial(flex_attention, BACKEND="TRITON"))(...)
     -> Dynamo/FX trace of HOP + score/mask subgraphs
     -> Inductor lower higher_order.flex_attention to Triton template
     -> Triton kernel launch
     -> backward: lower higher_order.flex_attention_backward to Triton bwd template

  -> torch.compile(partial(flex_attention, BACKEND="FLASH"))(...)
     -> Dynamo/FX trace of HOP + score/mask subgraphs
     -> Inductor lower to CuteDSL template
     -> async_compile.cutedsl(...) -> flash_attn.cute.interface._flash_attn_fwd
     -> backward: CuteDSL bwd template -> _flash_attn_bwd
```

---

## 1) Reference Backend Trace (`ref = flex_attention(...)`)

### 1.1 Forward (reference)

### Frame R-FWD-1: `torch.nn.attention.flex_attention.flex_attention`
- Link: [flex_attention()](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1468)
- Before:
  - `q_ref`, `k_ref`, `v_ref`: FP32 tensors, `requires_grad=True`
  - `score_mod`: user function or `None`
  - `block_mask`: provided by caller or `None`
- Actions:
  - Validates input/device/head compatibility.
  - Replaces `score_mod=None` with `_identity`.
  - Replaces `block_mask=None` with `_create_empty_block_mask`.
  - Converts `BlockMask` to tuple via `block_mask.as_tuple()`.
- After:
  - Prepared normalized args for HOP call.

Annotated snippet:

```python
# file: torch/nn/attention/flex_attention.py
# link: ../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1582
if score_mod is None:
    score_mod = _identity  # [R-FWD] no-op score transform
if block_mask is None:
    block_mask = _create_empty_block_mask(query, key)  # [R-FWD] dense full-attend mask
...
out, lse, max_scores = flex_fn(
    query,
    key,
    value,
    score_mod,
    block_mask.as_tuple(),  # [R-FWD] mask metadata + mask_mod callable
    scale,
    kernel_options,
)
```

### Frame R-FWD-2: HOP dispatch enters Autograd implementation
- Link: [flex_attention_autograd](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L892)
- Before:
  - Grad is enabled and inputs require grad.
- Actions:
  - Builds `fw_graph` + `joint_graph` by tracing `score_mod` via `create_fw_bw_graph`.
  - Uses `FlexAttentionAutogradOp.apply(...)`.
- After:
  - Forward result produced and context stores saved tensors + graphs for backward.

### Frame R-FWD-3: `FlexAttentionAutogradOp.forward`
- Link: [FlexAttentionAutogradOp.forward](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747)
- Actions:
  - Calls HOP again inside `_AutoDispatchBelowAutograd` so it redispatches to non-autograd kernel implementation.

### Frame R-FWD-4: CompositeExplicitAutograd path (`sdpa_dense`)
- Link: [sdpa_dense](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379)
- Actions:
  - Calls `math_attention(...)` (eager mathematical reference implementation).

### Frame R-FWD-5: math implementation and mod/mask application
- Links:
  - [math_attention](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L218)
  - [_math_attention_inner](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L175)
  - [_vmap_for_bhqkv](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L275)
- Actions:
  - Computes dense scores: `scores = q @ k^T`.
  - vmaps `score_mod` over `(b, h, q_idx, kv_idx)`.
  - vmaps `mask_mod` similarly.
  - Computes: `post_mod_scores = where(mask_mod(...), score_mod(...), -inf)`.
  - Softmax + matmul with `v`.
- After:
  - Returns `(out, logsumexp/log(2), max/log(2))`.

Annotated snippet:

```python
# file: torch/_higher_order_ops/flex_attention.py
# link: ../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L207
scores = (scores * scale).to(working_precision)
post_mod_scores = torch.where(
    mask_mod(b, h, m, n, *mask_mod_other_buffers),  # [R-FWD] block_mask.mask_mod
    score_mod(scores, b, h, m, n, *score_mod_other_buffers),  # [R-FWD] user score_mod
    torch.tensor(-float("inf"), dtype=working_precision, device=scores.device),
)
```

### Frame R-FWD-6: return to API surface
- Link: [_finalize_outputs](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1655)
- Actions:
  - Converts internal log2-domain stats to natural-log domain (`* ln(2)`) when returning aux.

### Reference forward call path (complete)
1. `compare_backends(...): ref = flex_attention(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L86))
2. `torch.nn.attention.flex_attention.flex_attention` ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1468))
3. `torch._higher_order_ops.flex_attention.flex_attention_autograd` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L892))
4. `create_fw_bw_graph` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L644))
5. `FlexAttentionAutogradOp.forward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747))
6. HOP redispatch to `sdpa_dense` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379))
7. `math_attention` -> `_math_attention_inner` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L218))
8. Return `(out, lse, max)` and finalize outputs ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1655))

---

### 1.2 Backward (reference)

### Frame R-BWD-1: autograd engine enters custom backward
- Entry call: `ref.backward(grad_out)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L88))
- Backward impl: [FlexAttentionAutogradOp.backward](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810)

### Frame R-BWD-2: call `flex_attention_backward` HOP
- Link: [FlexAttentionAutogradOp.backward -> flex_attention_backward(...)](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L857)
- Actions:
  - Rebuilds block mask tuple and passes saved `fw_graph`, `joint_graph`, `mask_graph`, and saved tensors.

### Frame R-BWD-3: CompositeExplicitAutograd backward (`sdpa_dense_backward`)
- Link: [sdpa_dense_backward](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L946)
- Actions:
  - Recomputes score path, softmax path, and gradient chain.
  - Uses `joint_graph` to compute gradient of `score_mod` wrt pre-mod scores.
  - Applies mask_mod again to zero invalid gradient lanes.
  - Produces `grad_query`, `grad_key`, `grad_value`, and optional captured-buffer grads.

Annotated snippet:

```python
# file: torch/_higher_order_ops/flex_attention.py
# link: ../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L1064
joint_score_mod = _vmap_for_bhqkv(
    joint_graph,
    prefix=(0,),
    suffix=(0,) + captured_buffers_in_dim,
    out_dims=out_dims,
)
...
grad_scores, _, _, _, _, *grad_score_mod_captured = joint_score_mod(
    scores, b, h, m, n, grad_score_mod, *score_mod_other_buffers
)
```

### Frame R-BWD-4: gradients materialized
- Returned tuple populates `q_ref.grad`, `k_ref.grad`, `v_ref.grad`.

### Reference backward call path (complete)
1. `ref.backward(grad_out)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L88))
2. Autograd -> `FlexAttentionAutogradOp.backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810))
3. `flex_attention_backward` HOP call ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L857))
4. `sdpa_dense_backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L946))
5. Returns `(dq, dk, dv, captured_grads)` -> autograd writes `.grad`

---

## 2) Triton Backend Trace (`flex_triton = torch.compile(...BACKEND="TRITON")`)

### 2.1 Forward (Triton)

### Frame T-FWD-0: compile wrapper
- Link: [compile_flex](../thirdparty/attention-gym/examples/flex_flash_attention.py#L36)
- `flex_triton = torch.compile(partial(flex_attention, kernel_options={"BACKEND":"TRITON"}), dynamic=...)`

### Frame T-FWD-1: Dynamo-compiling entry in user API
- Link: [flex_attention dynamo path](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1684)
- Actions:
  - When Dynamo is active, API calls HOP directly:
    - marks static `head` and `head_dim`;
    - calls `flex_attention_hop(...)`.

### Frame T-FWD-2: Proxy tracing captures score/mask subgraphs
- Link: [trace_flex_attention](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406)
- Actions:
  - Builds `score_graph = make_fx(score_mod)(...)`.
  - Builds `mask_graph = make_fx(mask_mod)(...)`.
  - Registers both as modules on FX root.
  - Rewrites block_mask tuple to carry `mask_graph` in last slot.
  - Creates proxy call to HOP function with graph modules as args.

Annotated snippet:

```python
# file: torch/_higher_order_ops/flex_attention.py
# link: ../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L443
score_graph = reenter_make_fx(score_mod)(*example_vals, *score_mod_other_buffers)
mask_graph = reenter_make_fx(mask_mod)(*mask_example_vals, *mask_mod_other_buffers)
...
block_mask = block_mask[:-1] + (mask_graph,)
...
out_proxy = proxy_mode.tracer.create_proxy(
    "call_function", flex_attention, proxy_args, {}
)
```

### Frame T-FWD-3: Inductor lowering of `higher_order.flex_attention`
- Link: [register_lowering flex_attention](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107)
- Actions:
  - Unpacks block_mask tuple into sparse metadata tensors + `mask_graph`.
  - Sanitizes kernel options and extracts `backend`.
  - Builds lowered IR subgraphs for score and mask via `build_subgraph_buffer`.

### Frame T-FWD-4: backend decision + template setup
- Link: [backend logic](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L163)
- For explicit TRITON backend:
  - decode path disabled unless backend `TRITON_DECODE` or `AUTO`+eligible.
  - FLASH path disabled (`_use_flex_flash_attention` returns false for non-FLASH).
- Link: [_use_flex_flash_attention](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247)

### Frame T-FWD-5: Triton template choices + autotune
- Links:
  - [flex_attention_template](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L98)
  - [maybe_append_choice loop](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L388)
  - [autotune_select_algorithm](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L487)
- Actions:
  - Creates candidate kernels over config space.
  - Injects subgraphs (score/mask) into template via `subgraphs=[subgraph_buffer, mask_graph_buffer]`.
  - Autotunes and selects fastest choice.

### Frame T-FWD-6: score_mod/mask_mod inlined into Triton kernel source
- Links:
  - [Triton `modification` hook](../thirdparty/pytorch/torch/_inductor/select_algorithm.py#L881)
  - [common.py.jinja score/mask call sites](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L53)
- Actions:
  - `modification(subgraph_number=0, ...)` emits inlined score_mod compute.
  - `modification(subgraph_number=1, ...)` emits inlined mask_mod compute.

Annotated snippet:

```python
# file: templates/common.py.jinja
# link: ../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L53
{{ modification(
    subgraph_number=0,   # [T-FWD] score_mod graph
    output_name="post_mod_scores",
    score="qk", b="off_z", h="off_h", m="m", n="n", out="qk"
) }}
...
{{ modification(
    subgraph_number=1,   # [T-FWD] mask_mod graph
    output_name="mask_mod_output",
    score="qk", b="off_z", h="off_h", m="m", n="n"
) }}
```

### Frame T-FWD-7: runtime launch and return
- Generated Triton kernel runs and writes:
  - output tensor,
  - `LSE` (fp32),
  - `MAX` (fp32, if requested).
- Link: [template stores](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja#L200)

### Triton forward call path (complete)
1. `flex_triton(...)` compiled wrapper call ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L111))
2. `torch.nn.attention.flex_attention.flex_attention` dynamo branch ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1684))
3. HOP proxy dispatch -> `trace_flex_attention` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406))
4. `make_fx` capture of `score_mod` and `mask_mod` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L443))
5. Inductor lowering: `register_lowering(higher_order.flex_attention)` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107))
6. `build_subgraph_buffer` for score/mask graphs ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/common.py#L167))
7. Triton template candidate generation + autotune ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L388))
8. Inlined `modification()` in Jinja templates ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L53))
9. Triton kernel launch and result return

---

### 2.2 Backward (Triton)

### Frame T-BWD-1: autograd entry
- Entry call: `triton_out.backward(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L114))
- Since forward used autograd-enabled HOP path, backward routes through `FlexAttentionAutogradOp.backward`.
- Link: [FlexAttentionAutogradOp.backward](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810)

### Frame T-BWD-2: HOP backward trace capture
- Link: [trace_flex_attention_backward](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L1129)
- Actions:
  - Ensures `fw_graph`, `joint_graph`, and `mask_graph` are FX graph modules.
  - Creates proxy call for `higher_order.flex_attention_backward`.

### Frame T-BWD-3: Inductor lowering of backward HOP
- Link: [register_lowering flex_attention_backward](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L613)
- Actions:
  - Builds lowered forward score subgraph buffer.
  - Builds lowered joint graph outputs via `process_joint_outputs`.
  - Builds lowered mask graph buffer.
  - Computes `delta = sum(out * grad_out, axis=-1)` and adjusts with `grad_logsumexp` when provided.

### Frame T-BWD-4: Triton bwd template generation
- Links:
  - [backward template object](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L529)
  - [maybe_append_choice for bwd](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L938)
  - [bwd Jinja mod calls](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja#L365)
- Subgraph numbering in bwd template:
  - `0`: forward score_mod graph
  - `1`: joint graph (d(score_mod)/d(score) application)
  - `2`: mask_mod graph
  - `3`: captured-buffer gradient scatters

Annotated snippet:

```python
# file: templates/flex_backwards.py.jinja
# link: ../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja#L580
{{ modification(
    subgraph_number=1,   # [T-BWD] joint_graph: maps d(score_mod) -> d(score)
    output_name="grad_scores",
    score="pre_mod_scores", b="off_z", h="off_hq", m="m", n="n",
    grad_score_mod="dsT"
) }}
```

### Frame T-BWD-5: autotune, launch, reduction
- Link: [autotune + reduce broadcasted grads](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L1007)
- Actions:
  - Executes selected Triton bwd kernel.
  - If KV was batch-broadcasted, reduces `grad_key`/`grad_value` along batch.
  - Casts captured buffer grads back to original dtype.

### Triton backward call path (complete)
1. `triton_out.backward(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L114))
2. `FlexAttentionAutogradOp.backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810))
3. `higher_order.flex_attention_backward` trace proxy ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L1129))
4. Inductor lowering `register_lowering(higher_order.flex_attention_backward)` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L613))
5. Triton bwd template generation + subgraph injection ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L938))
6. Kernel launch, reductions, return `(dq, dk, dv, captured_grads)`

---

## 3) CuTeDSL FLASH Backend Trace (`flex_flash = torch.compile(...BACKEND="FLASH")`)

### 3.1 Forward (FLASH/CuTeDSL)

### Frame F-FWD-0: compile wrapper
- Link: [compile_flex](../thirdparty/attention-gym/examples/flex_flash_attention.py#L36)
- `flex_flash = torch.compile(partial(flex_attention, kernel_options={"BACKEND":"FLASH"}), ...)`

### Frame F-FWD-1: same Dynamo/HOP trace front-end as Triton
- Links:
  - [flex_attention dynamo branch](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1684)
  - [trace_flex_attention](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406)
- Result:
  - FX `score_graph` and `mask_graph` are captured and attached.

### Frame F-FWD-2: Inductor FLASH eligibility checks
- Link: [FLASH scalar capture guard](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L167)
- Actions:
  - `_has_unsupported_captured_scalars(...)` rejects unsupported dynamic scalar captures (e.g., symbolic closure scalars with `dynamic=True`).
- Link: [_has_unsupported_captured_scalars](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L189)

### Frame F-FWD-3: backend selection to FLASH lowering
- Link: [_use_flex_flash_attention](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247)
- Requirements:
  - backend must be exactly `FLASH`.
  - `flash_attn.cute` importable.
  - score-mod captured input buffers cannot require grad in this path.

### Frame F-FWD-4: create CuteDSL forward template choice
- Link: [create_flex_flash_attention_kernel](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284)
- Actions:
  - Detects `has_score_mod`, `needs_block_mask`, `has_full_blocks`.
  - Adds `flash_attention_cutedsl_template` choice with subgraphs.
  - Returns output node + LSE node.

### Frame F-FWD-5: Jinja emits `@cute.jit` wrappers for traced mods
- Link: [flash_attention.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention.py.jinja#L14)
- Actions:
  - Emits `score_mod(...)` wrapper from subgraph 0 when score_mod is non-trivial.
  - Emits `mask_mod(...)` wrapper (subgraph 1 or 0 depending on score_mod presence).
  - Builds `BlockSparseTensorsTorch(...)` from mask tensors.
  - Calls `_flash_attn_fwd(...)`.

Annotated snippet:

```python
# file: templates/flash_attention.py.jinja
# link: ../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention.py.jinja#L15
@cute.jit
def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    {{ modification(subgraph_number=0, ...) }}  # [F-FWD] traced score_graph inlined
    return tSrS_ssa
{{ set_cute_hash("score_mod", "score") }}      # [F-FWD] stable callable hash for compile cache
```

### Frame F-FWD-6: CuTeDSL kernel compilation handoff in Inductor
- Links:
  - [CuteDSLScheduling.define_kernel](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py#L59)
  - [async_compile.cutedsl](../thirdparty/pytorch/torch/_inductor/async_compile.py#L571)
- Actions:
  - Generated code is wrapped in `async_compile.cutedsl(kernel_name, source_code)`.
  - Runtime loads python module and extracts `<kernel_name>_main` entrypoint.

### Frame F-FWD-7: flash-attn interface compiles/executes CuTe kernel
- Link: [_flash_attn_fwd](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)
- Actions:
  - Computes `score_mod_hash` / `mask_mod_hash` (`hash_callable`).
  - Build compile key from dtype/head dims/mod hashes/mask-sparsity/etc.
  - On cache miss: instantiate `FlashAttentionForwardSm90` or `FlashAttentionForwardSm100` and call `cute.compile(...)`.
  - Execute cached compiled callable.

### Frame F-FWD-8: low-level score/mask execution in flash-attn kernel object
- Links:
  - [FlashAttentionForwardBase.apply_score_mod](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2370)
  - [apply_score_mod_inner](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343)
  - [AttentionMask.apply_mask](../thirdparty/flash-attention/flash_attn/cute/mask.py#L127)
- Actions:
  - `apply_score_mod_inner` applies softmax scale + user score_mod with per-element `(b,h,q_idx,kv_idx)`.
  - mask path applies `mask_mod` and/or causal/local/bounds logic.

### FLASH forward call path (complete)
1. `flex_flash(...)` compiled wrapper call ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L101))
2. `flex_attention` dynamo branch ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1684))
3. HOP proxy trace capture of score/mask graphs ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406))
4. Inductor lowering `higher_order.flex_attention` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107))
5. FLASH eligibility + create_flex_flash_attention_kernel ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284))
6. CuTeDSL codegen + scheduling (`async_compile.cutedsl`) ([link](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py#L94))
7. `_flash_attn_fwd` compile cache / `cute.compile` / execute ([link](../thirdparty/flash-attention/flash_attn/cute/interface.py#L406))
8. Return output + LSE

---

### 3.2 Backward (FLASH/CuTeDSL)

### Frame F-BWD-1: autograd entry and HOP backward call
- Entry: `flash_out.backward(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L104))
- Autograd function: [FlexAttentionAutogradOp.backward](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810)

### Frame F-BWD-2: FLASH bwd eligibility + constraints
- Link: [FLASH bwd branch in lowering](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L762)
- Constraints enforced before kernel generation:
  - no captured/mutated grads in joint outputs for this backend revision;
  - deterministic + block_mask unsupported in strict deterministic mode;
  - `grad_logsumexp` (dLSE path) is not supported in FLASH backward.

### Frame F-BWD-3: create CuteDSL backward template
- Link: [create_flex_flash_attention_backward_kernel](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L477)
- Inputs include optional subgraphs:
  - `fw_subgraph_buffer` (score_mod fwd graph)
  - `joint_subgraph_buffer` (score_mod_bwd graph)
  - `mask_graph_buffer` (mask_mod)

### Frame F-BWD-4: Jinja emits bwd wrappers (`score_mod`, `score_mod_bwd`, `mask_mod`)
- Link: [flash_attention_backward.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja#L19)

Annotated snippet:

```python
# file: templates/flash_attention_backward.py.jinja
# link: ../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja#L36
@cute.jit
def score_mod_bwd(grad_score_mod_ssa, tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    {{ modification(subgraph_number=1, ...) }}  # [F-BWD] joint graph inlined here
    return grad_score_mod_ssa_out
```

### Frame F-BWD-5: flash-attn bwd interface compile and execution
- Link: [_flash_attn_bwd](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)
- Actions:
  - Preprocess kernel compile/execute (`FlashAttentionBackwardPreprocess`).
  - Main bwd kernel compile/execute (`FlashAttentionBackwardSm90`/`Sm100`) with score_mod/score_mod_bwd/mask_mod callables.
  - Postprocess kernel(s) cast accumulators to output dtypes.

### Frame F-BWD-6: low-level bwd mod/mask application
- Links:
  - [FlashAttentionBackwardSm100.apply_score_mod](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm100.py#L1835)
  - [FlashAttentionBackwardSm100.apply_score_mod_bwd](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm100.py#L1874)
  - [apply_score_mod_bwd_inner](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L473)
  - [apply_mask_sm100_transposed](../thirdparty/flash-attention/flash_attn/cute/mask.py#L499)

### FLASH backward call path (complete)
1. `flash_out.backward(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L104))
2. `FlexAttentionAutogradOp.backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810))
3. Inductor lowering `higher_order.flex_attention_backward` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L613))
4. FLASH bwd selection/validation ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L762))
5. CuteDSL bwd template render ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja#L1))
6. `_flash_attn_bwd` preprocess/main/postprocess compile+run ([link](../thirdparty/flash-attention/flash_attn/cute/interface.py#L834))
7. Return `(dq, dk, dv)` to autograd engine

---

## 4) How `score_mod` and `block_mask` are Stitched Into Kernels

### 4.1 Reference path stitching
- `score_mod`:
  - Defaults to `_identity` at API layer when omitted.
  - vmapped across `(b,h,q_idx,kv_idx)` in `_math_attention_inner`.
- `block_mask`:
  - `BlockMask.as_tuple()` provides sparse metadata + `mask_mod` callable as final tuple slot.
  - `mask_mod` vmapped; final score matrix uses `where(mask_mod, score_mod, -inf)`.
- Links:
  - [_identity default](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1582)
  - [BlockMask.as_tuple](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L686)
  - [_math_attention_inner where()](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L209)

### 4.2 Triton path stitching
- `score_mod` and `mask_mod` are first converted to FX graph modules in `trace_flex_attention`.
- Inductor lowers graphs into IR buffers via `build_subgraph_buffer`.
- Triton Jinja templates call `modification(subgraph_number=...)`; the codegen hook inlines graph bodies directly into Triton code.
- Backward additionally inlines `joint_graph` for `d(score_mod)/d(score)`.
- Links:
  - [trace_flex_attention make_fx](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L443)
  - [build_subgraph_buffer](../thirdparty/pytorch/torch/_inductor/kernel/flex/common.py#L167)
  - [modification hook](../thirdparty/pytorch/torch/_inductor/select_algorithm.py#L881)
  - [fwd template calls](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L53)
  - [bwd template calls](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja#L365)

### 4.3 FLASH/CuTeDSL path stitching
- The same traced FX subgraphs are injected into CuTeDSL templates via `modification(...)` in `CuteDSLTemplateKernel`.
- Templates emit concrete `@cute.jit` wrappers:
  - forward: `score_mod`, `mask_mod`
  - backward: `score_mod`, `score_mod_bwd`, `mask_mod`
- Captured tensor buffers are collected and passed as `aux_tensors` to flash-attn.
- `set_cute_hash` assigns stable hashes so flash-attn compile cache (`hash_callable`) keys these callables efficiently.
- `block_mask` sparse tensors become `BlockSparseTensorsTorch(...)` args to `_flash_attn_fwd/_bwd`.
- Links:
  - [CuteDSL modification hook](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py#L404)
  - [buffer unpacking + hash setting](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py#L321)
  - [FLASH fwd template score/mask wrappers](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention.py.jinja#L15)
  - [FLASH bwd template score/score_bwd/mask wrappers](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja#L20)
  - [hash_callable behavior](../thirdparty/flash-attention/flash_attn/cute/utils.py#L42)

### 4.4 BlockMask dataflow details
- Construction:
  - `create_block_mask(mask_mod, ...)` computes dense mask, converts to block-sparse ordered tensors.
  - `BlockMask` stores forward KV sparse metadata and generated backward-Q sparse metadata.
- Runtime usage:
  - Forward kernels iterate with `kv_num_blocks/kv_indices` (plus optional `full_kv_*`).
  - Backward kernels iterate with `q_num_blocks/q_indices` (plus optional `full_q_*`).
- Links:
  - [create_block_mask](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1207)
  - [BlockMask structure](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L520)
  - [fwd unpack of block_mask tuple](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L147)
  - [bwd unpack of block_mask tuple](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L635)

### 4.5 FLASH dynamic scalar limitation (why)
- FLASH backend rejects captured dynamic scalars (symbolic closure scalars / CPU 0-d tensors from capture) because they cannot be reliably inlined into generated CuTeDSL kernel templates in this path.
- This check is explicit before subgraph buffer building.
- Link: [_has_unsupported_captured_scalars](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L189)

---

## 5) Full End-to-End Call Chains

### 5.1 Reference backend

### Forward chain
1. `compare_backends` callsite ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L86))
2. `torch.nn.attention.flex_attention.flex_attention` ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1468))
3. HOP `flex_attention_autograd` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L892))
4. `create_fw_bw_graph` for score_mod ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L644))
5. `FlexAttentionAutogradOp.forward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747))
6. HOP redispatch -> `sdpa_dense` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379))
7. `math_attention` / `_math_attention_inner` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L218))
8. Return and finalize outputs ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1655))

### Backward chain
1. `ref.backward(grad_out)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L88))
2. `FlexAttentionAutogradOp.backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810))
3. HOP `flex_attention_backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L857))
4. `sdpa_dense_backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L946))
5. Gradients returned to autograd

### 5.2 Triton backend

### Forward chain
1. `flex_triton = torch.compile(partial(flex_attention, BACKEND="TRITON"))` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L36))
2. Compiled call enters `flex_attention` dynamo branch ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1684))
3. HOP proxy trace: `trace_flex_attention` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406))
4. `make_fx` score_mod/mask_mod graphs ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L443))
5. Inductor lowering `higher_order.flex_attention` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107))
6. `build_subgraph_buffer` for both graphs ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/common.py#L167))
7. Triton template choice generation + autotune ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L388))
8. Jinja `modification(...)` inline score/mask logic ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L53))
9. Launch selected Triton kernel

### Backward chain
1. `triton_out.backward(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L114))
2. `FlexAttentionAutogradOp.backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810))
3. `trace_flex_attention_backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L1129))
4. Inductor lowering `higher_order.flex_attention_backward` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L613))
5. bwd template generation + joint/mask subgraph inlining ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja#L365))
6. Launch Triton bwd kernel, reduce grads, return

### 5.3 FLASH/CuTeDSL backend

### Forward chain
1. `flex_flash = torch.compile(partial(flex_attention, BACKEND="FLASH"))` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L36))
2. `flex_attention` dynamo branch ([link](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1684))
3. HOP proxy trace capture score/mask graphs ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406))
4. Inductor lowering `higher_order.flex_attention` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107))
5. FLASH eligibility checks + scalar capture guard ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L167))
6. `create_flex_flash_attention_kernel` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284))
7. CuTeDSL scheduling emits `async_compile.cutedsl(...)` ([link](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py#L94))
8. flash-attn `_flash_attn_fwd`: compile cache key + `cute.compile(...)` + execute ([link](../thirdparty/flash-attention/flash_attn/cute/interface.py#L406))

### Backward chain
1. `flash_out.backward(...)` ([link](../thirdparty/attention-gym/examples/flex_flash_attention.py#L104))
2. `FlexAttentionAutogradOp.backward` ([link](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810))
3. Inductor lowering `higher_order.flex_attention_backward` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L613))
4. FLASH bwd checks + `create_flex_flash_attention_backward_kernel` ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L477))
5. CuTeDSL bwd template emits `score_mod_bwd` + `mask_mod` wrappers ([link](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja#L36))
6. flash-attn `_flash_attn_bwd`: preprocess/main/postprocess `cute.compile(...)` pipeline ([link](../thirdparty/flash-attention/flash_attn/cute/interface.py#L834))
7. return `(dq, dk, dv)`

---

## Key Functions Index
| Function | File | Purpose |
|----------|------|---------|
| `compare_backends` | [flex_flash_attention.py](../thirdparty/attention-gym/examples/flex_flash_attention.py#L66) | Benchmark harness calling reference/TRITON/FLASH forward+backward |
| `compile_flex` | [flex_flash_attention.py](../thirdparty/attention-gym/examples/flex_flash_attention.py#L36) | Builds backend-fixed compiled callable |
| `flex_attention` | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1468) | Public API: validation, defaults, HOP invocation |
| `BlockMask.as_tuple` | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L686) | Packs sparse metadata + `mask_mod` for kernels |
| `create_block_mask` | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1207) | Converts mask function into sparse block metadata |
| `trace_flex_attention` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406) | Captures `score_mod`/`mask_mod` as FX graphs for compilation |
| `flex_attention_autograd` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L892) | Autograd-aware HOP entry |
| `FlexAttentionAutogradOp.forward` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747) | Saves tensors/graphs; redispatches below autograd |
| `FlexAttentionAutogradOp.backward` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L810) | Calls backward HOP and returns gradients |
| `sdpa_dense` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379) | Reference forward compute path |
| `sdpa_dense_backward` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L946) | Reference backward compute path |
| `register_lowering(...flex_attention)` | [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107) | Main Inductor lowering entry for forward HOP |
| `register_lowering(...flex_attention_backward)` | [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L613) | Main Inductor lowering entry for backward HOP |
| `_use_flex_flash_attention` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247) | FLASH backend gate |
| `_use_flex_flash_attention_backward` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L439) | FLASH bwd backend gate |
| `create_flex_flash_attention_kernel` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284) | Builds forward CuteDSL template choice |
| `create_flex_flash_attention_backward_kernel` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L477) | Builds backward CuteDSL template choice |
| `modification` (Triton codegen) | [torch/_inductor/select_algorithm.py](../thirdparty/pytorch/torch/_inductor/select_algorithm.py#L881) | Inlines FX subgraphs into Triton template source |
| `modification` (CuteDSL codegen) | [cutedsl_kernel.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py#L404) | Inlines FX subgraphs into CuteDSL template source |
| `define_kernel` (CuteDSL scheduling) | [cutedsl_scheduling.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py#L59) | Emits `async_compile.cutedsl(...)` wrapper |
| `_flash_attn_fwd` | [flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94) | FLASH forward interface: compile key, `cute.compile`, execute |
| `_flash_attn_bwd` | [flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554) | FLASH backward interface: preprocess/main/postprocess kernels |
| `apply_score_mod_inner` | [flash_attn/cute/softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343) | Shared per-element score_mod application logic |
| `apply_score_mod_bwd_inner` | [flash_attn/cute/softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L473) | Shared per-element score_mod backward logic |
| `AttentionMask.apply_mask` | [flash_attn/cute/mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L127) | Applies mask_mod/causal/local masking in forward |
| `AttentionMask.apply_mask_sm100_transposed` | [flash_attn/cute/mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L499) | Applies mask_mod/masking in transposed SM100 backward path |
| `hash_callable` | [flash_attn/cute/utils.py](../thirdparty/flash-attention/flash_attn/cute/utils.py#L42) | Hashes score/mask callables for compile caching |

## Process Log (How This Trace Was Produced)
This section documents execution workflow and decisions. It intentionally does not include private internal chain-of-thought reasoning.

### A. Inputs and constraints I followed
1. Read task spec from [FLEX_ATTN_IMPL.md](../FLEX_ATTN_IMPL.md).
2. Followed repository instructions in `AGENTS.md`:
   - Prompt user before edits.
   - For large edits, create a plan file under `repo_root/your_name/...`.
   - Keep notes brief in `~/.codex/NOTES/cutlass`.
3. Followed user constraints:
   - Current directory is acceptable.
   - Notify before source-code edits (none were needed for this task).
   - If running GPU-required code, run `attach-srun` and confirm GPU allocation first.
4. No GPU-required code was executed, so `attach-srun` was not required.

### B. Planning and artifacts created
1. Created implementation plan file:
   - [codex/flex_attention_trace_plan.md](./flex_attention_trace_plan.md)
2. Produced final trace artifact:
   - [codex/flex_attention_trace.md](./flex_attention_trace.md)

### C. Execution sequence used
1. Open benchmark entrypoint and identify concrete call sites:
   - [flex_flash_attention.py](../thirdparty/attention-gym/examples/flex_flash_attention.py#L66)
2. Trace public API and block-mask plumbing:
   - [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1468)
3. Trace HOP/autograd and FX graph capture of `score_mod`/`mask_mod`:
   - [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406)
4. Trace Inductor forward/backward lowerings and backend routing:
   - [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107)
5. Trace FLASH backend gates and template choice creation:
   - [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247)
6. Trace Triton template generation and inlining hooks:
   - [torch/_inductor/kernel/flex/templates/flex_attention.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja#L1)
   - [torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja#L1)
   - [torch/_inductor/kernel/flex/templates/common.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja#L3)
   - [torch/_inductor/select_algorithm.py](../thirdparty/pytorch/torch/_inductor/select_algorithm.py#L881)
7. Trace CuTeDSL codegen/scheduling and async compile handoff:
   - [cutedsl_kernel.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py#L404)
   - [cutedsl_scheduling.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_scheduling.py#L59)
   - [async_compile.py](../thirdparty/pytorch/torch/_inductor/async_compile.py#L571)
8. Trace flash-attn CuTe runtime interface and score/mask apply functions:
   - [flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)
   - [flash_attn/cute/softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343)
   - [flash_attn/cute/mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L127)
9. Consolidate per-backend forward/backward frame chains and annotate where `score_mod` and `block_mask` are stitched.
10. Build final key-function index with clickable links.

### D. Delegation / subagents
1. Subagents used: none.
2. Delegation instructions: not applicable.

### E. Validation performed
1. Verified that all key call sites in this document point to concrete files/lines.
2. Verified forward and backward call chains are covered for:
   - Reference path
   - Triton path
   - FLASH/CuTeDSL path
3. Verified no source-code behavior changes were made; this task is documentation/tracing only.

### F. Notes-system updates
1. Updated repo notes index:
   - `/home/jeromeku/.codex/NOTES/cutlass/index.md`
2. Updated day note:
   - `/home/jeromeku/.codex/NOTES/cutlass/2026-02-13.md`
