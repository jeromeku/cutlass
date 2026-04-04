# FlexAttention: Full Call-Path Trace

**Date:** 2026-02-13
**Author:** Claude
**Source:** `thirdparty/attention-gym/examples/flex_flash_attention.py`, lines 79–129
**Scope:** All three backends (reference, Triton, CuteDSL) — forward and backward passes — with how `score_mod` and `block_mask` are dynamically stitched in.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Entry Point: `flex_attention` User API](#2-entry-point-flex_attention-user-api)
3. [Backend: Reference (eager)](#3-backend-reference-eager)
4. [Backend: Triton](#4-backend-triton)
5. [Backend: CuteDSL / Flash](#5-backend-cutedsl--flash)
6. [Why Both `mask_mod` and `score_mod`?](#6-why-both-mask_mod-and-score_mod)
7. [How `score_mod` and `block_mask` Are Stitched In](#7-how-score_mod-and-block_mask-are-dynamically-stitched-in)
8. [Key Functions Index](#8-key-functions-index)
9. [Code Map](#9-code-map)

---

## 1. Architecture Overview

```
User Code
  │
  ▼
torch.nn.attention.flex_attention.flex_attention()    ← public API wrapper
  │  torch.compile() wraps this with Dynamo
  ▼
torch._higher_order_ops.flex_attention.FlexAttentionHOP  ← Higher-Order Operator
  │
  ├─ [EAGER / REFERENCE]  DispatchKey.CompositeExplicitAutograd
  │      └─ sdpa_dense() → math_attention()             ← FP32 vmap-based reference
  │
  ├─ [COMPILED: TRITON]   torch._inductor lowering
  │      └─ flex_attention() in kernel/flex/flex_attention.py
  │             └─ TritonTemplate("flex_attention")     ← Triton GPU kernel
  │
  └─ [COMPILED: FLASH]    torch._inductor lowering
         └─ create_flex_flash_attention_kernel()
                └─ CuteDSLTemplate → flash_attn.cute interface.py
                       └─ FlashAttentionForwardSm90 / Sm100
                              └─ cute.compile() → CUDA kernel
```

All three backends share the same public entry point and dispatch chain. The backend diverges in the inductor lowering pass.

---

## 2. Entry Point: `flex_attention` User API

### Frame 0 — User code

```python
# thirdparty/attention-gym/examples/flex_flash_attention.py, line 36–41
def compile_flex(backend: Literal["FLASH", "TRITON"], *, dynamic: bool) -> Callable:
    return torch.compile(
        partial(flex_attention, kernel_options={"BACKEND": backend}),
        dynamic=dynamic,
    )

flex_flash  = compile_flex("FLASH",  dynamic=False)
flex_triton = compile_flex("TRITON", dynamic=False)
```

`flex_attention` here is `torch.nn.attention.flex_attention.flex_attention`, which is the public API function. It is wrapped with `torch.compile`, which registers Dynamo to intercept the call at graph-capture time.

### Frame 1 — `torch.nn.attention.flex_attention.flex_attention`

**File:** [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1)

```python
def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Optional[_score_mod_signature] = None,  # signature: (score,b,h,q,kv)->score
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[FlexKernelOptions] = None,
    _aux_request: Optional[AuxRequest] = None,
) -> Tensor | tuple[Tensor, AuxOutput]:
```

**What happens here:**

1. **Default score_mod:** If `score_mod` is `None`, substitutes `_identity` (a no-op that returns `score` unchanged).
2. **Default block_mask:** If `block_mask` is `None`, creates a dense mask where every block is attended to.
3. **scale:** Defaults to `1/sqrt(head_dim)`.
4. **Flattens `block_mask`** into a plain tuple that is passed into the HOP:
   ```python
   block_mask = block_mask._as_tuple()
   # Returns: (q_len, kv_len, kv_num_blocks, kv_indices,
   #           full_kv_num_blocks, full_kv_indices,
   #           q_num_blocks, q_indices, full_q_num_blocks, full_q_indices,
   #           SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE, mask_mod_fn)
   ```
5. **Pulls other_buffers** from `score_mod`'s closure (captured tensors used inside `score_mod`) using pytree walk, separating them from the positional arguments `(score, b, h, q, kv)`.
6. **Dispatches into the HOP:**
   ```python
   out, lse, max_scores = flex_attention_hop(
       query, key, value,
       score_mod,           # the callable, will be traced by Dynamo
       block_mask,          # flat tuple
       scale,
       kernel_options or {},
       score_mod_other_buffers,  # captured tensors from score_mod closure
       mask_mod_other_buffers,   # captured tensors from mask_mod closure
   )
   ```

### Frame 2 — `FlexAttentionHOP.__call__`

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L94)

```python
class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_attention", cacheable=True)  # cacheable=True enables compile caching

    def __call__(self, query, key, value, score_mod, block_mask,
                 scale, kernel_options,
                 score_mod_other_buffers=(), mask_mod_other_buffers=()):
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)
        return super().__call__(...)   # dispatches via DispatchKey
```

`HigherOrderOperator` selects the implementation based on the active dispatch key stack:

| Dispatch Key | Implementation |
|---|---|
| `AutocastCUDA` / `AutocastCPU` | Cast q/k/v, re-dispatch |
| `CompositeExplicitAutograd` | `sdpa_dense()` → **reference path** |
| `Autograd` | `FlexAttentionAutogradOp` (wraps forward, builds backward graph) |
| `ProxyTorchDispatchMode` | `trace_flex_attention()` — Dynamo tracing |
| Inductor lowering | `register_lowering(flex_attention)` in kernel/flex/ |

---

## 3. Backend: Reference (Eager)

The reference path is hit when `flex_attention` is called **without** `torch.compile`, e.g. in the test harness:

```python
# flex_flash_attention.py, line 86
ref = flex_attention(q_ref, k_ref, v_ref, score_mod=score_mod, block_mask=block_mask)
```

### Frame 3 — `FlexAttentionAutogradOp.forward`

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747)

```python
class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, fw_graph, joint_graph, block_mask, scale,
                kernel_options, mask_mod_other_buffers, *score_mod_other_buffers):
        # Save for backward
        ctx._fw_graph = fw_graph           # the traced score_mod forward graph
        ctx._joint_graph = joint_graph     # the joint fwd+bwd graph for score_mod
        ctx._mask_graph = block_mask[-1]   # mask_mod callable
        ctx.scale = scale
        # Dispatch below the Autograd key
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp, max_scores = flex_attention(
                query, key, value, fw_graph, block_mask, scale,
                kernel_options, score_mod_other_buffers, mask_mod_other_buffers,
            )
        # out has shape [B, H, Sq, Dv]
        # logsumexp has shape [B, H, Sq], dtype=float32  ← saved for bwd
        ctx.save_for_backward(query, key, value, out, logsumexp, *score_mod_other_buffers)
        return out, logsumexp, max_scores
```

`_AutoDispatchBelowAutograd` causes the next dispatch to skip the `Autograd` key and hit `CompositeExplicitAutograd`.

### Frame 4 — `sdpa_dense` (reference implementation)

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379)

```python
@flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(query, key, value, score_mod, block_mask, scale, kernel_options,
               score_mod_other_buffers=(), mask_mod_other_buffers=()):
    out, lse, max_scores = math_attention(
        query, key, value, score_mod, block_mask, scale, kernel_options,
        score_mod_other_buffers, mask_mod_other_buffers,
    )
    # Permute strides of output to match query's memory layout
    out = _permute_strides(out, query.stride())
    return out, lse, max_scores
```

### Frame 5 — `math_attention`

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L218)

```python
def math_attention(query, key, value, score_mod, block_mask, scale, ...):
    # 1. GQA broadcast: repeat K,V heads to match Q
    G = query.size(1) // key.size(1)
    value = torch.repeat_interleave(value, G, dim=1)
    key   = torch.repeat_interleave(key,   G, dim=1)

    # 2. Compute raw dot-product scores in fp32
    #    scores: [B, H, Sq, Skv]
    working_precision = torch.float32
    scores = query.to(working_precision) @ key.to(working_precision).transpose(-2, -1)

    # 3. Build position index tensors for vectorized score_mod / mask_mod
    b = torch.arange(0, scores.size(0))   # batch indices
    h = torch.arange(0, scores.size(1))   # head indices
    m = torch.arange(0, scores.size(2))   # query position indices
    n = torch.arange(0, scores.size(3))   # key/value position indices

    # 4. vmap score_mod and mask_mod over all 4 dimensions (see §6)
    score_mod_vmapped = _vmap_for_bhqkv(score_mod, prefix=(0,), suffix=(None,)*len(other_buffers))
    mask_mod_vmapped  = _vmap_for_bhqkv(mask_mod,  prefix=(),   suffix=(None,)*len(mask_buffers))

    # 5. Apply mask then score_mod
    scores = scores * scale
    post_mod_scores = torch.where(
        mask_mod_vmapped(b, h, m, n, *mask_mod_other_buffers),   # bool mask
        score_mod_vmapped(scores, b, h, m, n, *score_mod_other_buffers),  # modified scores
        torch.tensor(-inf),                                       # masked-out → -∞
    )

    # 6. Numerically stable softmax + output
    logsumexp = post_mod_scores.logsumexp(dim=-1)
    post_mod_scores = torch._safe_softmax(post_mod_scores, dim=-1)

    return (
        post_mod_scores.to(query.dtype) @ value.to(query.dtype),  # [B,H,Sq,Dv]
        logsumexp / math.log(2),   # convert to log2 domain (matches kernel convention)
        max_scores / math.log(2),
    )
```

**State at exit:**
- `out`: `[B, H, Sq, Dv]`, dtype = input dtype
- `logsumexp`: `[B, H, Sq]`, float32, in log2 space
- `max_scores`: `[B, H, Sq]`, float32, in log2 space

### Frame 6 — Reference Backward

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L946)

The reference backward is **not** autograd-through-forward — it is a hand-written backward that mirrors the FlashAttention-2 algorithm, dispatched via `CompositeExplicitAutograd`:

```python
@flex_attention_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense_backward(query, key, value, out, logsumexp, grad_out, grad_logsumexp,
                        fw_graph, joint_graph, block_mask, scale, kernel_options,
                        score_mod_other_buffers=(), mask_mod_other_buffers=()):
    # 1. Convert logsumexp from log2 domain back to natural log
    #    (the forward stored it in log2 space for kernel efficiency)
    logsumexp = logsumexp * math.log(2)           # [B, H, Sq]

    # 2. Recompute post_mod_scores (re-run forward inner pass)
    scores, post_mod_scores = _math_attention_inner(
        query, key, value, fw_graph, block_mask, scale, ...
    )
    # post_mod_scores: [B, H, Sq, Skv], with score_mod and mask applied

    # 3. Recompute softmax weights P = softmax(post_mod_scores)
    post_mod_scores = post_mod_scores - logsumexp.unsqueeze(-1)  # subtract LSE
    softmax_scores = torch.exp(post_mod_scores)                  # [B, H, Sq, Skv]

    # 4. Compute dV = P^T dO
    grad_value = softmax_scores.transpose(-2, -1) @ grad_out     # [B, H, Dv, Skv]

    # 5. Compute dP = dO V^T
    grad_softmax_scores = grad_out @ value.transpose(-2, -1)     # [B, H, Sq, Skv]

    # 6. Compute delta = (out * grad_out).sum(dim=-1, keepdim=True)
    #    delta is the "row-wise dot product" used in FlashAttn-2 bwd
    delta = (out * grad_out).sum(dim=-1, keepdim=True)

    # 7. dS = P * (dP - delta)  [softmax backward]
    grad_scores = softmax_scores * (grad_softmax_scores - delta)

    # 8. Apply joint_graph: get d(score_mod)/d(score) and multiply into grad_scores
    #    joint_graph returns the VJP of score_mod w.r.t. its score input
    grad_scores = apply_joint_graph(joint_graph, grad_scores, ...)

    # 9. Apply mask: zero out masked positions in grad_scores
    #    (mask_mod tells us which positions were masked to -inf)
    grad_scores = apply_mask_mod(grad_scores, mask_mod, ...)
    grad_scores = grad_scores * scale              # apply 1/sqrt(d) scale

    # 10. dQ = dS @ K,  dK = dS^T @ Q
    grad_query = grad_scores @ key                 # [B, H, Sq, Dqk]
    grad_key   = grad_scores.transpose(-2, -1) @ query  # [B, H, Skv, Dqk]

    # 11. GQA: reduce dK and dV over grouped query heads
    G = query.size(1) // key.size(1)
    if G > 1:
        grad_key   = grad_key.reshape(*grad_key.shape[:1], -1, G, *grad_key.shape[2:]).sum(dim=2)
        grad_value = grad_value.reshape(*grad_value.shape[:1], -1, G, *grad_value.shape[2:]).sum(dim=2)

    return grad_query, grad_key, grad_value, tuple(grad_captured_buffers)
```

The `FlexAttentionAutogradOp.backward` simply invokes this via the `flex_attention_backward` HOP:

```python
# torch/_higher_order_ops/flex_attention.py, line 810
class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_out, grad_logsumexp, grad_max_scores):
        query, key, value, out, logsumexp, *score_mod_other_buffers = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_buffers = flex_attention_backward(
            query, key, value, out, logsumexp,
            grad_out, grad_logsumexp,
            ctx._fw_graph,    # forward score_mod graph (for recomputing P)
            ctx._joint_graph, # VJP graph (for d score_mod / d score)
            block_mask, ctx.scale, ctx.kernel_options,
            score_mod_other_buffers, mask_mod_other_buffers,
        )
        return grad_query, grad_key, grad_value, ...
```

**Key numerical note:** The logsumexp is stored in log₂ space (divided by `log(2)`) in the forward pass to allow efficient `exp2()` in GPU kernels. The backward converts it back with `× log(2)` before use.

---

## 4. Backend: Triton

Called via:
```python
flex_triton = compile_flex("TRITON", dynamic=False)
triton_out  = flex_triton(q, k, v, score_mod=score_mod, block_mask=block_mask)
```

`torch.compile` runs Dynamo, which captures an FX graph, then Inductor lowers it to Triton kernels.

### Frame 3 — Dynamo captures the graph

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L483)

```python
@flex_attention.py_impl(ProxyTorchDispatchMode)
def flex_attention_proxy_torch_dispatch_mode(mode, query, key, value, score_mod,
                                              block_mask, scale, ...):
    return trace_flex_attention(mode, query, key, value, score_mod, block_mask, ...)
```

### Frame 4 — `trace_flex_attention`

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406)

```python
def trace_flex_attention(proxy_mode, query, key, value, score_mod, block_mask, scale, ...):
    # 1. Run score_mod eagerly with fake scalar tensors to get example output
    example_vals = [query.new_zeros((), requires_grad=query.requires_grad)] \
                 + [query.new_zeros((), dtype=torch.int) for _ in range(4)]

    # 2. Trace score_mod into its own GraphModule
    score_graph = reenter_make_fx(score_mod)(*example_vals, *score_mod_other_buffers)
    # score_graph is now an FX GraphModule representing score_mod's computation

    # 3. Trace mask_mod into its own GraphModule
    mask_graph  = reenter_make_fx(mask_mod)(*mask_example_vals, *mask_mod_other_buffers)

    # 4. Register score_graph and mask_graph as submodules on the root tracer
    qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score")
    proxy_mode.tracer.root.register_module(qualname, score_graph)   # stored as "sdpa_score"
    mask_qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_mask")
    proxy_mode.tracer.root.register_module(mask_qualname, mask_graph) # stored as "sdpa_mask"

    # 5. Replace block_mask[-1] (callable mask_mod) with the traced mask_graph
    block_mask = block_mask[:-1] + (mask_graph,)

    # 6. Emit a call_function node in the FX graph for flex_attention
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", flex_attention, proxy_args, {}
    )
    return track_tensor_tree(example_out, out_proxy, ...)
```

After this, the Dynamo-captured FX graph has a single `flex_attention` node, where `score_graph` and `mask_graph` are **subgraph modules** attached to the graph. These are the compiled representations of `score_mod` and `mask_mod`.

### Frame 5 — Inductor lowering: `register_lowering(flex_attention)`

**File:** [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107)

```python
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(query, key, value, subgraph, block_mask, scale, kernel_options, ...):
```

This function is called by Inductor for every `flex_attention` call node in the FX graph. It:

**Step 1 — Parse block_mask tuple:**
```python
(
    _, _,                               # q_length, kv_length (not used here)
    kv_num_blocks, kv_indices,          # forward block lists
    full_kv_num_blocks, full_kv_indices,# "full" (unmasked) block lists
    q_num_blocks, q_indices,            # backward block lists
    full_q_num_blocks, full_q_indices,
    SPARSE_Q_BLOCK_SIZE,
    SPARSE_KV_BLOCK_SIZE,
    mask_graph,                         # traced mask_mod GraphModule
) = block_mask
```

**Step 2 — Backend selection:**
```python
kernel_options, backend = _sanitize_kernel_options_for_triton(kernel_options)
# backend is one of "AUTO" | "TRITON" | "FLASH" | "TRITON_DECODE"
```

**Step 3 — Build subgraph buffers (score_mod and mask_mod inlining):**

```python
# Creates IR placeholder tensors for (score, b, h, m, n) — the 5 args to score_mod
placeholder_inps = [
    create_placeholder("score", query.get_dtype(), device),  # fp16/bf16 scalar
    create_placeholder("b", torch.int32, device),
    create_placeholder("h", torch.int32, device),
    create_placeholder("m", torch.int32, device),
    create_placeholder("n", torch.int32, device),
]
# build_subgraph_buffer traces the score_mod FX graph with these IR placeholders,
# producing an IR subgraph that can be inlined into the Triton kernel template
subgraph_buffer = build_subgraph_buffer(
    placeholder_inps + list(score_mod_other_buffers),
    subgraph,   # the sdpa_score GraphModule
)

# Same for mask_mod (4 args: b, h, m, n — no score)
mask_graph_buffer = build_subgraph_buffer(
    mask_graph_placeholder_inps + list(mask_mod_other_buffers),
    mask_graph,
)
```

**Step 4 — Backend dispatch:**
```python
if _use_flex_flash_attention(..., backend=backend):
    return create_flex_flash_attention_kernel(...)   # → Flash / CuteDSL path

# Otherwise: Triton path
choices = []
for conf in configs:   # autotuning configs
    flex_attention_template.maybe_append_choice(
        choices,
        input_nodes=[query, key, value, ...],
        layout=layout,
        subgraphs=[subgraph_buffer, mask_graph_buffer],  # ← inline score_mod + mask_mod
        **kernel_options,
    )
autotune_select_algorithm("flex_attention", choices, ...)
```

### Frame 6 — `TritonTemplate` renders the kernel

**File:** [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L98)

```python
flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=load_flex_template("flex_attention")  # reads flex_attention.jinja2
          + load_flex_template("utilities")
          + load_flex_template("common"),
)
```

The Jinja2 template contains hooks like `{{ modification(score, b, h, m, n) }}` which are replaced with the inlined IR code from `subgraph_buffer`. The resulting Triton Python source is compiled by Triton into a CUDA kernel.

**Grid:**
```python
@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    return (cdiv(num_queries, meta["BLOCK_M"]), batch_size, q_heads)
    # Each block handles one BLOCK_M chunk of query sequence
    # Loops over KV blocks internally, only visiting blocks listed in kv_indices
```

### Frame 7 — Triton Backward

**File:** [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L700)  *(lowering for `flex_attention_backward` HOP)*

```python
@register_lowering(torch.ops.higher_order.flex_attention_backward, ...)
def flex_attention_backward(query, key, value, out, logsumexp, grad_out, grad_logsumexp,
                             fw_graph, joint_graph, block_mask, scale, kernel_options, ...):
    # 1. Build forward subgraph buffer (for re-computing scores in bwd pass)
    fw_subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), fw_graph
    )
    # 2. Build joint (forward+backward) subgraph buffer
    #    joint_graph computes both score_mod forward AND d(score_mod)/d(score) in one pass
    joint_subgraph_buffer = build_subgraph_buffer(
        joint_placeholder_inps + list(score_mod_other_buffers), joint_graph
    )
    # 3. Dispatch to Triton bwd template or Flash bwd
    if _use_flex_flash_attention_backward(..., backend=backend):
        return create_flex_flash_attention_backward_kernel(...)

    # Triton bwd: flex_attention_backward_template
    flex_attention_backward_template.maybe_append_choice(
        choices,
        input_nodes=[query, key, value, out, grad_out, logsumexp, ...],
        layout=layout,
        subgraphs=[fw_subgraph_buffer, joint_subgraph_buffer, mask_graph_buffer],
        **kernel_options,
    )
    # Returns (grad_query, grad_key, grad_value, grad_other_buffers)
```

The backward Triton kernel implements the standard FlashAttention-2 backward algorithm, but with `score_mod` and `mask_mod` inlined:
- **dV**: `dV += P^T dO`
- **dP**: `dP = dO V^T`
- **dS**: applies the joint graph to get `d(score_mod)/d(score)`, then scales by `P * (dP - delta)`
- **dQ, dK**: from dS via matrix multiplications

---

## 5. Backend: CuteDSL / Flash

Called via:
```python
flex_flash = compile_flex("FLASH", dynamic=False)
flash_out  = flex_flash(q_flash, k_flash, v_flash, score_mod=score_mod, block_mask=block_mask)
```

Frames 1–4 (user API → Dynamo tracing) are **identical to Triton**. The divergence happens at Frame 5 when `_use_flex_flash_attention` returns `True`.

### Frame 5 — `_use_flex_flash_attention`

**File:** [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247)

```python
def _use_flex_flash_attention(subgraph, mask_graph, kernel_options,
                               num_score_mod_placeholders, backend):
    if backend != "FLASH":
        return False   # Only activated when explicitly requested

    can_use, reason = _can_use_flex_flash_attention(subgraph, mask_graph, ...)
    if not can_use:
        raise RuntimeError(f"BACKEND='FLASH' but: {reason}")
    return True
```

Checks performed:
- `ensure_flash_available()`: verifies `flash_attn.cute` is importable
- `input_buffers_require_grads(...)`: captured tensors in score_mod must not require grad
- `_has_unsupported_captured_scalars(...)`: captured scalars (Python ints/floats captured as sympy.Symbol or 0-dim CPU tensors) are not supported — they cannot be serialized into the CuteDSL template

### Frame 6 — `create_flex_flash_attention_kernel`

**File:** [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284)

```python
def create_flex_flash_attention_kernel(query, key, value, block_mask, scale,
                                       kernel_options, subgraph_buffer, mask_graph_buffer, ...):
    # 1. Allocate output tensors in Inductor IR
    output = empty_strided(size=[B, H, Sq, Dv], stride=out_strides, dtype=dtype, ...)
    lse    = empty_strided(size=[B, H, Sq],       stride=None,       dtype=torch.float32, ...)

    # 2. Patch FixedLayout.make_indexer to use CuteDSL hierarchical indexing
    #    (see §6 for details)
    with patch_fixed_layout_indexer_for_cutedsl():
        error = flash_attention_cutedsl_template.maybe_append_choice(
            choices,
            input_nodes=[query, key, value, lse, kv_num_blocks, kv_indices, ...],
            layout=output_layout,
            mutated_inputs=[lse],
            subgraphs=[subgraph_buffer, mask_graph_buffer],  # ← score_mod + mask_mod
            SM_SCALE=scale,
            HAS_SCORE_MOD=has_score_mod,
            NEEDS_BLOCK_MASK=needs_block_mask,
            SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
            SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
        )

    # 3. Wrap choice render to apply CuteDSL indexer patch at render time
    for choice in choices:
        wrap_choice_render_with_cutedsl_indexer(choice)

    template_output = choices[0].output_node()
    return (template_output, lse)
```

### Frame 7 — `CuteDSLTemplate` renders and calls flash_attn.cute

**File:** [torch/_inductor/codegen/cutedsl/cutedsl_template.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_template.py)

The `CuteDSLTemplate` generates Python code that calls `flash_attn.cute.interface._flash_attn_fwd`. The generated code passes:
- `score_mod`: the inlined Python callable recovered from the subgraph
- `mask_mod`: the inlined Python callable from the mask subgraph
- Block mask tensors (`kv_num_blocks`, `kv_indices`, etc.)

### Frame 8 — `_flash_attn_fwd` (flash_attn.cute interface)

**File:** [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)

This is the main CuteDSL entry point. It:

**Step 1 — Validate and normalize inputs:**
```python
q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
# Validates shapes, dtypes (must be fp16 or bf16), device (must be CUDA)
# Computes softmax_scale = 1/sqrt(head_dim) if not provided
# Determines GQA ratio: qhead_per_kvhead = num_head // num_head_kv
```

**Step 2 — Allocate outputs:**
```python
out = torch.empty(*q_batch_seqlen_shape, num_head, head_dim_v, dtype=q.dtype)
lse = torch.empty(lse_shape, dtype=torch.float32)  # log-sum-exp for backward
```

**Step 3 — Build compile key (determines kernel caching):**
```python
compile_key = (
    dtype, head_dim, head_dim_v, qhead_per_kvhead,
    causal, score_mod_hash, mask_mod_hash,      # ← score_mod and mask_mod are part of the key!
    use_block_sparsity, ...,
    m_block_size, n_block_size, q_stage,
    num_threads, is_split_kv, pack_gqa,
    compute_capability, ...
)
```

`score_mod_hash` and `mask_mod_hash` are computed via `utils.hash_callable(score_mod)`, so different score_mods produce different compiled kernels.

**Step 4 — Instantiate kernel object (if not cached):**
```python
if compile_key not in _flash_attn_fwd.compile_cache:
    if compute_capability == 9:   # SM90 (Hopper)
        fa_fwd = FlashAttentionForwardSm90(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            is_causal=causal, pack_gqa=pack_gqa,
            tile_m=m_block_size, tile_n=n_block_size,
            num_stages=2, num_threads=num_threads,
            mask_mod=mask_mod,    # ← passed as Python callable
            score_mod=score_mod,  # ← passed as Python callable
            has_aux_tensors=...,
        )
    elif compute_capability in [10, 11]:  # SM100/SM110 (Blackwell)
        fa_fwd = FlashAttentionForwardSm100(...)
```

### Frame 9 — `cute.compile` — JIT compilation

**File:** [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L498)

```python
# Convert tensors to CuTe layout objects
q_tensor = to_cute_tensor(q)     # wraps torch.Tensor → cute.Tensor with layout info
k_tensor = to_cute_tensor(k)
v_tensor = to_cute_tensor(v)
o_tensor = to_cute_tensor(out)
lse_tensor = to_cute_tensor(lse, assumed_align=4)

# JIT-compile the kernel — score_mod and mask_mod are captured inside fa_fwd
_flash_attn_fwd.compile_cache[compile_key] = cute.compile(
    fa_fwd,               # FlashAttentionForwardSm90 Python object = kernel spec
    q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
    softmax_scale, current_stream,
    ...,
    options="--enable-tvm-ffi",   # TVM FFI backend for Python↔CUDA interop
)
```

`cute.compile` runs the CuteDSL JIT compiler:
1. Traces `FlashAttentionForwardSm90.__call__` with the tensor layouts to generate CuTe IR
2. Inlines `score_mod` and `mask_mod` into the per-tile inner loop at the point where they're called inside `FlashAttentionForwardSm90`
3. Lowers through MLIR → PTX → CUBIN via nvcc/ptxas
4. Caches the compiled CUBIN by `compile_key`

### Frame 10 — `FlashAttentionForwardSm90` — the kernel spec

**File:** [thirdparty/flash-attention/flash_attn/cute/flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py)

This is a CuteDSL `@cutlass.jit`-decorated class. Its `__call__` method describes the CUDA kernel in Python using cute ops:

```python
class FlashAttentionForwardSm90:
    def __init__(self, dtype, head_dim, ..., score_mod=None, mask_mod=None):
        self.score_mod = score_mod   # store for inlining
        self.mask_mod  = mask_mod

    def __call__(self, q, k, v, o, lse, softmax_scale, stream, ...):
        # This method is traced by cute.compile, not actually called at this point

        # Outer loop: iterate over query blocks (parallelized across SMs)
        for m_block in range(num_m_blocks):
            q_tile = load_q_tile(q, m_block)          # TMA load from global mem

            # Initialize running max and sum for online softmax
            row_max  = -inf
            row_sum  = 0.0

            # Inner loop: iterate over key/value blocks
            for n_block in kv_block_list:              # from block_mask sparse list
                k_tile = load_k_tile(k, n_block)      # TMA async load
                v_tile = load_v_tile(v, n_block)

                # QK matmul: scores = q_tile @ k_tile.T * scale
                scores = mma(q_tile, k_tile) * softmax_scale   # shape [BLOCK_M, BLOCK_N]

                # *** score_mod injection ***
                if self.score_mod is not None:
                    # b, h, m, n are tile-relative integer indices
                    for i in range(BLOCK_M):
                        for j in range(BLOCK_N):
                            b_idx = batch_idx
                            h_idx = head_idx
                            m_idx = m_block * BLOCK_M + i
                            n_idx = n_block * BLOCK_N + j
                            scores[i,j] = self.score_mod(scores[i,j], b_idx, h_idx, m_idx, n_idx)

                # *** mask_mod injection ***
                if self.mask_mod is not None and is_partial_block:
                    for i, j in ...:
                        if not self.mask_mod(b_idx, h_idx, m_idx, n_idx):
                            scores[i,j] = -inf

                # Online softmax update (Flash-style)
                new_max = max(row_max, scores.max())
                row_sum = row_sum * exp(row_max - new_max) + exp(scores - new_max).sum()
                row_max = new_max

                # PV accumulation: o_tile += softmax(scores) @ v_tile
                acc += softmax_approx(scores) @ v_tile

            # Finalize softmax, store output
            o_tile = acc / row_sum
            store_o_tile(o, o_tile, m_block)
            store_lse(lse, row_max + log(row_sum), m_block)
```

When `cute.compile` traces this, it inlines `score_mod` and `mask_mod` as CUDA C++ code at the exact point they're called in the inner loop, producing a single fused CUDA kernel.

### Frame 11 — Calling the compiled kernel

**File:** [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L520)

```python
# Cache hit: call the previously compiled kernel
_flash_attn_fwd.compile_cache[compile_key](
    q.detach(),        # actual torch.Tensor data
    k.detach(),
    v.detach(),
    out.detach(),
    lse,
    softmax_scale,
    current_stream,    # cuda.CUstream
    ...,
    aux_tensors,       # captured buffers for score_mod (e.g., ALiBi bias table)
)
# At return: out contains attention output, lse contains log-sum-exp for bwd
```

### Frame 12 — CuteDSL Flash Backward

**File:** [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)

The backward pass consists of **three separate compiled kernels**:

#### Kernel 1: Preprocess (`_flash_attn_bwd`, compile_cache_pre)

```python
fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim_v, arch, m_block_size, ...)
_flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(fa_bwd_pre, ...)
# Computes:
#   dpsum[b,h,q] = (out * dout).sum(dim=-1)   ← "delta" in FlashAttention-2 bwd
#   lse_log2[b,h,q] = lse[b,h,q] * log2(e)   ← convert to log2 for efficient exp2 in bwd
#   dq_accum[:] = 0                            ← zero accumulator for dQ
```

#### Kernel 2: Main backward (`_flash_attn_bwd`, compile_cache)

```python
if compute_capability == 9:
    fa_bwd_obj = FlashAttentionBackwardSm90(
        ..., score_mod=score_mod, score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod, ...
    )
else:
    fa_bwd_obj = FlashAttentionBackwardSm100(
        ..., score_mod=score_mod, score_mod_bwd=score_mod_bwd, ...
    )

_flash_attn_bwd.compile_cache[compile_key] = cute.compile(fa_bwd_obj, ...)
# Computes dK, dV (outer loop over KV blocks) and dQ_accum (scattered into accum)
# score_mod_bwd is the derivative of score_mod: dS[i,j] = score_mod_bwd(S[i,j])
```

#### Kernel 3: Postprocess (`_flash_attn_bwd`, compile_cache_post)

```python
fa_bwd_post = FlashAttentionBackwardPostprocess(dtype, head_dim, arch, m_block_size, ...)
_flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(fa_bwd_post, ...)
# Converts dq_accum (float32) → dq (bf16/fp16), applying softmax_scale
```

**Important:** `score_mod_bwd` — the derivative of `score_mod` — is derived inside Inductor using the `joint_graph`. For the backward, Inductor calls `create_fw_bw_graph` which uses `AOTConfig.create_joint` to compute the VJP of `score_mod`, producing a `joint_graph` that evaluates both the forward score_mod and its gradient simultaneously. This joint graph is then lowered separately and passed to the CuteDSL backward kernel as `score_mod_bwd`.

---

## 6. Why Both `mask_mod` and `score_mod`?

Before tracing how they're stitched in, it's worth understanding what each one is for — they are **different in kind**, not just in usage.

### `score_mod` — continuous modification of attention weights

```python
def score_mod(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    ...
```

`score_mod` takes a **scalar attention logit** (the raw QKᵀ dot product for one (q, kv) pair, before softmax) and returns a **modified scalar**. It can do anything differentiable: add a bias, apply a nonlinearity, scale, etc.

Examples:
- **ALiBi**: `return score - abs(q_idx - kv_idx) * slope[h]` — position-dependent penalty
- **Tanh softcap**: `return softcap * tanh(score / softcap)` — bounds logit magnitude
- **Learned bias**: `return score + bias_table[b, h, q_idx, kv_idx]` — additive learned offset

Because `score_mod` operates on a **float value** and must be **differentiable** (it sits inside the softmax), it participates in the backward pass. That's why the framework traces its VJP via `create_fw_bw_graph`.

### `mask_mod` — binary gating of attention positions

```python
def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:  # returns bool
    ...
```

`mask_mod` takes **only position indices** (no score) and returns a **boolean**: should this (q, kv) pair attend at all? A `False` result sets the logit to `-inf` before softmax, zeroing that position's weight.

Examples:
- **Causal**: `return q_idx >= kv_idx`
- **Sliding window**: `return abs(q_idx - kv_idx) <= window_size`
- **Document boundaries**: `return doc_id[q_idx] == doc_id[kv_idx]`

### Why not just one function?

| | `score_mod` | `mask_mod` |
|---|---|---|
| Input | float score + indices | indices only |
| Output | modified float | bool |
| Differentiable? | Yes — needs backward | No — just gates |
| Block-sparse precomputable? | **No** — needs the score value | **Yes** — index-only, runs at mask build time |
| Benefit of early evaluation | n/a | Entire K/V tiles skipped, zero compute |

The split exists because **structural sparsity** (which tokens can ever attend to each other) can be determined purely from positions, while **content-dependent modifications** need the actual dot-product value. By separating them:

1. `create_block_mask()` pre-evaluates `mask_mod` over the full (Sq × Skv) grid to build sparse block lists — **once**, before the kernel runs, on CPU
2. The kernel uses those lists to skip entire K/V tiles (zero memory loads, zero FLOPs)
3. At tile boundaries ("partial blocks"), `mask_mod` is re-applied element-wise as a fine-grained check
4. `score_mod` runs only on the tiles that survived the sparse filter

You *could* implement causal masking in `score_mod` via `score = -inf if q_idx < kv_idx else score`, but you'd lose the block-sparse speedup — the kernel would still load every K/V tile and just null out the upper-triangle elements. The `mask_mod` / `score_mod` split is what makes structural sparsity a first-class optimization rather than a masked softmax.

---

## 6. How `score_mod` and `block_mask` Are Dynamically Stitched In

### 6.1 `score_mod` stitching

The key insight is that `score_mod` is traced into a sub-FX graph and then **inlined** at the kernel code generation stage. The flow is:

```
User's Python function
    ↓
reenter_make_fx(score_mod)(*example_vals)  [in trace_flex_attention]
    ↓
FX GraphModule "sdpa_score"               ← sub-graph with all ops captured
    ↓
build_subgraph_buffer(placeholder_inps + other_buffers, subgraph)  [in inductor lowering]
    ↓
SubgraphResults (Inductor IR nodes)        ← IR nodes that can be inlined
    ↓
TritonTemplate or CuteDSLTemplate sees SubgraphResults
    ↓
Template rendering: inlines IR code where {{ modification(...) }} appears
    ↓
Final GPU kernel: score_mod ops are fused into the attention inner loop
```

For example, `generate_alibi_bias(H)` returns:
```python
def alibi_bias(score, b, h, q_idx, kv_idx):
    bias = -torch.abs(q_idx - kv_idx) * alibi_slopes[h]
    return score + bias
```

After tracing, the `sdpa_score` graph contains nodes: `abs`, `sub`, `getitem` (to index `alibi_slopes`), `mul`, `neg`, `add`. These get inlined into the Triton kernel as Triton IR ops, or into the CuteDSL kernel as CuTe/CUDA C++ ops.

### 6.2 `block_mask` / `mask_mod` stitching

`block_mask` has two components:

**A. Sparse block lists** (`kv_num_blocks`, `kv_indices`, etc.)
These are regular tensors passed as kernel arguments. The kernel uses them to skip entirely zero (masked-out) blocks:

```python
# Triton kernel inner loop (simplified):
for block_n in range(kv_num_blocks[batch, head, q_block]):    # only iterate valid blocks
    kv_block = kv_indices[batch, head, q_block, block_n]
    k_tile = load_k(k, kv_block)
    ...
```

**B. `mask_mod` callable** (for partial / boundary blocks)
Full blocks that pass the sparse filter are computed without masking (fast path). Only **partial** blocks at the boundary are subject to `mask_mod`:

```python
# Triton / CuteDSL inner loop for boundary blocks:
is_full_block = block_n < full_kv_num_blocks[...]
if not is_full_block:
    # Apply mask_mod element-wise within the block
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if not mask_mod(b, h, m_base+i, n_base+j):
                scores[i, j] = -inf
```

### 6.3 CuteDSL-specific: `HierarchicalIndex` and indexer patching

CuteDSL requires multi-dimensional tensor indices (not flat linear offsets), while Inductor normally uses flat SymPy offsets. The solution:

**File:** [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L51)

```python
class HierarchicalIndex(sympy.Function):
    """Wraps N-D index tuple as a single sympy.Expr node for CuteDSL codegen."""
    @classmethod
    def eval(cls, *args):
        return None   # inert — no simplification

@contextmanager
def patch_fixed_layout_indexer_for_cutedsl():
    """Temporarily swap FixedLayout.make_indexer to emit HierarchicalIndex nodes."""
    original_make_indexer = FixedLayout.make_indexer

    def cutedsl_make_indexer(self):
        def indexer(indices):
            return HierarchicalIndex(*indices)  # wrap (i, j, k, ...) → one SymPy node
        return indexer

    FixedLayout.make_indexer = cutedsl_make_indexer
    try:
        yield
    finally:
        FixedLayout.make_indexer = original_make_indexer
```

When CuteDSL codegen sees a `HierarchicalIndex(b, h, q, kv)` node, it emits `tensor[b, h, q, kv]` in the generated CUDA/CuTe code, letting CuteDSL handle stride arithmetic. This is critical for index tensors like `alibi_slopes[h]` inside `score_mod`.

### 6.4 `create_fw_bw_graph` — deriving the backward score_mod

**File:** [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L644)

```python
def create_fw_bw_graph(score_mod, index_values, other_buffers):
    def joint_f(score, b, h, m, n, example_grad, *other_buffers):
        # Define forward function
        def fw_with_masks(*args):
            fw_out = score_mod(*args)
            return ((fw_out,), (fw_out.requires_grad,))

        # Create joint fwd+bwd computation using AOT Autograd
        joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
        args = [score, b, h, m, n] + list(other_buffers)
        optional_grad = [example_grad] if example_grad.requires_grad else []
        _, grads = joint(args, optional_grad)   # grads[0] = d(score_mod)/d(score)
        return grads

    # Trace joint_f into an FX graph (the "joint_graph")
    joint_graph = make_fx(joint_f)(
        *unwrapped_score_mod_indexes, example_grad, *unwrapped_other_buffers
    )
    return score_mod, joint_graph
```

The `joint_graph` computes both `score_mod(score, ...)` **and** its gradient `d score_mod / d score` in a single fused pass, which is then inlined into the backward kernel as `score_mod_bwd`.

---

## 7. Key Functions Index

| Function | File | Purpose |
|---|---|---|
| `flex_attention` (public) | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L1) | User-facing API; normalizes args, calls HOP |
| `FlexAttentionHOP.__call__` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L94) | Higher-Order Operator; dispatches by key |
| `sdpa_dense` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379) | Reference (eager) forward |
| `math_attention` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L218) | vmap-based FP32 attention reference |
| `_vmap_for_bhqkv` | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L275) | 4-level vmap over (b, h, q, kv) dims |
| `FlexAttentionAutogradOp.forward` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747) | Autograd wrapper; saves tensors for bwd |
| `FlexAttentionAutogradOp.backward` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L800) | Invokes `flex_attention_backward` HOP |
| `create_fw_bw_graph` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L644) | AOT Autograd joint graph for score_mod derivative |
| `trace_flex_attention` | [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406) | Dynamo: traces score_mod + mask_mod into FX graphs |
| `flex_attention` (lowering) | [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107) | Inductor: selects backend and creates kernel |
| `build_subgraph_buffer` | [torch/_inductor/kernel/flex/common.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/common.py) | Creates inlinable IR from score_mod FX graph |
| `_use_flex_flash_attention` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247) | Decides whether to use Flash backend |
| `create_flex_flash_attention_kernel` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284) | Creates CuteDSL template choice |
| `patch_fixed_layout_indexer_for_cutedsl` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L96) | Patches indexer to emit hierarchical indices |
| `HierarchicalIndex` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L51) | SymPy wrapper for N-D CuteDSL indices |
| `_flash_attn_fwd` | [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94) | CuteDSL: validates, caches, calls forward kernel |
| `_flash_attn_bwd` | [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554) | CuteDSL: 3-phase backward (pre/main/post) |
| `FlashAttentionForwardSm90` | [thirdparty/flash-attention/flash_attn/cute/flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | CuteDSL kernel spec for SM90 (Hopper) |
| `FlashAttentionForwardSm100` | [thirdparty/flash-attention/flash_attn/cute/flash_fwd_sm100.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd_sm100.py) | CuteDSL kernel spec for SM100 (Blackwell) |
| `FlashAttentionBackwardSm90` | [thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py) | CuteDSL backward kernel for SM90 |
| `FlashAttentionBackwardPreprocess` | [thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py) | Computes delta = (o * do).sum, lse in log2 |
| `FlashAttentionBackwardPostprocess` | [thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py) | Converts dq_accum (fp32) → dq (fp16/bf16) |
| `ensure_flash_available` | [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L27) | Checks flash_attn.cute importable |
| `BlockMask` | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py#L478) | Block-sparse attention mask format |
| `create_block_mask` | [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py) | Eagerly evaluates mask_mod to build sparse block lists |

---

## 8. Code Map

### Entry point and user API

- [flex_flash_attention.py (example)](../thirdparty/attention-gym/examples/flex_flash_attention.py)
- [torch/nn/attention/flex_attention.py](../thirdparty/pytorch/torch/nn/attention/flex_attention.py)

### Higher-Order Operator & Dispatch

- [torch/_higher_order_ops/flex_attention.py](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py)
  - `FlexAttentionHOP` — [line 94](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L94)
  - `math_attention` — [line 218](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L218)
  - `sdpa_dense` — [line 379](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L379)
  - `trace_flex_attention` — [line 406](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L406)
  - `create_fw_bw_graph` — [line 644](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L644)
  - `FlexAttentionAutogradOp` — [line 747](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L747)

### Inductor Lowering

- [torch/_inductor/kernel/flex/flex_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py)
  - `flex_attention` lowering — [line 107](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L107)
  - `flex_attention_grid` — [line 73](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L73)
- [torch/_inductor/kernel/flex/flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py)
  - `ensure_flash_available` — [line 27](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L27)
  - `HierarchicalIndex` — [line 51](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L51)
  - `patch_fixed_layout_indexer_for_cutedsl` — [line 96](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L96)
  - `_use_flex_flash_attention` — [line 247](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L247)
  - `create_flex_flash_attention_kernel` — [line 284](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L284)
  - `create_flex_flash_attention_backward_kernel` — [line 477](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L477)
- [torch/_inductor/kernel/flex/flex_decoding.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_decoding.py)
- [torch/_inductor/kernel/flex/flex_cpu.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_cpu.py)

### CuteDSL / Flash-Attention Kernels

- [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py)
  - `_flash_attn_fwd` — [line 94](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)
  - `_flash_attn_bwd` — [line 554](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)
  - `FlashAttnFunc` — [line 1261](../thirdparty/flash-attention/flash_attn/cute/interface.py#L1261)
- [thirdparty/flash-attention/flash_attn/cute/flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) — SM90 fwd kernel
- [thirdparty/flash-attention/flash_attn/cute/flash_fwd_sm100.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd_sm100.py) — SM100 fwd kernel
- [thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py) — SM90 bwd kernel
- [thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm100.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm100.py) — SM100 bwd kernel
- [thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py)
- [thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py)
- [thirdparty/flash-attention/flash_attn/cute/cute_dsl_utils.py](../thirdparty/flash-attention/flash_attn/cute/cute_dsl_utils.py) — `to_cute_tensor`

### Triton / CuteDSL Jinja2 Templates

- [torch/_inductor/kernel/flex/templates/flex_attention.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja) — Triton forward template
- [torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja) — Triton backward template
- [torch/_inductor/kernel/flex/templates/flash_attention.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention.py.jinja) — CuteDSL forward template
- [torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja) — CuteDSL backward template
- [torch/_inductor/kernel/flex/templates/common.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/common.py.jinja) — Shared template utilities
