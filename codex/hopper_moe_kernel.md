# Hopper MoE Kernel Trace

## Scope

This trace starts at [moe_minimal.py#L91](../thirdparty/sonic-moe/tests/moe_minimal.py#L91):

```python
y_kernel = moe_kernel(x_kernel, kernel_backend_moe=kernel_backend_moe)[0]
```

and follows the forward path down to the grouped GEMM wrapper and CuTe DSL launch owned by `grouped_gemm.HopperWgmma_MoE_kernel`.

The concrete test configuration in this trace is:

- `T=8192`
- `H=768`
- `I=256`
- `E=128`
- `K=8`
- `dtype=torch.bfloat16`
- `activation=SWIGLU`
- `kernel_backend_moe=KernelBackendMoE.sonicmoe`
- `add_bias=False`
- `is_compiling=False`
- `use_quack_gemm=False`

That matters because several heuristics inside `HopperWgmma_MoE_kernel` branch on `I`, activation type, and whether `A` is gathered or TMA-loaded.

## Overall Flow

```text
test_moe line 91
  -> MoE.forward
    -> moe_TC_softmax_topk_layer
      -> top-k routing + metadata
      -> _UpProjection.apply
        -> _up_projection_forward
          -> HopperWgmma_MoE_Up_proj_Fwd.__init__   [cold compile path only]
            -> HopperWgmma_MoE_kernel.__init__
          -> cute.compile(...)
          -> HopperWgmma_MoE_Up_proj_Fwd.__call__
            -> HopperWgmma_MoE_kernel.__call__
              -> self.kernel(...).launch(...)
```

## Why This Kernel Looks Different From A Dense GEMM

This is not a simple dense `A @ B` launch.

- `A` is token data that must be regrouped by expert, so the up-projection path uses `is_A_gather=True`.
- `B` is expert weight data that remains regular enough to be TMA-loaded efficiently.
- The kernel is persistent, so CTAs stay resident and pull work from a tile scheduler instead of launching one CTA per output tile.
- The output path may need both normal output `D` and an auxiliary activation buffer `Y`, so the epilogue is tiled and pipelined instead of being a single flat store loop.
- TMA descriptors are updated at runtime because the effective tensor slice changes whenever the scheduler advances to a new expert-token group.

Those constraints explain most of the code structure.

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `test_moe` | [../thirdparty/sonic-moe/tests/moe_minimal.py#L35](../thirdparty/sonic-moe/tests/moe_minimal.py#L35) | Test entrypoint that invokes the MoE forward path |
| `MoE.forward` | [../thirdparty/sonic-moe/sonicmoe/moe.py#L208](../thirdparty/sonic-moe/sonicmoe/moe.py#L208) | Chooses the Sonic fused path for the MoE layer |
| `moe_TC_softmax_topk_layer` | [../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L428](../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L428) | Builds routing metadata and dispatches up/down projections |
| `_UpProjection.forward` | [../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L94](../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L94) | Autograd wrapper for the grouped GEMM up-projection |
| `_up_projection_forward` | [../thirdparty/sonic-moe/sonicmoe/functional/forward.py#L54](../thirdparty/sonic-moe/sonicmoe/functional/forward.py#L54) | Converts Torch tensors to CuTe tensors and handles compile caching |
| `HopperWgmma_MoE_Up_proj_Fwd.__init__` | [../thirdparty/sonic-moe/sonicmoe/functional/moe_config.py#L39](../thirdparty/sonic-moe/sonicmoe/functional/moe_config.py#L39) | Chooses the concrete Hopper kernel configuration |
| `HopperWgmma_MoE_Up_proj_Fwd.__call__` | [../thirdparty/sonic-moe/sonicmoe/functional/moe_config.py#L116](../thirdparty/sonic-moe/sonicmoe/functional/moe_config.py#L116) | Adapts wrapper arguments into a grouped GEMM launch |
| `HopperWgmma_MoE_kernel.__init__` | [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L75](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L75) | Derives tile, warp-group, stage, and tensormap policy |
| `HopperWgmma_MoE_kernel.__call__` | [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L751](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L751) | Builds CuTe MMA/TMA objects, scheduler state, shared storage, and launches the kernel |
| `HopperWgmma_MoE_kernel.kernel` | [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1360](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1360) | CuTe DSL kernel body with producer warps, WGMMA mainloop, and epilogue |

## Frame 0: The Test Entrypoint

Source: [../thirdparty/sonic-moe/tests/moe_minimal.py#L88](../thirdparty/sonic-moe/tests/moe_minimal.py#L88)

```python
with torch.autocast(x_torch.device.type, torch.float32):
    # Run the kernel path under AMP. Inputs remain bf16, but many reductions
    # and the router math are allowed to use float32 where the underlying ops choose to.
    with enable_quack_gemm(use_quack_gemm):
        # In this test, use_quack_gemm=False, so the Sonic grouped GEMM path
        # stays active and the QuACK fallback stays disabled.
        y_kernel = moe_kernel(x_kernel, kernel_backend_moe=kernel_backend_moe)[0]
        # `moe_kernel` is just the raw `MoE` module here because `is_compiling=False`.
        # `[0]` discards aux_loss and keeps only the hidden-state output tensor.
```

State before this frame:

- `x_kernel.shape == (8192, 768)`
- `x_kernel.dtype == torch.bfloat16`
- `moe_kernel` is an instance of `sonicmoe.moe.MoE`
- `kernel_backend_moe == KernelBackendMoE.sonicmoe`

State after this frame:

- control transfers to `MoE.forward`

## Frame 1: `MoE.forward` Selects The Sonic Fast Path

Source: [../thirdparty/sonic-moe/sonicmoe/moe.py#L208](../thirdparty/sonic-moe/sonicmoe/moe.py#L208)

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    kernel_backend_moe: KernelBackendMoE = KernelBackendMoE.sonicmoe,
    is_inference_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = hidden_states.shape
    # Save the incoming shape so the output can be restored later.

    hidden_states = hidden_states.view(-1, self.hidden_size)
    # Flatten batch/query dimensions into `(total_tokens, hidden_size)`.
    # For this test, the tensor is already effectively `(8192, 768)`.

    if kernel_backend_moe == KernelBackendMoE.sonicmoe and self.num_experts <= 32768:
        hidden_states, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            hidden_states,
            self.router.weight,
            self.c_fc.weight.permute(1, 2, 0),
            self.c_fc.bias,
            self.c_proj.weight.permute(1, 2, 0),
            self.c_proj.bias,
            self.top_k,
            self.stream_id,
            self.activation_function,
            is_inference_mode or not self.training,
        )
```

Line-by-line logic:

- `original_shape = hidden_states.shape`
  Keeps the pre-flattened logical shape for the return path.
- `hidden_states = hidden_states.view(-1, self.hidden_size)`
  Converts the MoE layer into token-major form.
- `if kernel_backend_moe == ...`
  Picks the fused path. This trace goes here because the test sets `sonicmoe`, and `E=128` satisfies the expert-count guard.
- `self.router.weight`
  Supplies the router linear weight with shape `(E, H) = (128, 768)`.
- `self.c_fc.weight.permute(1, 2, 0)`
  Reorders the expert up-projection weight into `(2I, H, E) = (512, 768, 128)` for SWIGLU.
- `self.c_proj.weight.permute(1, 2, 0)`
  Reorders the expert down-projection weight into `(H, I, E) = (768, 256, 128)`.
- `self.top_k`
  Here `K=8`, so each token is routed to 8 experts.
- `self.stream_id`
  Hands the current CUDA stream down into the custom ops and CuTe DSL launch layer.
- `is_inference_mode or not self.training`
  Keeps the up-projection kernel aware of whether it can skip some training-only outputs. In the test this is `False`.

Important consequence:

- `moe_TC_softmax_topk_layer` becomes the real control hub for routing plus grouped GEMM.

## Frame 2: Routing And Group Construction In `moe_TC_softmax_topk_layer`

Source: [../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L428](../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L428)

```python
E = router_w.size(0)
router_logits = F.linear(x, router_w)
# Dense router projection: `(T, H) x (E, H)^T -> (T, E)` = `(8192, 128)`.

topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)
# Produces the top-8 experts per token and their post-softmax routing weights.

T, K = topk_indices.size()
TK = T * K
device = topk_indices.device
# `TK=8192*8=65536` is the number of routed token-expert pairs.

s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
# These tensors describe how the irregular token->expert mapping is linearized.

TC_topk_router_metadata_triton(
    topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
)
# A Triton helper computes:
# - how many routed tokens go to each expert,
# - cumulative offsets per expert,
# - which input token each grouped-GEMM row should read,
# - and how to scatter results back.

y1, z = _UpProjection.apply(
    x,
    w1,
    b1,
    expert_frequency_offset,
    T * K,
    K,
    stream_id,
    x_gather_idx,
    s_scatter_idx,
    s_reverse_scatter_idx,
    None,
    False,
    activation_type,
    is_inference_mode_enabled,
)
```

What these metadata tensors mean:

- `expert_frequency_offset[e]` marks the starting row in the grouped GEMM for expert `e`.
- `x_gather_idx[row]` says which original token to load into grouped-GEMM row `row`.
- `s_scatter_idx` and `s_reverse_scatter_idx` are the inverse maps needed for backward and output recombination.

This is the first MoE-specific irregularity. Dense GEMM would read `A` contiguously. This kernel instead manufactures a grouped row order and later implements a specialized `gatherA` load path for it.

## ASCII: What `gatherA` Means

```text
Original token matrix X:           Grouped rows for expert GEMM:

token 0  -> experts 2,17,41,...   row 0   -> token 33, expert 0
token 1  -> experts 5,17,18,...   row 1   -> token 402, expert 0
token 2  -> experts 0,17,99,...   row 2   -> token 1001, expert 0
...                                ...
                                   row 900 -> token 2, expert 17

x_gather_idx[row] tells the kernel which original token row to fetch.
expert_frequency_offset tells the kernel where each expert's row block begins.
```

## Frame 3: `_UpProjection.forward` Sets Up The Grouped GEMM

Source: [../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L111](../thirdparty/sonic-moe/sonicmoe/functional/__init__.py#L111)

```python
T, H = x.shape
I, H, E = w1.shape
is_glu_activation = is_glu(activation_type)
if is_glu_activation:
    I //= 2
TK = total_expert_freq

if is_using_quack_gemm():
    ...
else:
    z = torch.empty(TK, (2 * I if is_glu_activation else I), dtype=x.dtype, device=x.device)
    y1 = torch.empty(TK, I, dtype=x.dtype, device=x.device)
    _up_projection_forward(
        x=x,
        w1=w1,
        z=z,
        y1=y1,
        b1=b1,
        expert_frequency_offset=expert_frequency_offset,
        expert_schedule_order=None,
        x_gather_idx=x_gather_idx,
        stream_id=stream_id,
        activation_type=activation_type.value,
        is_glu_activation=is_glu_activation,
        is_inference_mode_enabled=is_inference_mode_enabled,
    )
```

Line-by-line logic:

- `T, H = x.shape`
  The input tokens are still in ordinary token-major order.
- `I, H, E = w1.shape`
  `w1` is already in kernel-friendly `(2I, H, E)` layout.
- `if is_glu_activation: I //= 2`
  SWIGLU stores gate and up branches interleaved in the first dimension, so the logical intermediate size is half the stored weight dimension.
- `z = torch.empty(TK, 2 * I, ...)`
  Allocates the pre-activation tensor needed by SWIGLU. For this test, `z.shape == (65536, 512)`.
- `y1 = torch.empty(TK, I, ...)`
  Allocates the post-activation grouped result. Here `y1.shape == (65536, 256)`.
- `_up_projection_forward(...)`
  Crosses into the custom op wrapper that converts tensors into CuTe DSL values and eventually launches the grouped GEMM.

Why `z` and `y1` both exist:

- `z` preserves the raw up-projection output for backward.
- `y1` stores the activated output that will feed the down-projection.

## Frame 4: `_up_projection_forward` Converts Tensors And Handles Compilation

Source: [../thirdparty/sonic-moe/sonicmoe/functional/forward.py#L54](../thirdparty/sonic-moe/sonicmoe/functional/forward.py#L54)

```python
mX = convert_torch_tensor_to_cute_tensor(x.detach(), (0, 1), 1, 16, 8, stream=stream_id)
mW1 = convert_torch_tensor_to_cute_tensor(w1.detach(), (2, 0, 1), 1, 16, 8, stream=stream_id)
mZ = convert_torch_tensor_to_cute_tensor(z, (0, 1), 1, 16, 8, stream=stream_id)
mY1 = convert_torch_tensor_to_cute_tensor(y1, (0, 1), 1, 16, 8, stream=stream_id)
mE_offset = convert_torch_tensor_to_cute_tensor(expert_frequency_offset, (0,), 0, 4, 1, stream=stream_id)
mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1, stream=stream_id)
# The wrapper converts Torch tensors into CuTe runtime tensor descriptors with
# explicit stride order, alignment, and address-space assumptions.

current_stream = cuda.CUstream(stream_id)

compile_w1_key = (E, H, I, (b1 is None), x.dtype, activation_type, is_inference_mode_enabled)
if compile_w1_key not in _up_projection_forward.compile_cache:
    w1_module = HopperWgmma_MoE_Up_proj_Fwd(
        E, H, I, activation_type=ActivationType(activation_type), inference_mode=is_inference_mode_enabled
    )
    tensormaps = [w1_module.module.generate_tensormap(None, None, None) for _ in range(2)]
    _up_projection_forward.compile_cache[compile_w1_key] = cute.compile(
        w1_module,
        mX,
        mW1,
        mZ,
        mY1,
        mB1,
        mE_offset,
        mX_gather,
        tensormaps[0],
        tensormaps[1],
        mE_permute_order,
        current_stream,
    )
    _up_projection_forward.compile_cache[TENSORMAP] = tensormaps

w1_tensormaps = _up_projection_forward.compile_cache[TENSORMAP]
_up_projection_forward.compile_cache[compile_w1_key](
    mX,
    mW1,
    mZ,
    mY1,
    mB1,
    mE_offset,
    mX_gather,
    w1_tensormaps[0],
    w1_tensormaps[1],
    mE_permute_order,
    current_stream,
)
```

Important details:

- The compile key for this test is effectively:
  `(128, 768, 256, True, torch.bfloat16, "swiglu", False)`.
- On the cold path, this code instantiates the wrapper, allocates tensormap workspace, and JIT-compiles a CuTe DSL callable.
- On the warm path, it skips instantiation and compilation and jumps directly to the cached callable.

Why tensormaps are allocated here:

- The grouped GEMM does not use one static descriptor for the entire run.
- The token-group size changes across experts, so the kernel updates TMA descriptors per scheduled work tile.
- `generate_tensormap()` preallocates 128-byte descriptor slots that the runtime can mutate.

## Frame 5: `HopperWgmma_MoE_Up_proj_Fwd` Chooses The Kernel Configuration

Source: [../thirdparty/sonic-moe/sonicmoe/functional/moe_config.py#L39](../thirdparty/sonic-moe/sonicmoe/functional/moe_config.py#L39)

For this test:

- activation is GLU (`SWIGLU`)
- `I=256 >= 128`
- `inference_mode=False`

So the wrapper chooses:

```python
up_config = HopperGEMMConfig(
    tile_shape_mnk=(128, 256, 64),
    cluster_shape_mnk=(2, 1),
    epi_tile_size=32,
    is_pingpong=False,
    initial_d_epi_stage=2,
    raster_order=RasterOrderOption.AlongM,
)
```

and then instantiates:

```python
self.module = HopperWgmma_MoE_kernel(
    E,
    cutlass.Float32,
    up_config.tile_shape_mnk,
    (*up_config.cluster_shape_mnk, 1),
    pingpong=up_config.is_pingpong,
    is_persistent=True,
    compute_swiglu=compute_swiglu,
    ...
    is_A_gather=True,
    epi_tile_size=up_config.epi_tile_size,
    initial_d_epi_stage=up_config.initial_d_epi_stage,
    inference_mode=inference_mode,
)
```

This is where the high-level algorithmic choices become kernel traits:

- `is_A_gather=True`
  means token rows are fetched through a specialized gather path instead of regular TMA.
- `compute_swiglu=True`
  means the epilogue must produce both the normal output tile and a gated activation tile.
- `is_persistent=True`
  means a CTA stays alive and asks the scheduler for more expert groups instead of exiting after one tile.
- `pingpong=False`
  means the chosen shape is large enough that a conventional staged pipeline is preferred over alternating warp groups on shared buffers.

### Why `pingpong` exists, even though this test does not use it

Small or awkward tile shapes can underutilize WGMMA pipelines if one warp-group stalls waiting on memory or epilogue turnover. Ping-pong mode creates two alternating warp groups that hand off shared-memory buffers through named barriers.

This test does not need it because:

- `I=256` is large enough to justify a `(128, 256, 64)` tile,
- the chosen tile already exposes enough work per CTA,
- and the non-ping-pong path can afford larger N without the extra synchronization complexity.

## Frame 6: `HopperWgmma_MoE_kernel.__init__` Encodes The Performance Policy

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L75](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L75)

This constructor is where the kernel’s performance model is encoded. The most important lines are below.

```python
self.is_glu = is_glu or (compute_swiglu or compute_geglu or compute_reglu)
self.is_normal_act = is_normal_act or (compute_gelu or compute_relu_sq or compute_relu or compute_silu)

self.need_adhoc_epilogue_store = self.is_glu or self.is_normal_act or compute_dz_and_partial_ds_and_y1s
self.need_epilogue_load = compute_dz_and_partial_ds_and_y1s
# GLU means the epilogue is not just "convert accumulators and store D".
# It must also produce the activation output Y.

if not self.pingpong:
    if tile_M not in [64, 128, 192, 256, 320]:
        raise ValueError(...)
    ...
if not self.tile_shape_mnk[2] % 16 == 0:
    raise ValueError("CTA tile shape K must be divisible by 16")
# The tile validation here is not cosmetic.
# These legal shapes are aligned with what the Hopper WGMMA/TMA pipeline can sustain cleanly.

if not self.pingpong:
    if tile_M == 320:
        atom_layout_m, atom_layout_n = 1, 2
    elif tile_M == 192:
        if tile_N <= 128:
            atom_layout_m, atom_layout_n = 3, 1
        else:
            atom_layout_m, atom_layout_n = 1, 2
    else:
        atom_layout_m = tile_shape_mnk[0] // 64 if tile_shape_mnk[0] < 256 else 2
        atom_layout_n = 1
...
self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)
```

For this test:

- `tile_shape_mnk = (128, 256, 64)`
- `pingpong = False`
- so `atom_layout_m = 128 // 64 = 2`
- `atom_layout_n = 1`
- `atom_layout_mnk = (2, 1, 1)`

Interpretation:

- the CTA is split into two WGMMA “atoms” along M,
- each atom covers a 64-row subproblem,
- the kernel avoids splitting N here because 256 columns are still manageable with one atom in N.

This is one of the key tensor-core tuning heuristics. The code is trying to keep the warpgroup MMA tiles aligned to Hopper-friendly 64-row chunks while avoiding a shape that would make the epilogue awkward or fragment registers too badly.

More performance-critical lines:

```python
self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
self.num_threads_per_warp_group = 128
self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
self.tma_warp_id = self.mma_warp_groups * 4
self.universal_copy_bits = 128
```

For this test:

- `mma_warp_groups = 2`
- `threads_per_cta = (2 + 1) * 128 = 384`
- warp IDs `0..7` are the two WGMMA warp groups
- warp IDs `8..11` are the producer/load side

That split is a major IPC optimization:

- some warps spend their registers on math,
- some spend their time on loading and scheduler duties,
- and `setmaxregister_increase/decrease` later gives those two roles different register budgets.

The up-projection gather-specific choice appears here:

```python
self.num_load_A_threads = (
    min(self.tile_M * self.tile_K // 8, self.threads_per_cta - self.tma_warp_id * cute.arch.WARP_SIZE)
    if is_A_gather
    else 0
)
```

For this test:

- `tile_M * tile_K // 8 = 128 * 64 // 8 = 1024`
- available producer threads after the MMA warp groups = `384 - 8*32 = 128`
- so `num_load_A_threads = 128`

Meaning:

- four warps are dedicated to feeding irregular `A` gathers.
- this is the bandwidth answer to MoE’s irregularity: use enough threads to recover coalescing and throughput even though the source rows are not contiguous.

Register budgeting:

```python
regs_per_thread = math.prod(self.tile_shape_mnk[:2]) // self.num_mma_threads
heavy_register_pressure = regs_per_thread >= 208
...
if self.mma_warp_groups == 3:
    self.num_regs_load, self.num_regs_mma = 56, 152
else:
    self.num_regs_load, self.num_regs_mma = (56, 224)
```

The point is not an exact occupancy target. The point is to bias registers toward the WGMMA consumer warps without starving the producer side. This is classic Hopper tuning:

- more registers for MMA helps accumulator-heavy code and epilogue staging,
- fewer registers for producer warps helps CTA residency and keeps the load side light,
- the split depends on whether `A` is gathered because gather warps need more per-thread address state.

## Frame 7: `_setup_attributes` Derives Epilogue Tiling And Stage Counts

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L296](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L296)

```python
self.d_epi_tile = self._sm90_compute_tile_shape_or_override(
    self.tile_shape_mnk,
    self.atom_layout_mnk,
    self.d_dtype,
)
self.c_epi_tile = self.d_epi_tile
if const_expr(self.compute_dz_and_partial_ds_and_y1s):
    self.y_epi_tile = self.d_epi_tile
elif const_expr(self.is_glu):
    self.y_epi_tile = (self.d_epi_tile[0], self.d_epi_tile[1] // 2)
```

For this test:

- `_sm90_compute_tile_shape_or_override()` sees `tile_M % 128 == 0` and `atom_layout_mnk[0] > 1`
- so it returns:
  `d_epi_tile = (gcd(128, 128), gcd(32, 256)) = (128, 32)`
- because this is GLU, `y_epi_tile = (128, 16)`

Why that matters:

- the accumulator tile is large enough to keep WGMMA busy,
- but the epilogue store tile is deliberately narrower,
- so the kernel can pipeline stores, activation work, and possible auxiliary outputs without exploding shared-memory usage.

This is one of the main store-latency hiding tricks in the file:

- compute in a bigger tile,
- drain the epilogue in subtiles that fit shared memory and TMA store cadence.

Stage budgeting appears in `_compute_stages()`:

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2641](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2641)

```python
remaining_bytes = (
    (smem_capacity - occupancy * 1024) // occupancy
    - mbar_helpers_bytes
    - epi_bytes
    - self.prefetch_token_idx_size * 4
    - (self.tile_shape_mnk[1] * (self.bias_dtype.width // 8) if self.use_bias else 0)
    - 1024
)
ab_stage = remaining_bytes // ab_bytes_per_stage
```

This is the occupancy/throughput balancing act in one formula:

- reserve some shared memory for barriers and helpers,
- reserve some for epilogue staging,
- reserve some for prefetched indices and bias if needed,
- then spend the remainder on A/B mainloop stages.

Why it matters:

- too few `ab_stage` buffers and the MMA side stalls waiting on memory,
- too many `ab_stage` buffers and the CTA burns so much shared memory that occupancy collapses,
- too many epilogue stages and stores dominate shared memory,
- too few epilogue stages and the store side cannot hide TMA/store latency.

This kernel explicitly budgets those tradeoffs instead of relying on a fixed stage count.

## Frame 8: `HopperWgmma_MoE_kernel.__call__` Builds The Launch

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L751](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L751)

This method prepares every object the CuTe DSL kernel needs before launch.

### 8A. It records dtypes and layouts

```python
self.a_dtype = mA.element_type
self.b_dtype = mB.element_type
self.c_dtype = mC.element_type if mC is not None else None
self.d_dtype = mD.element_type
self.s_dtype = cutlass.Float32

self.a_layout = utils.LayoutEnum.from_tensor(mA)
self.b_layout = utils.LayoutEnum.from_tensor(mB)
self.d_layout = utils.LayoutEnum.from_tensor(mD)
```

This is where the DSL runtime becomes shape/layout aware. The later shared-memory layout selection, TMA atom construction, and register-store paths all depend on these layout enums.

### 8B. It builds the tiled WGMMA object

```python
tiled_mma = sm90_utils.make_trivial_tiled_mma(
    self.a_dtype,
    self.b_dtype,
    self.a_layout.sm90_mma_major_mode(),
    self.b_layout.sm90_mma_major_mode(),
    self.acc_dtype,
    self.atom_layout_mnk,
    tiler_mn=(64, self.tile_shape_mnk[1] // self.atom_layout_mnk[1]),
)
```

For this test:

- the WGMMA tile basis is `64 x 256` per atom,
- and there are `2` atoms along M,
- so the CTA accumulates a `128 x 256` output tile.

This is the most direct tensor-core utilization decision in the code:

- make the atom shape match Hopper-friendly WGMMA fragments,
- then tile across M/N with `atom_layout_mnk`.

### 8C. It decides how A and B will be loaded

```python
if const_expr(self.is_A_gather):
    A_tiled_copy = self._make_tiled_copy_2D(...)
    tma_atom_a = tma_tensor_a = None
else:
    A_tiled_copy = None
    tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(...)

tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(...)
```

This is the core MoE-specific bandwidth split:

- `A` is irregular and loaded through `gatherA` with 128-bit copies,
- `B` is regular and loaded through TMA.

Why this is sensible:

- trying to TMA-load `A` would be poor for grouped MoE because routed token rows are no longer contiguous,
- trying to gather-load `B` would waste the regularity of expert weights.

### 8D. It creates the tile scheduler and shared storage

```python
tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
grid = TileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

@cute.struct
class SharedStorage:
    mainloop_pipeline_array_ptr: ...
    tensormap_buffer: ...
    sD: ...
    ...
    sA: ...
    sB: ...
```

This is where the persistent launch becomes concrete:

- `grid` is sized from the scheduler, not from a plain output-tile count,
- `SharedStorage` makes the staged memory contract explicit,
- the kernel launch later just points at this structure and the DSL compiler lays it out in shared memory.

### 8E. It finally launches the CuTe kernel

```python
self.kernel(
    ... many structured arguments ...
).launch(
    grid=grid,
    block=[self.threads_per_cta, 1, 1],
    cluster=self.cluster_shape_mnk,
    smem=allocated_smem_size,
    stream=stream,
    min_blocks_per_mp=1,
)
```

At this point:

- the Python trace ends,
- the generated CuTe/CUTLASS kernel takes over,
- and the remaining work is happening in the JIT-generated Hopper kernel body.

## Frame 9: Inside `kernel()` - Startup And Pipeline Construction

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1405](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1405)

```python
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

if warp_idx == self.tma_warp_id:
    if const_expr(not self.is_A_gather):
        cpasync.prefetch_descriptor(tma_atom_a)
    cpasync.prefetch_descriptor(tma_atom_b)
    ...
```

Why only the TMA warp does this:

- descriptor prefetch is a control task, not a math task,
- the kernel assigns it to the producer side to keep consumer warp groups focused on WGMMA.

Then the mainloop pipeline is created:

```python
if const_expr(self.is_A_gather):
    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1 + self.num_load_A_threads
    )
    mcast_size = self.num_mcast_ctas_b
    pipeline_class = PipelineTmaCpAsync
else:
    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
    pipeline_class = pipeline.PipelineTmaAsync
...
mainloop_pipeline = pipeline_class.create(...)
```

Interpretation:

- in the `gatherA` case, the producer side is hybrid:
  cp.async-style gathered `A` + TMA-loaded `B`.
- in the regular case, both inputs are TMA-driven.

This is the first big latency-hiding pattern:

- producer warps fill staged shared-memory buffers,
- consumer warp groups drain them into WGMMA,
- both sides advance pipeline states independently.

## Frame 10: Tensormap Workspace And Runtime Descriptor Updates

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1547](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1547)

```python
if cutlass.const_expr(self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM):
    tensormap_smem_ptr = shared_storage.tensormap_buffer.data_ptr()
    tensormap_smem_ptr = self.align_tensormap_smem_ptr(tensormap_smem_ptr)
    ...
```

and later:

```python
def update_tma_desc_ptr(...):
    if const_expr(self.compute_weight_gradient):
        tensor_shape = (mTensor.shape[0], token_group_size)
        start_ptr = (mTensor.iterator + token_start * mTensor.stride[1]).toint()
    else:
        tensor_shape = (token_group_size, mTensor.shape[1])
        start_ptr = (mTensor.iterator + token_start * mTensor.stride[0]).toint()

    tensor_gmem_ptr = cute.make_ptr(...)
    real_tensor = cute.make_tensor(...)
    tensormap_manager.update_tensormap(...)
    tensormap_manager.fence_tensormap_update(tensormap_ptr)
```

This is critical for grouped MoE:

- each expert’s token block has a different logical `M`,
- so one static TMA descriptor cannot describe all work tiles,
- the kernel rewrites descriptor base pointers and shapes as it moves between token groups.

Why shared-memory tensormap update mode is attractive:

- update the descriptor in shared memory,
- publish it through the tensormap manager,
- then use the updated descriptor for the next TMA transaction,
- avoiding full host-side descriptor rebuilds or a huge precomputed descriptor table.

## Frame 11: Producer Side - `gatherA` + TMA-Load `B`

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1718](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1718)

When `is_A_gather=True`, the producer path sets up:

```python
A_g2s_thr_copy = A_tiled_copy.get_slice(tidx - self.tma_warp_id * cute.arch.WARP_SIZE)
gA_mk = cute.local_tile(mA_mkl, (self.tile_M, self.tile_K), (0, None))
tAgA = A_g2s_thr_copy.partition_S(gA_mk)
```

Then, inside the tile loop:

```python
tmAIdx = self.prefetch_gather_idx_for_A_when_vary_M(...)
...
self.load_A_gather(
    mA_mkl,
    tmAIdx,
    sAIdx_prefetch,
    M_offset,
    tAsA[None, None, None, mainloop_producer_state.index],
    tApA,
    A_g2s_thr_copy,
    K_offset,
    token_group_size,
    A_thr_copy_elems,
)
```

### Why `gatherA` is the main MoE bandwidth specialization

The up-projection’s left operand is token data. After routing, the rows needed by expert `e` are scattered across the original token matrix. The kernel therefore:

1. linearizes expert work into grouped rows,
2. prefetches the gather indices,
3. loads the needed token rows with 128-bit copies into staged shared memory.

Relevant helper:

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L551](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L551)

```python
def load_A_gather(...):
    ...
    threads_per_stride_1_dim = const_expr(stride_1_tile // copy_elems_per_thr_load)
    num_other_dim_per_load = const_expr(self.num_load_A_threads // threads_per_stride_1_dim)
    ...
    tPrAptr = self.elem_pointer(mA, (MIdx, KIdx)).align(
        self.universal_copy_bits // copy_elems_per_thr_load
    )
    mA_cur_copy = cute.make_tensor(tPrAptr, ((copy_elems_per_thr_load, 1), 1))
    cute.copy(A_g2s_thr_copy, mA_cur_copy, tAsA[...], pred=tApA[...])
```

What this is optimizing:

- `128`-bit vectorized loads instead of scalar gathers,
- enough producer threads to cover the irregular access pattern,
- predicate masks to avoid out-of-bounds traffic at ragged group boundaries,
- optional shared-memory prefetch of gather indices for the weight-gradient case.

This is the kernel’s answer to “MoE routing wrecks the nice memory pattern of dense GEMM”.

At the same time `B` remains regular:

```python
cute.copy(
    tma_atom_b,
    tBgB_nkl[None, k_tile],
    tBsB[None, mainloop_producer_state.index],
    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
    mcast_mask=b_mcast_mask,
    tma_desc_ptr=b_tma_desc_ptr,
)
```

So the producer side is asymmetric on purpose:

- `A`: gather/cp.async path
- `B`: TMA path with optional multicast

## Frame 12: Consumer Side - WGMMA Mainloop

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1949](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1949)

```python
if warp_idx < self.tma_warp_id:
    cute.arch.setmaxregister_increase(self.num_regs_mma)
    cute.arch.setmaxregister_increase(self.num_regs_mma)
    ...
    thr_mma = tiled_mma.get_slice(...)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
```

This is the compute-heavy half:

- increase the register budget for MMA warps,
- partition the staged `A` and `B` tiles into WGMMA fragments,
- accumulate into FP32 registers.

Mainloop overlap pattern:

```python
peek_ab_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)
tiled_mma.set(warpgroup.Field.ACCUMULATE, False)

for k_tile in ...:
    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state, peek_ab_full_status)
    warpgroup.fence()
    for k_blk_idx in ...:
        cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
    warpgroup.commit_group()
    warpgroup.wait_group(k_pipe_mmas)
    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
```

This is the second major latency-hiding pattern:

- wait only when the next stage is actually needed,
- issue WGMMA work in groups,
- overlap the next producer fill with the current consumer compute,
- release shared-memory stages as soon as the math side is done with them.

Why this helps IPC:

- the producer side keeps the shared-memory stages warm,
- the consumer side tries to keep at least one WGMMA group in flight,
- register budgeting prevents the producer side from stealing too much state from the math side.

## Frame 13: Epilogue Tiling, Activation, And Store Latency Hiding

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2162](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2162)

The epilogue is where this kernel does a lot more than a plain GEMM store.

It first retile-copies accumulator registers into epilogue fragments:

```python
copy_atom_D_r2s = sm90_utils.sm90_get_smem_store_op(...)
tiled_copy_D_r2s = cute.make_tiled_copy_S(copy_atom_D_r2s, tiled_copy_D_atom)
tRS_sD = tiled_copy_D_r2s.get_slice(tidx).partition_D(sD)
tRS_rD = cute.make_rmem_tensor(tRS_rD_layout, self.acc_dtype)
```

If epilogue load is needed, it creates a parallel `C` load pipeline. If activation output `Y` is needed, it sets up a second retiled store path for `Y`.

Then the real epilogue loop runs over `epi_tile_num` subtiles:

```python
for epi_idx in cutlass.range_constexpr(epi_tile_num):
    for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
        tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
    ...
    if const_expr((self.is_glu or self.is_normal_act) and not self.compute_dz_and_partial_ds_and_y1s):
        tRS_rY = cute.make_rmem_tensor_like(tRS_sY[None, None, None, 0], self.y_dtype)
        self.compute_activation(tRS_rD, tRS_rY)
    ...
    cute.copy(tiled_copy_D_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
    cute.copy(tiled_copy_Y_r2s, tRS_rY, tRS_sY[(None, None, None, epi_buffer)])
```

For this test, this means:

- the `128 x 256` accumulator tile is drained as `128 x 32` `D` subtiles,
- and `128 x 16` `Y` subtiles because SWIGLU halves the N-width of the activated output.

### Why epilogue tiling matters

This is one of the kernel’s main latency-hiding strategies.

- WGMMA likes large tiles because it amortizes instruction overhead and increases arithmetic intensity.
- TMA store and activation handling like smaller tiles because they reduce shared-memory footprint and let stores be pipelined.

So the kernel chooses:

- large mainloop tile for tensor-core efficiency,
- narrower epilogue tiles for store/manageability efficiency.

### Why GLU makes the epilogue heavier

`compute_activation()` for GLU:

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1272](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1272)

```python
if const_expr(self.is_glu):
    ...
    for i in cutlass.range_constexpr(cute.size(tRS_rD) // 2):
        tRS_rY[i] = (act_func(tRS_rD[2 * i]) * tRS_rD[2 * i + 1]).to(self.y_dtype)
```

The epilogue is not only converting accumulators. It is also:

- computing the activation,
- forming the gated product,
- and storing the narrower activation result separately.

That is why the kernel tracks both `D` and `Y` epilogue storage.

### How store latency is hidden

After the register-to-shared copies:

```python
cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
epilogue_barrier.arrive_and_wait()
if is_tma_warp:
    cute.copy(tma_atom_d, bSG_sD[None, epi_buffer], bSG_gD[None, gmem_coord], tma_desc_ptr=d_tma_desc_ptr)
    if const_expr(self.need_adhoc_epilogue_store):
        cute.copy(tma_atom_y, bSG_sY[None, epi_buffer], bSG_gY[None, gmem_coord], tma_desc_ptr=y_tma_desc_ptr)
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(...)
```

That sequence does three things:

1. make shared-memory writes visible to the async TMA path,
2. have only the designated TMA warp issue the stores,
3. pipeline multiple epilogue subtile stores using bulk async groups.

Again the design is asymmetric on purpose:

- many threads cooperate to prepare epilogue data in registers/shared memory,
- few threads issue the actual global-memory store transactions.

## Frame 14: `generate_tensormap()` Explains Why Compilation Allocates Extra Workspace

Source: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2587](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2587)

```python
if self.is_persistent:
    total_ctas = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
if self.pingpong:
    total_ctas *= 2
tensormaps_torch = torch.empty(total_ctas, 128 // 8, dtype=torch.int64, device="cuda")
```

This preallocates a `128`-byte descriptor slot per resident work owner:

- one per CTA in the persistent case,
- doubled for ping-pong because each alternating warp-group may need its own descriptor workspace.

This is a concrete manifestation of the runtime descriptor-update strategy described earlier.

## Appendix: What The CuTe DSL Launch Lowers Into

This `HopperWgmma_MoE_kernel` path is not a Triton kernel. It is a CuTe DSL JIT path that lowers into generated CuTe/CUTLASS-style CUDA code.

The stack is:

```text
Python wrapper
  -> @cute.jit wrapper objects
    -> @cute.kernel DSL body
      -> CuTe/CUTLASS MLIR lowering
        -> cuda.launch_ex
          -> JIT-loaded cubin + generated host launcher
```

Useful source points:

- `cute.compile(...)` in the SonicMoE wrapper: [../thirdparty/sonic-moe/sonicmoe/functional/forward.py#L97](../thirdparty/sonic-moe/sonicmoe/functional/forward.py#L97)
- `HopperWgmma_MoE_kernel.__call__` issuing `self.kernel(...).launch(...)`: [../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1038](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1038)
- `@cute.kernel` launch helper path in the installed DSL: [../.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/cutlass.py#L1101](../.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/cutlass.py#L1101)
- `cuda_launch_func(...)` lowering to MLIR CUDA launch ops: [../.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/cutlass.py#L605](../.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/cutlass.py#L605)
- JIT executor loading the generated binary and running the host launcher: [../.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py#L960](../.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py#L960)

That matters because the performance model seen in `grouped_gemm.py` is the real kernel structure, not a high-level placeholder that Triton later rewrites.

## Performance Patterns To Notice

### 1. MoE-specific memory-bandwidth optimization: `gatherA`

- The token matrix is irregular after routing, so the kernel does not force it through a normal TMA path.
- `x_gather_idx` linearizes the routed rows.
- `load_A_gather()` uses `128`-bit loads, many producer threads, and predicates to recover bandwidth from that irregular access pattern.
- In the weight-gradient case, the index stream itself is prefetched into shared memory so the address-generation cost does not serialize the hot path.

### 2. Hiding load/store latency: producer/consumer pipelines plus epilogue tiling

- Producer warps and consumer warp groups are separate.
- The mainloop pipeline overlaps `A`/`B` fill with WGMMA consumption.
- The epilogue pipeline can overlap `C` loads, activation work, and `D`/`Y` stores.
- Narrow epilogue tiles (`128x32` for `D`, `128x16` for `Y` here) reduce store-side working set and allow more store buffering than a monolithic epilogue would.

### 3. Tensor-core tuning: shape legality plus atom layout heuristics

- `K` must be divisible by `16`, and `M/N` come from a curated legal set.
- `atom_layout_mnk` is chosen so CTA tiles decompose into Hopper-friendly 64-row WGMMA units.
- For this test, `(128, 256, 64)` with `(2,1,1)` means two 64-row WGMMA atoms along M.
- Some odd cases, like `tile_M=192` or `320`, intentionally split along `N` because splitting along `M` would create poor residual shapes or awkward epilogues.

### 4. Occupancy vs throughput: stage count and register budgeting

- `_compute_stages()` explicitly spends shared memory between A/B stages, epilogue stages, helpers, and optional index/bias storage.
- producer warps get smaller register budgets,
- consumer WGMMA warp groups get larger ones,
- and the kernel chooses persistent execution so CTAs can amortize setup costs while staying close to an occupancy floor that still keeps the SM busy.

### 5. Ping-pong mode as a throughput recovery tool

- Not used in this exact test path.
- When enabled, it lets two warp groups alternate through shared-memory buffers via named barriers.
- That helps smaller or more synchronization-heavy shapes keep math issuing while the other group handles buffer turnover.
- The code doubles descriptor workspace and explicitly advances pipeline state for the “other” warp group, which shows that ping-pong is a real pipeline strategy, not a minor tweak.

## Short “Before / After” Summary

Before line 91:

- tokens are contiguous,
- routing has not been applied,
- no grouped GEMM state exists,
- no CuTe kernel has been compiled for this configuration in the cold path.

After the up-projection grouped GEMM launch is assembled:

- tokens have been logically regrouped by expert,
- `A` is designated for gather loading,
- `B` is designated for TMA loading,
- tensormap workspace exists for dynamic per-group descriptor updates,
- the persistent Hopper kernel has a producer/consumer pipeline and epilogue tiling strategy tuned for this exact shape.

## Process Note

- I followed the `trace-code` workflow and traced the path top-down from the test entrypoint through the MoE wrapper, routing helper, autograd wrapper, custom-op wrapper, kernel config wrapper, and CuTe DSL launch code.
- I delegated two small explorer tasks: one to map the Python call chain from `moe_minimal.py:91` into `HopperWgmma_MoE_kernel`, and one to investigate the lower-level grouped-GEMM lowering path through the installed CuTe DSL runtime. I used both results and cross-checked the kernel-level details locally.
- Tools used: `rg`, `nl`, `sed`, and local source inspection only.
