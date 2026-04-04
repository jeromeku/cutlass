# Hopper MoE Kernel Trace Plan

## Goal

Produce a deep trace starting at `thirdparty/sonic-moe/tests/moe_minimal.py:91` and following the execution path into `sonicmoe.functional.grouped_gemm.HopperWgmma_MoE_kernel`, with heavy inline commentary and exact source links.

## Chunked Plan

1. Entrypoint frame
   Trace `test_moe()` around line 91 to establish runtime state: tensor shapes, autocast context, backend selection, and the exact call expression that enters the MoE module.

2. MoE forward path
   Trace `sonicmoe.moe.MoE.forward()` through the `KernelBackendMoE.sonicmoe` branch into `moe_TC_softmax_topk_layer()`, showing where routing, expert metadata, and grouped token layout are constructed.

3. Autograd wrapper frames
   Trace `_UpProjection.forward()` and the call to `_up_projection_forward()`, explaining why the code allocates `z`/`y1`, how the expert offsets and gather/scatter indices are threaded through, and where the grouped GEMM path diverges from the QuACK fallback.

4. Forward custom-op and compile cache
   Trace `sonicmoe.functional.forward._up_projection_forward()` line by line: torch-to-CuTe tensor conversion, stream plumbing, compile-cache key construction, `HopperWgmma_MoE_Up_proj_Fwd` instantiation, tensormap allocation, `cute.compile(...)`, and the cached callable invocation.

5. Kernel wrapper construction
   Trace `HopperWgmma_MoE_Up_proj_Fwd.__init__()` and `__call__()` from `sonicmoe.functional.moe_config`, focusing on how problem shape and activation mode choose a specific `HopperWgmma_MoE_kernel` configuration and how its arguments are adapted for the grouped GEMM launch.

6. Grouped GEMM launch preparation
   Trace `HopperWgmma_MoE_kernel.__init__()` and `HopperWgmma_MoE_kernel.__call__()` in `sonicmoe.functional.grouped_gemm`: tile validation, warp-group counts, TMA policy, attribute setup, tiled MMA/TMA object construction, tile scheduler setup, shared-storage layout, and the final `.launch(...)`.

7. Lower-level kernel notes
   Add focused notes on `generate_tensormap()`, `update_tma_desc_ptr()`, and the role of the generated CuTe DSL kernel body so the reader can connect the Python launch wrapper to the runtime TMA/tensormap updates without expanding every helper in the file.

## Deliverable

- Main trace doc: `codex/hopper_moe_kernel.md`
- Optional diagrams: sequence diagram, dataflow diagram, and module map if requested

## Open Choice

- Diagrams are optional. If requested, I will include ASCII sequence/dataflow/module diagrams in the final trace.
