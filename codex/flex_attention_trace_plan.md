# FlexAttention Trace Plan

## Task
Produce a frame-by-frame implementation trace for the benchmark call sites in `thirdparty/attention-gym/examples/flex_flash_attention.py` (reference, Triton, and CuTeDSL/FLASH backends), including both forward and backward paths.

## Constraints
- Documentation-only task (no source code changes).
- Output file must be `codex/flex_attention_trace.md`.
- Include VSCode-clickable source links.
- Do not skip call frames between Python API, HOP/autograd, Inductor lowering, and backend kernel generation/launch.

## Deliverables
1. `codex/flex_attention_trace.md`
   - End-to-end call path for each backend:
     - Reference (`flex_attention` eager path)
     - Triton (`kernel_options={"BACKEND":"TRITON"}`)
     - CuTeDSL FLASH (`kernel_options={"BACKEND":"FLASH"}`)
   - Forward + backward for each backend.
   - Annotated inline code snippets with state before/after key frames.
   - Explicit explanation of `score_mod` and `block_mask` integration.
   - Key-functions index table.

## Source Coverage
- Entry benchmark:
  - `thirdparty/attention-gym/examples/flex_flash_attention.py`
- User-facing API + mask construction:
  - `thirdparty/pytorch/torch/nn/attention/flex_attention.py`
- HOP + autograd + trace capture:
  - `thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py`
- Inductor lowering and backend selection:
  - `thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py`
  - `thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py`
  - `thirdparty/pytorch/torch/_inductor/kernel/flex/common.py`
  - `thirdparty/pytorch/torch/_inductor/select_algorithm.py`
  - `thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py`
- Triton template internals:
  - `thirdparty/pytorch/torch/_inductor/kernel/flex/templates/*.jinja`
- CuTeDSL/flash-attn integration:
  - `thirdparty/flash-attention/flash_attn/cute/interface.py`
  - `thirdparty/flash-attention/flash_attn/cute/flash_fwd.py`
  - `thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm100.py`
  - `thirdparty/flash-attention/flash_attn/cute/softmax.py`
  - `thirdparty/flash-attention/flash_attn/cute/mask.py`
  - `thirdparty/flash-attention/flash_attn/cute/utils.py`

## Outline
1. Big-picture architecture map.
2. Frame-by-frame trace for benchmark call site.
3. Reference backend forward/backward trace.
4. Triton backend forward/backward trace.
5. FLASH/CuTeDSL backend forward/backward trace.
6. Dedicated section: dynamic stitching of `score_mod` and `block_mask`.
7. Consolidated full call paths (forward + backward per backend).
8. Key-functions index.
