# Flash Attention BWD Trace Plan

## Objective
Create a deep backward trace for CuTeDSL flash attention in:
- [interface.py:_flash_attn_bwd](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)

Output document:
- `codex/flash_attn_bwd.md`

## Scope
1. Launcher-level backward flow:
   - preprocess / main / postprocess compilation and dispatch
   - compile key construction, score/mask hook hashing and specialization
2. Main SM90 backward kernel (`flash_bwd_sm90.py`):
   - data structures, producer/consumer split, pipeline/barrier choreography
   - GEMM + pointwise sequence for dS/dP/dQ/dK/dV
3. Helper call stack:
   - score_mod + score_mod_bwd hooks
   - mask_mod application with `swap_AB` handling
   - block sparse helper paths
4. Performance analysis:
   - layout transforms, thread-value mapping, data movement, accumulation strategy

## Constraints
- Documentation-only changes.
- No source-code edits.
