# Flash Attention FWD (SM90) Trace Plan

## Objective
Produce a deep, frame-by-frame trace of the CuTeDSL SM90 forward path rooted at:
- [interface.py:_flash_attn_fwd](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)

Output document:
- `codex/flash_attn_fwd.md`

## Plan
1. Trace host-side launcher flow in `interface.py`:
   - input normalization, compile key construction, compile cache lookup
   - `FlashAttentionForwardSm90` instantiation and `cute.compile` boundary
2. Trace SM90 kernel object setup in `flash_fwd.py`:
   - kernel config, shared-memory structures, pipeline objects, scheduler selection
   - TMA atom/tensor setup and launch grid/block configuration
3. Trace kernel execution frame-by-frame:
   - producer path (`load`) and consumer path (`mma`)
   - Q/K/V movement, pipeline/barrier synchronization, overlap strategy
4. Trace helper stack used by SM90 forward:
   - `Softmax`, `AttentionMask`, `SeqlenInfoQK`, `BlockInfo`, `TileScheduler`
   - `PackGQA`, score-mod plumbing, pipeline wrappers
5. Add deep performance commentary:
   - data layouts, thread/value mapping, occupancy/register tradeoffs
   - memory bandwidth optimizations, overlap patterns, predicate strategy

## Constraints
- No source code edits.
- Documentation-only deliverable with VSCode-clickable links.
