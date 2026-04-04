# `load_A_gather` Extraction Investigation Plan

## Goal

Identify the best integration point for extracting `load_A_gather` from `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py` into a reusable standalone CuTe kernel/helper, without editing source files.

## Chunks

1. Trace `load_A_gather` and its immediate callers in `grouped_gemm.py`.
2. Map the helper's closed-over state (`self` fields, helper methods, barriers, layouts, copy objects).
3. Inspect the existing Sonic-MoE functional layout for a natural destination file or module boundary.
4. Summarize recommended integration point, candidate files, and tricky extraction dependencies.

## Notes

- No source changes are planned in this pass.
- The output should call out whether `load_A_gather` is better extracted alone or together with its gather-index prefetch helpers.
