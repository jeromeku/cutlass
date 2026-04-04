# Standalone `gather_A` Extraction Plan

## Goal

Extract the `load_A_gather` device logic from `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py` into a standalone CuTe kernel that can be studied in isolation, keep the implementation localized under `codex/`, and add a torch test that validates the standalone kernel against a PyTorch reference.

## Proposed Scope

1. Add a standalone kernel module under `codex/`.
   - Create a small wrapper class dedicated to gather-A.
   - Keep the extracted device logic structurally close to the original implementation.
   - Lift only the helper pieces that `load_A_gather` directly depends on:
     - `elem_pointer`
     - `min_i32`
     - `prefetch_gather_idx_for_A_when_vary_M`
     - `load_A_gather`
     - the copy-layout helper needed to build the gather copy pattern

2. Match the original outer decomposition instead of hardcoding one CTA.
   - Tile over `M` with `grid.x = ceil_div(token_group_size, tile_M)`.
   - Tile over `K` with `grid.y = K_extent / tile_K`.
   - Reconstruct `M_offset` and `K_offset` from CTA coordinates inside the kernel.
   - Write the gathered tile into a global output tensor so the result remains easy to inspect.

3. Add a thin Python wrapper and local runtime helpers under `codex/`.
   - Use the existing CuTe conversion and `cute.compile(...)` pattern.
   - Avoid importing the full `sonicmoe` package so the standalone study path stays isolated.

4. Add a targeted torch test under `codex/`.
   - Compare the standalone kernel output to a pure PyTorch reference gather.
   - Cover:
     - a normal in-bounds tiled case
     - a ragged tail case where the last `M` tile is partial
     - a nonzero `K_start`

5. Add explanatory comments and rerun a focused GPU test pass.

## Design Notes

- To preserve the original implementation closely, the kernel keeps the same thread-to-copy decomposition and the same predicated vector copy behavior.
- For testability, the standalone kernel writes into a global output tensor instead of shared memory only. That output represents the same gathered values the original kernel would stage into `sA`, but across every CTA tile in the requested region.
- The work remains fully localized under `codex/`.
- I do not currently expect any extra libraries or tools beyond the existing repo `.venv`.

## Chunks

1. Completed: extract the standalone kernel, wrapper, and local runtime helper under `codex/`.
2. Completed: refactor the launcher so the kernel tiles across multiple CTAs in `M` and `K`.
3. Completed: add the torch reference test and validate with a focused GPU pytest run.
4. Completed: add inline comments/docstrings explaining the purpose of each helper and kernel entry point.
