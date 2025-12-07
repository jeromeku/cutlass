HopperWgmmaGemmPersistentKernel: Python → MLIR → CuTe C++
=========================================================

This is a frame-by-frame trace of `HopperWgmmaGemmPersistentKernel._setup_attributes` and `__call__` in `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py`, showing every method hop down to the MLIR ops/types emitted, and the corresponding CuTe C++ abstractions.

-----------------------------------------------------------------------
1) Decorator plumbing: how `__call__` becomes MLIR
-----------------------------------------------------------------------
- `__call__` is annotated with `@cute.jit` (`examples/python/CuTeDSL/hopper/dense_gemm_persistent.py:340`). The decorator is `BaseDSL.jit_runner` → `jit_wrapper` (`python/CuTeDSL/cutlass/base_dsl/dsl.py:464-520`).
- First invocation enters `BaseDSL._preprocess_and_execute` (`dsl.py:405-458`):
  - Lazily instantiates the DSL object (`CuTeDSL`) via `_get_dsl` (`dsl.py:404-423`).
  - Runs the AST preprocessor; `get_function_ptr` materializes a Python callable whose body builds MLIR (`dsl.py:432-458`).
- Host IR generation happens in `CutlassBaseDSL.generate_original_ir` (`python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:960-1032`):
  - Creates `func.func` entry with block args typed by `generate_mlir_function_types` (`dsl.py:748-780`).
  - Argument typing: `cute.Tensor` args go through `__get_mlir_types__` on the runtime tensor wrapper (`python/CuTeDSL/cutlass/cute/runtime.py:122-219`), producing a `memref<?x...>` MLIR type that corresponds to a CuTe/CUTLASS tensor descriptor (C++ `cute::Tensor` / CUTLASS `TensorRef`).
  - `max_active_clusters` is `Constexpr`, so it is removed from the MLIR signature.
  - `cuda.CUstream` is adapted by `StreamAdapter` to MLIR `gpu.async.token` (`python/CuTeDSL/cutlass/base_dsl/runtime/stream_adapter.py:25-43`); C++ analog is a CUDA stream handle passed to a `__global__` launch.
- When the kernel launch is encountered later in `__call__`, `KernelLauncher` builds a `cuda.kernel` (`KernelOp`) and `gpu.launch_func` (`cutlass.py:636-738`). Those MLIR ops lower to an NVVM `__global__` kernel, equivalent to writing a C++ kernel and invoking it with `<<<grid, block, smem, stream>>>`.

-----------------------------------------------------------------------
2) `_setup_attributes` trace (called inside JIT-traced `__call__`)
-----------------------------------------------------------------------
Entry: `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py:322-413`

Frame 1: Shape validation
- Plain Python guards on `self.tile_shape_mnk` (lines 350-354). Because `tile_shape_mnk` is set earlier and not an MLIR SSA value, no MLIR is emitted; the checks gate the rest of the trace.

Frame 2: Build the MMA atom and `TiledMma`
- Call `sm90_utils.make_trivial_tiled_mma` (`hopper_helpers.py:92-154`).
  - Chooses an SM90 WGMMA op class (e.g., `MmaF16BF16Op`, `MmaF8Op`, `MmaI8Op`).
  - Calls `cute.make_mma_atom` (`python/CuTeDSL/cutlass/cute/atom.py:492-507`):
    - `_make_trait` on the op emits the MLIR type for the MMA instruction (`_cute_ir.make_mma_atom`, op name `cute.make_mma_atom` in `_cute_ops_gen.py:2290+`), producing an SSA `Value` of type `!cute.mma_atom`.
    - C++ analog: constructs a `cute::SM90_*` instruction descriptor (e.g., `SM90_64x8x16_F16F16F16_SS`) in `include/cute/arch/mma_sm90_gmma.hpp` and wraps it in `cute::MMA_Atom` (`include/cute/atom/mma_atom.hpp`).
  - `cute.make_tiled_mma` (`atom.py:510-559`) emits MLIR `cute.make_tiled_mma` (`_cute_ops_gen.py:2327+`) returning `!cute.tiled_mma`. This is the tiled MMA object used in the kernel.
    - C++ analog: `cute::TiledMMA<...>` (`include/cute/atom/mma_atom.hpp`) built from the chosen SM90 atom and the provided `atom_layout_mnk`.
- `cute.size(self.tiled_mma.shape_mnk, mode=[2])` emits MLIR `cute.size` (`_cute_ops_gen.py:1467+`) to read the K extent of the MMA tile; C++ analog is `size<2>(tiled_mma.shape())`.
- `self.tile_shape_mnk` is updated in Python using that static K tile; no MLIR op is emitted for the tuple assignment.

Frame 3: CTA layout and multicast flags
- `self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))` → MLIR `cute.make_layout` (`_cute_ops_gen.py:2115+`) producing `!cute.layout`.
  - C++ analog: `cute::make_layout(make_shape(M, N, 1))` in `include/cute/layout.hpp`.
- `num_mcast_ctas_*` assignments are Python scalars; no MLIR emitted.

Frame 4: Epilogue tile shape
- `_sm90_compute_tile_shape_or_override` (`dense_gemm_persistent.py:1031-1051`) uses `cute.size` (MLIR `cute.size`) and Python `min`. The `size` calls emit MLIR; `min` is on Python integers, so no MLIR.
- Resulting `epi_tile` is a Python tuple captured for later use.

Frame 5: Stage counts (AB/epilogue)
- `_compute_stages` (`dense_gemm_persistent.py:1054-1099`) is pure Python arithmetic; because it only sees Python ints, no MLIR ops are created here. Outputs (`ab_stage`, `epi_stage`) are cached Python ints.

Frame 6: Shared-memory layouts
- `_make_smem_layouts` (`dense_gemm_persistent.py:1102-1215`) emits several MLIR ops:
  - `cute.make_layout` for base layouts.
  - `cute.compose` / `cute.composed_layout` ops when staging layouts (see `_cute_ops_gen.py` entries for `compose` and `make_composed_layout`).
  - `cute.round_up`, `cute.aligned_size`, `cute.cosize` → MLIR ops of the same names.
  - C++ analogs: `cute::make_layout`, `cute::compose`, `cute::cosize`, etc., from `include/cute/layout.hpp` and `include/cute/algorithm/copy.hpp`. The resulting MLIR `!cute.composed_layout` corresponds to `cute::ComposedLayout` C++ type.

-----------------------------------------------------------------------
3) `__call__` trace (host JIT function body)
-----------------------------------------------------------------------
Entry: `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py:340-524`

Frame A: Argument materialization
- The host `func.func` entry block already has SSA args for `(a, b, c, stream)` typed as described in §1.
- Type checks via `cutlass.const_expr` generate MLIR constants and comparisons in the `arith` dialect (`cutlass/base_dsl/ast_helpers.py` lowers `const_expr` into inline constants where possible). C++ analog would be `static_assert`/`if` on host before launching a kernel.

Frame B: Attribute setup
- Calls `_setup_attributes` → frames 1–6 above (MLIR emitted where noted).

Frame C: Build TMA load/store atoms and tensors
- `_make_tma_atoms_and_tensors` (`dense_gemm_persistent.py:1227-1254`):
  - Chooses an op class (`CopyBulkTensorTileG2SOp` or multicast variant) from `cute.nvgpu.cpasync`.
  - Calls `cute.nvgpu.cpasync.make_tiled_tma_atom` (see `python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:205-250`):
    - Emits MLIR `cute.nvgpu.cp_async.bulk_tensor_tile_(g2s|g2s_multicast)` ops that produce a `!cute.copy_atom` and a staged `!cute.tensor` view in shared memory.
    - C++ analog: `cute::make_tma_atom` for SM90 TMA (`include/cute/arch/cp_async.hpp`), producing a `Copy_Atom` and tensor descriptor usable with `cp.async.bulk.tensor`.
- `_make_tma_store_atoms_and_tensors` (`dense_gemm_persistent.py:1207-1224`) similarly emits a TMA store atom (`CopyBulkTensorTileS2GOp`) for C.

Frame D: Grid/tile scheduler
- `_compute_grid` (`dense_gemm_persistent.py:1306-1490`):
  - Builds `PersistentTileSchedulerParams` (class in `python/CuTeDSL/cutlass/utils/static_persistent_tile_scheduler.py:77-188`).
    - Its constructor calls `cute.make_layout`, `cute.ceil_div`, etc. → MLIR ops for layout math.
    - Returns a Python object implementing `DynamicExpression`; its `__extract_mlir_values__`/`__new_from_mlir_values__` expose MLIR SSA values (scheduler state). C++ analog: a struct of layout/shape ints used by the persistent scheduler helper in `cutlass/utils/static_persistent_tile_scheduler` (runtime analogue of CUTLASS scheduler utilities).
  - Computes `grid` as Python ints; no MLIR.

Frame E: Shared storage type
- The nested `SharedStorage` `@cute.struct` (`dense_gemm_persistent.py:476-506`) expands to MLIR `cute.struct` type/ops:
  - `cute.struct.MemRange` fields become `!cute.struct<memref>` members.
  - `cute.struct.Align` adds alignment attributes.
  - C++ analog: a `struct` placed in `__shared__` with aligned arrays; matches how CUTLASS declares shared storage blocks in templates.

Frame F: Kernel launch construction
- `self.kernel(...).launch(...)`:
  - `self.kernel` is a `@cute.kernel` JIT (device) function; the call creates a `KernelLauncher` (`cutlass.py:896-739`).
  - `KernelLauncher.launch` builds:
    - MLIR `cuda.kernel` (`cuda_dialect.KernelOp`) with arguments typed from the device-side signature (`CutlassBaseDSL._generate_jit_func_args_for_known_types` handles `CopyAtom`, `Tensor`, `TiledMma`, layouts, scheduler params).
    - Entry block uses `generate_execution_arguments` to reconstruct Python objects from block args via `__new_from_mlir_values__` (e.g., `CopyAtom.__new_from_mlir_values__` in `python/CuTeDSL/cutlass/cute/atom.py:86-89`).
    - Emits the kernel body MLIR by running `HopperWgmmaGemmPersistentKernel.kernel`, whose ops include `nvgpu.cp.async`/`nvgpu.tma`/`nvgpu.wgmma` etc. (not expanded here since the request was for `__call__`/`_setup_attributes`).
    - Emits `gpu.launch_func` to invoke the kernel with `grid/block/cluster/smem` (`cutlass.py:650-738`).
  - C++ analog: instantiating a `__global__` kernel template with parameters `(CopyAtom, Tensor descriptors, TiledMMA, layouts, scheduler params)` and launching it with `cudaLaunchKernel`.

-----------------------------------------------------------------------
4) MLIR op → CuTe C++ type map (used along the trace)
-----------------------------------------------------------------------
- `cute.make_layout` MLIR (`_cute_ops_gen.py:2115+`) → `cute::Layout` / `cute::make_layout` (`include/cute/layout.hpp`).
- `cute.make_tiled_mma` MLIR (`_cute_ops_gen.py:2327+`) → `cute::TiledMMA` wrapping an SM90 WGMMA atom (`include/cute/atom/mma_atom.hpp`, SM90 atoms in `include/cute/arch/mma_sm90_gmma.hpp` and traits in `include/cute/atom/mma_traits_sm90_gmma.hpp`).
- `cute.nvgpu.cp_async.bulk_tensor_tile_*` MLIR (emitted by `make_tiled_tma_atom`) → SM90 TMA descriptor + `Copy_Atom` (`include/cute/arch/cp_async.hpp`, `include/cute/atom/copy_atom.hpp`).
- `cute.struct` MLIR → C++ struct/tuple composition used for shared storage (mirrors CUTLASS shared storage structs; see `include/cutlass/epilogue/thread/` for analogous layouts and `include/cute/atom/copy_atom.hpp` for memranges).
- `gpu.launch_func`/`cuda.kernel` MLIR → C++ `__global__` kernel and CUDA launch (`cuda.h`), with argument lowering consistent with CUTLASS kernel entrypoints (`include/cutlass/device_kernel.h` patterns).

-----------------------------------------------------------------------
Key takeaways
-----------------------------------------------------------------------
- Every `cute.*` call inside `_setup_attributes` and `__call__` is a `@dsl_user_op`; during tracing it emits a same-named MLIR op in the `cute` or `cute.nvgpu` dialect, carrying rich types (`!cute.layout`, `!cute.tiled_mma`, `!cute.copy_atom`, `!cute.tensor`).
- The protocol methods (`__get_mlir_types__`, `__extract_mlir_values__`, `__new_from_mlir_values__`) on runtime wrappers bridge Python objects to MLIR SSA values; these map one-to-one to the CuTe C++ value categories (layout/shape integers, descriptors, atom traits).
- Launch construction mirrors a C++ CUTLASS kernel launch: MLIR `cuda.kernel` + `gpu.launch_func` correspond to a `__global__` definition and a `cudaLaunchKernel` call; the arguments are the same objects you would pass to a templated CUTLASS kernel (`TiledMMA`, TMA descriptors, layouts, scheduler params, shared storage).
