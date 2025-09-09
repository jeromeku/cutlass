# CuTe DSL: Where PTX/CUBIN Is Generated and How to Extract PTX

This note documents where the CuTe DSL compilation pipeline produces the GPU device image, how it is embedded into MLIR, and two practical ways to extract PTX for inspection.

## Where the Device Image Is Produced

- The CuTe pipeline string is built in the Cutlass DSL shim and Base DSL:
  - `cutlass_dsl/cutlass.py` selects a default: `builtin.module(cute-to-nvvm{cubin-format=bin ...})`.
  - `base_dsl/dsl.py` adds options (e.g., `toolkitPath`, `cubin-chip=<arch>`) via `preprocess_pipeline`.
- The pass manager is executed here, which generates the device image and embeds it:
  - `base_dsl/compiler.py`: `PassManager.parse(...).run(module.operation)`.
- The resulting image is carried in `gpu.binary` as a `bin = "..."` payload inside the MLIR module.
  - At runtime this blob is discovered and (for the default CUBIN flow) loaded via the CUDA driver in `base_dsl/jit_executor.py`.

By default, the pipeline emits a CUBIN (`cubin-format=bin`). If configured to emit PTX instead, the same `gpu.binary` payload will contain PTX text.

## Extracting PTX Without Changing the Pipeline

Use `dsl_tutorials/utils/ptx_tools.py`:

```python
from dsl_tutorials.utils.ptx_tools import extract_ptx_from_cute

# Compile-only and try to recover PTX from CUBIN via cuobjdump (if embedded)
ptx_map = extract_ptx_from_cute(my_jit_fn, *args, cuobjdump="/usr/local/cuda/bin/cuobjdump", verbose=True)
```

This works when your CUBIN contains PTX sections. It does not transform the pipeline.

## Forcing the Pipeline to Emit PTX (Developer Inspection)

For inspection, you may prefer the pipeline to embed PTX directly. The CuTe default expects CUBINs for execution, so we patch two behaviors temporarily:

1) Replace `cubin-format=bin` with `cubin-format=ptx` in the assembled pipeline string.
2) Bypass CUDA driver preloading (which expects CUBIN) so `cute.compile(..., compile_only=True)` still succeeds.

Use the helper context below (see `dsl_tutorials/utils/patch_ptx_pipeline.py`):

```python
from dsl_tutorials.utils.patch_ptx_pipeline import patch_pipeline_to_ptx
from dsl_tutorials.utils.ptx_tools import extract_ptx_from_cute

with patch_pipeline_to_ptx(skip_driver_load=True):
    ptx_map = extract_ptx_from_cute(my_jit_fn, *args, verbose=True)
    for sym, ptx in ptx_map.items():
        print(f"===== {sym} =====\n{ptx[:600]}\n...")
```

Notes:
- The patch is process-local and reversible. It does not modify installed packages on disk.
- Use this only for PTX inspection. Executing kernels under this patch is not supported because the runtime expects CUBINs for module loading.

## Architecture Overrides

If you also want to force a specific target architecture/toolkit during compilation, pair with the arch override utilities:

```python
from dsl_tutorials.utils.ptx_tools import override_cute_arch

with override_cute_arch("sm_90a", toolkit_path="/usr/local/cuda"):
    with patch_pipeline_to_ptx():
        ptx_map = extract_ptx_from_cute(my_jit_fn, *args, verbose=True)
```

This will log the effective settings and yield PTX text embedded by the patched pipeline.

