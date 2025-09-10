"""
Temporary patch to force CuTe DSL pipeline to emit PTX instead of CUBIN.

Why: The default CuTe pipeline emits CUBIN (binary) and the runtime preloads it
via the CUDA driver. For inspection/debugging, we sometimes want the pipeline to
embed PTX text in the MLIR `gpu.binary` payload so it can be read directly.

This context manager:
- Rewrites the assembled pipeline string from `cubin-format=bin` to `cubin-format=ptx`.
- Optionally bypasses CUDA driver preloading (`JitExecutor.update_jit_cuda_modules`) so
  `cute.compile(..., compile_only=True)` completes successfully under PTX output.

Usage:

    from dsl_tutorials.utils.patch_ptx_pipeline import patch_pipeline_to_ptx
    from dsl_tutorials.utils.ptx_tools import extract_ptx_from_cute

    with patch_pipeline_to_ptx(skip_driver_load=True):
        ptx_map = extract_ptx_from_cute(my_jit_fn, *args, verbose=True)
        print(next(iter(ptx_map.values()))[:600])

Note: Do not attempt to execute GPU kernels while this patch is active; the
runtime path expects CUBINs for module loading.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Callable


@contextlib.contextmanager
def patch_pipeline_to_ptx(skip_driver_load: bool = True):
    """Patch CuTe pipeline to emit PTX and optionally skip CUDA driver loads.

    - Alters BaseDSL.preprocess_pipeline to post-process the final pipeline string
      by replacing `cubin-format=bin` with `cubin-format=ptx`.
    - If `skip_driver_load=True`, monkey-patches JitExecutor.update_jit_cuda_modules
      to a no-op, avoiding attempts to load PTX as a CUBIN.
    """
    # Resolve modules from either installed or source layouts
    from cutlass.base_dsl import dsl
    from cutlass.base_dsl import jit_executor
    dsl_mod = dsl
    jit_exec_mod = jit_executor

    BaseDSL = getattr(dsl_mod, "BaseDSL")

    # Keep originals
    original_preprocess = BaseDSL.preprocess_pipeline
    original_update = None
    if skip_driver_load and jit_exec_mod is not None:
        JitExecutor = getattr(jit_exec_mod, "JitExecutor")
        original_update = JitExecutor.update_jit_cuda_modules

    def _patched_preprocess(self, pipeline: str, arch: str) -> str:  # type: ignore[override]
        p = original_preprocess(self, pipeline, arch)
        
        # Replace the binary output selection to PTX text
        return p.replace("cubin-format=bin", "cubin-format=assembly")

    # Apply patches
    BaseDSL.preprocess_pipeline = _patched_preprocess  # type: ignore[assignment]

    if skip_driver_load and jit_exec_mod is not None:
        JitExecutor = getattr(jit_exec_mod, "JitExecutor")

        def _noop_update(self, kernel_symbols):  # type: ignore[override]
            return self

        JitExecutor.update_jit_cuda_modules = _noop_update  # type: ignore[assignment]

    try:
        yield
    finally:
        # Restore originals
        BaseDSL.preprocess_pipeline = original_preprocess  # type: ignore[assignment]
        if skip_driver_load and jit_exec_mod is not None and original_update is not None:
            JitExecutor.update_jit_cuda_modules = original_update  # type: ignore[assignment]

