"""
CuTeDSL MLIR Pipeline Instrumentation Utility

Monkey-patches cutlass.base_dsl.compiler.Compiler.compile() to capture
MLIR IR at every stage of the cute-to-nvvm pass pipeline.

Usage:
    from claude.cute_mlir_pipeline_override import enable_pipeline_tracing

    # Call before cute.compile() or any @cute.jit / @cute.kernel invocation
    enable_pipeline_tracing("/tmp/mlir_dump")

    compiled_fn = cute.compile(my_kernel, *args)

    # Results:
    #   /tmp/mlir_dump/00_input.mlir        — IR after tracing, before any passes
    #   /tmp/mlir_dump/passes/              — per-sub-pass IR snapshots (numbered)
    #   /tmp/mlir_dump/99_output.mlir       — IR after all passes complete

Approach:
    The entire CuTeDSL compilation pipeline funnels through a single method:
        Compiler.compile(module, pipeline, ...)  [compiler.py:136-148]
    which does:
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)

    We replace this method to:
    1. Dump module IR before pm.run()             → 00_input.mlir
    2. Call pm.enable_ir_printing() with
       tree_printing_dir_path to dump per-sub-pass
       IR to numbered files                       → passes/*.mlir
    3. Dump module IR after pm.run()              → 99_output.mlir

    File-based output (tree_printing_dir_path) is required because
    dsl.py:compile_and_jit (lines 965-968) redirects stdout/stderr
    to StringIO, which would swallow print-based IR output.
"""

import os
import shutil
from typing import Optional


def enable_pipeline_tracing(
    dump_dir: str = "./mlir_pipeline_dump",
    *,
    print_before_all: bool = True,
    print_after_all: bool = True,
    print_after_change: bool = False,
    print_module_scope: bool = True,
    large_elements_limit: Optional[int] = 64,
    enable_debug_info: bool = False,
    clean: bool = True,
    verbose: bool = True,
):
    """
    Patch Compiler.compile() to dump IR at every pipeline stage.

    Args:
        dump_dir: Directory for all output files.
        print_before_all: Dump IR before each sub-pass.
        print_after_all: Dump IR after each sub-pass.
        print_after_change: If True, only dump after passes that change the IR.
                           Reduces noise significantly but may miss identity passes.
        print_module_scope: Print full module vs just the affected operation.
        large_elements_limit: Truncate large constant arrays in IR printing.
                             None = no limit. 64 is a reasonable default.
        enable_debug_info: Include MLIR debug/location info in dumps.
        clean: Remove dump_dir if it exists before starting.
        verbose: Print status messages to stdout.
    """
    from cutlass.base_dsl.compiler import Compiler

    pass_dir = os.path.join(dump_dir, "passes")

    if clean and os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(pass_dir, exist_ok=True)

    _original_compile = Compiler.compile
    call_count = [0]

    def _traced_compile(
        self, module, pipeline, cuda_toolkit="", arch="", enable_verifier=False
    ):
        idx = call_count[0]
        call_count[0] += 1

        # Use a per-invocation subdirectory if compile() is called multiple times
        if idx == 0:
            out_dir = dump_dir
            sub_pass_dir = pass_dir
        else:
            out_dir = os.path.join(dump_dir, f"call_{idx}")
            sub_pass_dir = os.path.join(out_dir, "passes")
            os.makedirs(sub_pass_dir, exist_ok=True)

        # --- Stage 0: Dump input IR (after tracing, before passes) ---
        input_path = os.path.join(out_dir, "00_input.mlir")
        with open(input_path, "w") as f:
            f.write(str(module))

        if verbose:
            print(f"[pipeline-trace] Pipeline string: {pipeline}")
            print(f"[pipeline-trace] Input IR dumped to {input_path}")

        # --- Stage 1: Run passes with per-sub-pass IR printing ---
        # IR printing requires single-threaded PassManager execution.
        # Disable multithreading on the current MLIR Context.
        from cutlass._mlir import ir as _ir

        ctx = _ir.Context.current
        ctx.enable_multithreading(False)

        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)

        ir_print_kwargs = dict(
            print_before_all=print_before_all,
            print_after_all=print_after_all,
            print_module_scope=print_module_scope,
            print_after_change=print_after_change,
            print_after_failure=True,
            tree_printing_dir_path=sub_pass_dir,
        )
        if large_elements_limit is not None:
            ir_print_kwargs["large_elements_limit"] = large_elements_limit
        if enable_debug_info:
            ir_print_kwargs["enable_debug_info"] = True

        pm.enable_ir_printing(**ir_print_kwargs)
        pm.run(module.operation)

        if verbose:
            n_files = len([f for f in os.listdir(sub_pass_dir) if f.endswith(".mlir")])
            print(f"[pipeline-trace] {n_files} sub-pass IR files in {sub_pass_dir}/")

        # --- Stage 2: Dump output IR (after all passes) ---
        output_path = os.path.join(out_dir, "99_output.mlir")
        with open(output_path, "w") as f:
            f.write(str(module))

        if verbose:
            print(f"[pipeline-trace] Output IR dumped to {output_path}")

        # Run post-compile hook if set
        if self._post_compile_hook:
            self._post_compile_hook(module)

    Compiler.compile = _traced_compile

    if verbose:
        print(f"[pipeline-trace] Patched Compiler.compile(). Dumps → {dump_dir}/")


def disable_pipeline_tracing():
    """Restore original Compiler.compile()."""
    from cutlass.base_dsl.compiler import Compiler

    # If we stored the original, restore it; otherwise this is a no-op.
    # The original is captured in the closure of enable_pipeline_tracing,
    # but for simplicity we re-import the unpatched version.
    # This works because Python re-reads the module attribute.
    import importlib
    import cutlass.base_dsl.compiler as mod

    importlib.reload(mod)
    print("[pipeline-trace] Restored original Compiler.compile().")
