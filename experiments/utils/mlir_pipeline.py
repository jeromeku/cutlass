# ruff: noqa E402

def dump_mlir_pipeline(dump_dir="cute_pipeline", *, dump_ptx: bool = False, module_name: str = None):
    import os
    from cutlass.base_dsl import compiler as _cute_compiler    
    from cutlass.base_dsl.utils.logger import log
    from cutlass._mlir.ir import Module
    CompilationError = _cute_compiler.CompilationError
    logger = log()
    module_name = module_name or "MODULE"

    def _compile(
        self: _cute_compiler.Compiler,
        module: Module,
        pipeline: str,
        cuda_toolkit: str = "",
        arch: str = "",
        enable_verifier=False,
    ):
        """Compiles the module by invoking the pipeline."""
        try:
            ctx = module.context
            if dump_ptx:
                print("DEPRECATED: use cuteDSL's builtin PTX / CUBIN dumping tools")
                pipeline = pipeline.replace("cubin-format=bin", "cubin-format=isa")
            logger.debug(f"Dumping mlir pipeline to {dump_dir}")
            breakpoint()
            os.makedirs(dump_dir, exist_ok=True)
            before_path = os.path.join(dump_dir, "BEFORE.mlir")
            after_path = os.path.join(dump_dir, "AFTER.mlir")
            module.dump()
            with open(before_path, 'w') as f:
                print(module, file=f)

            pm = self.passmanager.PassManager.parse(pipeline)
            ctx.enable_multithreading(False)
            ctx.emit_error_diagnostics = True
            pm.enable_verifier(enable_verifier)

            # Print before/after every pass, include locations, and dump each pass’s IR to a directory.
            pm.enable_ir_printing(
                print_before_all=True,
                print_after_all=True,
                print_module_scope=True,
                print_after_change=False,
                print_after_failure=True,
                enable_debug_info=True,
                tree_printing_dir_path=dump_dir
            )
            pm.run(module.operation)

            with open(after_path, 'w') as f:
                print(module, file=f)

        except Exception as e:
            error_msg = str(e)
            nvvm_error, ir_msg = self._process_error(error_msg)

            if nvvm_error:
                raise CompilationError(
                    error_msg,
                    nvvm_error=nvvm_error,
                    ir_context=ir_msg,
                    cuda_toolkit=cuda_toolkit,
                    arch=arch,
                ) from e
            raise e
                
    _original_compile = _cute_compiler.Compiler.compile
    _cute_compiler.Compiler.compile = _compile
