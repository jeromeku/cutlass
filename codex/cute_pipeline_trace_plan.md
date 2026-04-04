# CuTeDSL Pipeline Trace Plan

1. Trace the concrete entrypoint in `experiments/rmsnorm.py` (compile call at line 325) and record runtime inputs/options.
2. Follow `cute.compile` into `CompileCallable`, callable normalization, and `@cute.jit` dispatch.
3. Trace AST preprocessing (`DSLPreprocessor`) and show how rewritten control flow feeds runtime tracing.
4. Trace IR generation (`generate_original_ir`) through `@cute.kernel` lowering and launch construction.
5. Document MLIR pass pipeline stages and binary lowering (PTX/CUBIN), including toolchain details.
6. Compare `enable_tvm_ffi=True/False` paths and summarize `@cute.jit` vs `@cute.kernel`.
7. Produce final trace doc with code map, sequence diagram, flowchart, class/module map, and key function index.
