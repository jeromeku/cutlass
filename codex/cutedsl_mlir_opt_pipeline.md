# CuTeDSL custom `mlir-opt` pipeline replay

CuTeDSL does not shell out to a standalone `mlir-opt` binary during normal compilation. It runs the pass pipeline in-process through the Python MLIR bindings:

- [Compiler.compile](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136)
- [CutlassBaseDSL._get_pipeline](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L254)
- [BaseDSL.preprocess_pipeline](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1000)
- [CuTeDSL.preprocess_pipeline](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L266)

## Default CuTeDSL pipeline

For the CuTe DSL singleton in this environment, the default pipeline is:

```text
builtin.module(cute-to-nvvm{cubin-format=bin opt-level=3 enable-assertions=false preserve-line-info=false})
```

After preprocessing for architecture and CUDA dialect options, it becomes:

```text
builtin.module(cute-to-nvvm{cubin-format=bin opt-level=3 enable-assertions=false preserve-line-info=false cubin-chip=sm_100a enable-cuda-dialect=true cuda-dialect-external-module=true})
```

## Replay the pipeline on dumped MLIR

```python
from pathlib import Path

from cutlass._mlir import ir, passmanager
from cutlass.cutlass_dsl.cutlass import CuTeDSL

dsl = CuTeDSL._get_dsl()

pipeline = dsl.preprocess_pipeline(
    dsl._get_pipeline(None),
    dsl.envar.arch,
)

with ir.Context() as ctx:
    module = ir.Module.parse(Path("generated.mlir").read_text())
    pm = passmanager.PassManager.parse(pipeline, context=ctx)
    pm.enable_verifier(True)
    pm.enable_ir_printing(
        print_before_all=False,
        print_after_all=True,
        print_module_scope=True,
        print_after_change=True,
    )
    pm.run(module.operation)

print(module)
```

That is the closest equivalent to running `mlir-opt generated.mlir -pass-pipeline='...'`, except it uses the same loaded CUTLASS/CuTe dialect and pass registrations as CuTeDSL itself.

## If you want to use your own pipeline string

Replace `dsl._get_pipeline(None)` with your own pipeline string:

```python
pipeline = "builtin.module(cute-to-nvvm{cubin-format=bin cubin-chip=sm_100a enable-cuda-dialect=true cuda-dialect-external-module=true})"
```

Then parse and run it the same way.

## If you want CuTeDSL to compile with your custom pipeline

Pass the pipeline directly into the JIT entry point:

```python
compiled = cute.compile(my_func, *args, pipeline=your_pipeline)
```

`compile_and_cache(...)` forwards that through [BaseDSL._func](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1505) and [BaseDSL.compile_and_cache](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1165), then runs it with [Compiler.compile](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136).

## Important caveat

I did not find a shipped standalone `mlir-opt`, `cutlass-opt`, or `cute-opt` binary in this repo checkout or `.venv`. A plain upstream `mlir-opt` binary will usually not know the CUTLASS-specific passes such as `cute-to-nvvm`, so the in-process Python replay above is the supported route unless you build a custom opt binary that links the same dialect/pass libraries.
