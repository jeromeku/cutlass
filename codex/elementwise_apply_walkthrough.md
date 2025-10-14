# Elementwise Apply: Full Pipeline Walkthrough

This document dissects the entire compilation pipeline for the Ampere example elementwise_apply.py, line by line, with deep call stacks and source links you can Ctrl/Cmd+Click in VSCode.

Target file:
- [examples/python/CuTeDSL/ampere/elementwise_apply.py](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py)

We cover three main stages:
- Python AST Parsing and transformation
- Host path via `@cute.jit`
- Device path via `@cute.kernel`

For each stage, we show inputs/outputs and where each translation occurs.

---

## 1) Python AST Parsing (preprocess → transformed AST)

CuTeDSL optionally rewrites Python AST to normalize control flow (for/if/while) before IR generation.

- Entry points
  - Decorators are resolved through `BaseDSL.jit` and `BaseDSL.kernel`:
    - [python/CuTeDSL/cutlass/base_dsl/dsl.py:494](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:494) `def jit(...)`
    - [python/CuTeDSL/cutlass/base_dsl/dsl.py:504](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:504) `def kernel(...)`
  - When a decorated function is first called, `_preprocess_and_execute` runs the AST preprocessor:
    - [python/CuTeDSL/cutlass/base_dsl/dsl.py:556](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:556)

- AST machinery
  - Transformer class: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:131](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:131) `class DSLPreprocessor`
  - Build transformed module: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:445](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:445) `def transform(...)`
  - Execute the transformed code: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:238](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:238) `def exec(...)`

- What gets transformed
  - `cutlass.range_constexpr(...)` loops → fully unrolled. `range_dynamic`/`range` → `scf.for` (later during IR).
  - `if cutlass.const_expr(...)` → compile-time folding; dynamic `if` → `scf.if` (later during IR).

- Example from elementwise_apply_kernel
  - Lines [100–103](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:100) and [121–125](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:121) use `cutlass.range_constexpr(len(...))` to iterate at compile time. The preprocessor ensures these loops become straight-line IR (no runtime loop).

Inputs/Outputs

| Stage | Input | Output |
|---|---|---|
| AST Preprocess | Python function objects (decorated) | New Python function objects compiled from transformed AST |

Snippet: AST → transformed function materialization

```python
# python/CuTeDSL/cutlass/base_dsl/dsl.py:556
if hasattr(func, "_transformed_ast"):
    func._transformed_ast = func._dsl_object.run_preprocessor(func)  # transform
    fcn_ptr = func._dsl_object.get_function_ptr(func)  # compile+exec
    return DSLCallable(fcn_ptr)
```

---

## 2) Host Path via `@cute.jit` (Python → MLIR → JIT)

The host entry (`elementwise_apply`) is decorated with `@cute.jit` and orchestrates kernel launch.

- Where `@cute.jit` attaches
  - Example declaration: [examples/python/CuTeDSL/ampere/elementwise_apply.py:171](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:171)
  - Decorator root: [python/CuTeDSL/cutlass/base_dsl/dsl.py:494](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:494)

- Host IR building
  - Overall flow: [python/CuTeDSL/cutlass/base_dsl/dsl.py:1327](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:1327) `_func`
  - Convert args to MLIR types: [python/CuTeDSL/cutlass/base_dsl/dsl.py:842](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:842) `generate_mlir_function_types`
  - Create module + host function: [python/CuTeDSL/cutlass/base_dsl/dsl.py:1067](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:1067) `generate_original_ir`
  - `func.func` for host entry: [python/CuTeDSL/cutlass/base_dsl/dsl.py:1094](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:1094)

Snippet: building host `func.func`

```python
# python/CuTeDSL/cutlass/base_dsl/dsl.py:1090-1100
fop = func.FuncOp(function_name, (func_types, []), loc=loc)
fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
with ir.InsertionPoint(fop.add_entry_block()):
    ir_args, ir_kwargs = self.generate_execution_arguments(args, kwargs, fop, args_spec)
    result = funcBody(*ir_args, **ir_kwargs)
    func.ReturnOp([])
```

- Launch site (user code)
  - The compiled host function constructs tilers/layouts then invokes the kernel launcher:
    - [examples/python/CuTeDSL/ampere/elementwise_apply.py:265](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:265) `elementwise_apply_kernel(...).launch(...)`

- Pipeline options injection and compilation
  - Pipeline preprocess: [python/CuTeDSL/cutlass/base_dsl/dsl.py:973](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:973)
  - Compile + JIT: [python/CuTeDSL/cutlass/base_dsl/compiler.py:168](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/compiler.py:168)
  - PassManager execution: [python/CuTeDSL/cutlass/base_dsl/compiler.py:135](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/compiler.py:135)

Snippet: pipeline option stitching

```python
# python/CuTeDSL/cutlass/base_dsl/dsl.py:973-1001
options = {"toolkitPath": self.envar.cuda_toolkit if self.envar.cuda_toolkit else None,
           self.pass_sm_arch_name: arch}
opt_str = " ".join(f"{k}={v}" for k,v in options.items() if v)
if opt_str:
    pattern = re.compile(r"{(.+)}"); match = pattern.search(pipeline)
    pipeline = (re.sub(r"{.+}", f"{{{match[1]} {opt_str}}}", pipeline)
                if match else pipeline.rstrip(")") + f"{{{opt_str}}})"
```

Inputs/Outputs

| Stage | Input | Output |
|---|---|---|
| Host JIT | Python function + args | MLIR module (host `func.func` + `gpu.module`) + JITed host entry |

---

## 3) Device Path via `@cute.kernel` (Python → MLIR kernel → CUBIN)

The device kernel `elementwise_apply_kernel` is decorated with `@cute.kernel`.

- Where `@cute.kernel` attaches
  - Example: [examples/python/CuTeDSL/ampere/elementwise_apply.py:78](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:78)
  - Decorator root: [python/CuTeDSL/cutlass/base_dsl/dsl.py:504](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:504)
  - Kernel generation wrapper: [python/CuTeDSL/cutlass/base_dsl/dsl.py:1536](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:1536) `kernel_launcher`

- Building the GPU container and kernel function
  - GPU module creation: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:206](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:206) `_build_gpu_module`
  - Kernel func op emission is delegated to helper: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:318](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:318) `_kernel_helper`
  - Launch op generation: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:377](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:377)

Snippet: generating `gpu.launch_func`

```python
# python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:377-410
kernel_sym = ir.SymbolRefAttr.get(["kernels", kernel_name])
token = gpu.launch_func(
    gpu.AsyncTokenType.get() if is_async else None,
    cfg.async_deps,
    kernelSym,
    *cfg.grid,
    *cfg.block,
    kernelOperands,
    **dict(zip(("cluster_size_x","cluster_size_y","cluster_size_z"), tuple(cfg.cluster))),
    dynamic_shared_memory_size=cfg.smem,
)
```

- The kernel body translation
  - Slice/index ops like `t[idx]` map into Cute/Cute-NVGPU ops which later lower to NVGPU/NVVM.
  - Examples (per-thread slicing and fragment ops):
    - Slicing by CTA and thread:
      - [examples/.../elementwise_apply.py:95](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:95) `ctaInputs = [t[cta_coord] for t in inputs]`
      - [examples/.../elementwise_apply.py:117](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:117) `thrInputs = [t[thr_coord] for t in tidfrgInputs]`
    - Fragment creation and predication:
      - [examples/.../elementwise_apply.py:128](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:128)
      - [examples/.../elementwise_apply.py:132](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:132)
    - Copy and compute:
      - [examples/.../elementwise_apply.py:157](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:157)
      - [examples/.../elementwise_apply.py:162](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:162)
      - [examples/.../elementwise_apply.py:168](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:168)

- Lowering to CUBIN and binding
  - Pass pipeline (`cute-to-nvvm`) lowers Cute/Cute-NVGPU → NVGPU/NVVM and embeds CUBIN:
    - [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:214](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:214)
  - CUBIN extraction and CUDA Driver binding:
    - [python/CuTeDSL/cutlass/base_dsl/jit_executor.py:330](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/jit_executor.py:330)
    - [python/CuTeDSL/cutlass/base_dsl/jit_executor.py:259](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/jit_executor.py:259)
  - Host function entry lookup/invoke:
    - [python/CuTeDSL/cutlass/_mlir/execution_engine.py:13](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/_mlir/execution_engine.py:13)

Inputs/Outputs

| Stage | Input | Output |
|---|---|---|
| Device Kernel | Python kernel body | MLIR `gpu.module` + kernel `func.func` → NVVM → embedded CUBIN |

---

## Line-by-Line Highlights for elementwise_apply.py

| Line(s) | Code (trimmed) | What happens |
|---|---|---|
| [78](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:78) | `@cute.kernel` | Registers device function with DSL; callsite becomes a launcher. |
| [79–86](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:79) | `def elementwise_apply_kernel(op, inputs, gC, cC, shape, tv_layout)` | Kernel signature → MLIR function params (Cute types map to MLIR types via DSL typing). |
| [87–125](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:87) | Thread/Block ids, CTA/thread slicing | Emits indexing ops and layout compositions (Cute/NVGPU). |
| [128–136](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:128) | Fragments + predicate fragment | Local register fragments; pred computed for bounds. |
| [146–155](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:146) | `make_copy_atom` load/store | Configures copy atoms (vector width, dtype). |
| [157–168](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:157) | `cute.copy` + compute + store | Loads into frags, applies `op` (constexpr specialized), stores back. |
| [171](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:171) | `@cute.jit` | Host entry decoration. |
| [172–239](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:172) | `def elementwise_apply(...)` | Host code computes tilers/layouts and tile tensors. |
| [265–276](/home/jeromeku/cutlass/examples/python/CuTeDSL/ampere/elementwise_apply.py:265) | `.launch(grid=..., block=..., stream=...)` | Enqueues `gpu.launch_func` with kernel symbol and runtime operands. |

---

## End-to-End Diagrams

High-level compilation flow

```
Python (jit+kernel)
   │  ASTPreprocessor (optional)
   ▼
MLIR module { host func.func + gpu.module@kernels { kernel func.func } }
   │  PassManager: cute-to-nvvm { cubin-chip=..., toolkitPath=... }
   ▼
MLIR gpu.binary (embedded CUBIN)
   │  JitExecutor: cuModuleLoadData + cuModuleGetFunction
   ▼
ExecutionEngine.invoke(_mlir_ciface_<host>) → gpu.launch_func(kernel_ptr)
```

Where the key pieces live

| Responsibility | File |
|---|---|
| Decorators (jit/kernel) | [python/CuTeDSL/cutlass/base_dsl/dsl.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:494) |
| AST transform/exec | [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:131) |
| Host IR creation | [python/CuTeDSL/cutlass/base_dsl/dsl.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:1067) |
| GPU module/kernel | [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:206) |
| Pipeline injection | [python/CuTeDSL/cutlass/base_dsl/dsl.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:973) |
| PassManager+JIT | [python/CuTeDSL/cutlass/base_dsl/compiler.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/compiler.py:135) |
| CUBIN extraction | [python/CuTeDSL/cutlass/base_dsl/jit_executor.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/jit_executor.py:330) |
| Host invoke lookup | [python/CuTeDSL/cutlass/_mlir/execution_engine.py](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/_mlir/execution_engine.py:13) |

---

## Notes on constexpr `op`

The `op` parameter is annotated `cutlass.Constexpr` in both host and kernel signatures. At compile time, the DSL specializes the kernel for the provided operator (e.g., `operator.add` vs user lambda), so there is no runtime function pointer overhead inside the device code.

- Adapter/filtering of constexpr vs runtime args happens in:
  - [python/CuTeDSL/cutlass/base_dsl/runtime/jit_arg_adapters.py:20](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/runtime/jit_arg_adapters.py:20) `is_arg_spec_constexpr`
  - [python/CuTeDSL/cutlass/base_dsl/jit_executor.py:54](/home/jeromeku/cutlass/python/CuTeDSL/cutlass/base_dsl/jit_executor.py:54) `filter_runtime_arg_spec`

This is why, when benchmarking the compiled function, the `op` is omitted (already specialized), as noted in the example.

---

## Summary

- Python source is optionally transformed by the DSL AST preprocessor, then executed to build MLIR.
- Host `@cute.jit` produces a `func.func` that prepares operands and enqueues `gpu.launch_func`.
- Device `@cute.kernel` produces a kernel `func.func` inside `gpu.module @kernels`.
- The MLIR pass pipeline lowers to NVVM and embeds a target‐specific CUBIN in `gpu.binary`.
- The runtime extracts CUBIN, loads via CUDA Driver API, resolves the kernel symbol, and invokes the host entry through MLIR’s ExecutionEngine.
