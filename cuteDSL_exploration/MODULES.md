Module Map (cutlass_package)

High-level

- `_mlir/` — MLIR Python bindings layer
  - `ir.py` — core IR classes imported from `_mlir_libs`.
  - `passmanager.py` — exposes `PassManager`.
  - `execution_engine.py` — wraps `_mlirExecutionEngine.ExecutionEngine` with `lookup`, `invoke`, and `register_runtime`.
  - `dialects/` — auto‑generated dialect packages:
    - `gpu/__init__.py` (imports from `_mlirDialectsGPU`), `nvvm.py`, `nvgpu.py`, `cute.py` etc.
  - `runtime/np_to_memref.py` — converts numpy/ctypes to MLIR memref ABI structs.

- `base_dsl/` — DSL architecture and runtime
  - `dsl.py` — central class `BaseDSL` with:
    - `jit()` / `kernel()` classmethods returning decorators;
    - `jit_runner()` decorator factory and `_preprocess_and_execute()`;
    - IR generation: `generate_mlir()`, `generate_original_ir()`, `build_module()`;
    - pipeline plumping: `compile_and_jit()`, `compile_and_cache()`;
    - kernel utilities: `_KernelGenHelper`, `kernel_launcher()`, `generate_kernel_operands_and_types()`;
    - module hash + cache management.
  - `compiler.py` — `Compiler(passmanager, execution_engine)` with `compile()`, `jit()`, `compile_and_jit()` and `CompileOptions`.
  - `jit_executor.py` — `JitExecutor` with arg marshalling, CUDA cubin extraction and loading.
  - `runtime/cuda.py` — CUDA driver helpers (module load, get function, set attribute, launch, streams, alloc).
  - `ast_preprocessor.py` — optional AST transformer for control flow (`for/if/while`).
  - `typing.py`, `runtime/jit_arg_adapters.py` — type system and arg adapter registry used to bridge Python objects to MLIR/C ABI.

- `cutlass_dsl/` — CuTe dialect on top of base DSL
  - `cutlass.py` — `CutlassBaseDSL` (GPU module creation + pipeline customization) and `CuTeDSL` (wires MLIR bits and enables preprocessing). Also defines `KernelLauncher`.

- `cute/` — Public API
  - `__init__.py` re‑exports ops from `cute.core`, `cute.typing`, `cute.math`, exposes `jit = _dsl.CuTeDSL.jit` and `kernel = _dsl.CuTeDSL.kernel`.
  - `core.py` — CuTe algebra/cartesian utilities (many decorated with `@cute.jit` for IR caching/demonstration).
  - `arch/`, `nvgpu/` — GPU‑specific helpers and NVVM wrappers.

Pointers to notable definitions

- `CuTeDSL` instantiation: `cutlass_package/cutlass_dsl/cutlass.py:408–422`.
- `cute.jit` / `cute.kernel` aliases: `cutlass_package/cute/__init__.py` (bottom section).
- `@jit` entry: `cutlass_package/base_dsl/dsl.py:485–493`, `457–483`.
- IR module assembly: `cutlass_package/base_dsl/dsl.py:1083–1146`.
- Pipeline + compile + JIT: `cutlass_package/base_dsl/compiler.py:135–185`.
- Execution engine wrapper: `cutlass_package/_mlir/execution_engine.py:1–38`.
- Kernel generation + `gpu.launch_func`: `cutlass_package/base_dsl/dsl.py:1559–1712` and helper `cutlass_package/cutlass_dsl/cutlass.py:232–277`.
- CUDA cubin extraction + kernel pointer: `cutlass_package/base_dsl/jit_executor.py:260–357`.

