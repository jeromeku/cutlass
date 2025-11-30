# CuTeDSL Compilation Pipeline: Complete Technical Deep Dive

This document provides an exhaustive, frame-by-frame trace of the CuTeDSL compilation pipeline, from Python decorators through MLIR generation to CUDA kernel execution.

## Table of Contents

1. [Overview](#overview)
2. [Decorator Entry Points](#decorator-entry-points)
3. [@cute.jit vs @cute.kernel](#cutejit-vs-cutekernel)
4. [Complete Compilation Flow](#complete-compilation-flow)
5. [Stage 1: AST Preprocessing](#stage-1-ast-preprocessing)
6. [Stage 2: Function Tracing & Argument Processing](#stage-2-function-tracing--argument-processing)
7. [Stage 3: MLIR Generation](#stage-3-mlir-generation)
8. [Stage 4: Kernel Generation (for @cute.kernel)](#stage-4-kernel-generation)
9. [Stage 5: MLIR Compilation Pipeline](#stage-5-mlir-compilation-pipeline)
10. [Stage 6: PTX/CUBIN Generation](#stage-6-ptxcubin-generation)
11. [Stage 7: Kernel Binding & Launching](#stage-7-kernel-binding--launching)
12. [cute.compile Function](#cutecompile-function)

---

## Overview

CuTeDSL is a Python-embedded DSL that compiles high-level Python code to efficient CUDA kernels via MLIR. The compilation pipeline consists of several distinct stages:

```
Python Code (@cute.jit/@cute.kernel)
    ↓
[Optional] AST Preprocessing
    ↓
Function Tracing & Argument Processing
    ↓
MLIR Module Generation
    ↓
[For @kernel] Kernel Generation with Launch Config
    ↓
MLIR Pass Pipeline (cute-to-nvvm)
    ↓
PTX Generation
    ↓
CUBIN Assembly
    ↓
JIT Execution Engine Creation
    ↓
Kernel Binding & Launch
```

---

## Decorator Entry Points

### Location: [python/CuTeDSL/cutlass/cute/__init__.py](../python/CuTeDSL/cutlass/cute/__init__.py#L196-L208)

The public API exports decorator aliases:

```python
# Used as internal symbol
from .. import cutlass_dsl as _dsl

# Aliases
jit = _dsl.CuTeDSL.jit
kernel = _dsl.CuTeDSL.kernel
compile = _dsl.CompileCallable()
```

**Key Points:**
- `@cute.jit` → `CuTeDSL.jit` class method
- `@cute.kernel` → `CuTeDSL.kernel` class method
- `cute.compile` → `CompileCallable()` instance

---

## @cute.jit vs @cute.kernel

### Base Implementation: [python/CuTeDSL/cutlass/base_dsl/dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L504-L517)

Both decorators are thin wrappers around `BaseDSL.jit_runner()`:

```python
@classmethod
def jit(cls, *dargs, **dkwargs):
    """
    Decorator to mark a function for JIT compilation for Host code.
    """
    frame = inspect.currentframe().f_back
    return BaseDSL.jit_runner(cls, "_func", frame, *dargs, **dkwargs)

@classmethod
def kernel(cls, *dargs, **dkwargs):
    """
    Decorator to mark a function for JIT compilation for GPU.
    """
    frame = inspect.currentframe().f_back
    return BaseDSL.jit_runner(cls, "_kernel_helper", frame, *dargs, **dkwargs)
```

### Key Differences

| Feature | @cute.jit | @cute.kernel |
|---------|-----------|--------------|
| **Executor** | `_func` | `_kernel_helper` |
| **Purpose** | Host functions | Device kernel functions |
| **Launch Config** | Not required | Requires `LaunchConfig` |
| **GPU Module** | Optional | Mandatory |
| **Function Op** | `func.FuncOp` | `cuda_dialect.KernelOp` |
| **Return Op** | `func.ReturnOp` | `cuda_dialect.ReturnOp` |
| **Launch Op** | None | `cuda_dialect.launch_ex()` |

**Critical Insight:**
- The `executor_name` parameter (`"_func"` vs `"_kernel_helper"`) determines which code generation path is taken
- `@cute.jit` generates standard MLIR functions
- `@cute.kernel` generates CUDA kernel ops with explicit launch configurations

---

## Complete Compilation Flow

### High-Level Call Graph

```
@cute.jit/kernel decorator
    ↓
BaseDSL.jit_runner()  [dsl.py:419-478]
    ↓
func.__wrapped__ = original function
    ↓
[On function call]
    ↓
BaseDSL._preprocess_and_execute()  [dsl.py:480-502]
    ↓
DSLPreprocessor.transform() [optional]  [ast_preprocessor.py]
    ↓
BaseDSL._func() or _kernel_helper()  [dsl.py:1498-1567 or cutlass.py:652-747]
    ↓
BaseDSL.generate_mlir()  [dsl.py:1332-1414]
    ↓
Compiler.compile_and_jit()  [compiler.py:176-193]
    ↓
PassManager.run()  [compiler.py:136-164]
    ↓
ExecutionEngine creation  [compiler.py:166-174]
    ↓
JitExecutor.run_compiled_program()  [jit_executor.py:492-508]
```

---

## Stage 1: AST Preprocessing

### Location: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py)

### Purpose
Transform Python AST to insert DSL-specific control flow constructs, making Python control flow traceable in MLIR.

### What Gets Transformed

1. **For Loops** → `@loop_selector` decorator
2. **If-Elif-Else** → `@if_selector` decorator
3. **Variable Scoping** → Yield operations for state management

### Frame-by-Frame Trace

#### Entry Point: `BaseDSL.run_preprocessor()` [dsl.py:1416-1430]

```python
def run_preprocessor(self, funcBody):
    if not hasattr(funcBody, "_preprocessed"):
        function_name = funcBody.__name__
        self.funcBody = funcBody
        log().info("Started preprocessing [%s]", function_name)
        exec_globals = self._get_globals()
        # Transform the AST
        transformed_ast = self.preprocessor.transform(funcBody, exec_globals)
        if self.envar.print_after_preprocessor:
            log().info(f"# Printing unparsed AST after preprocess of func=`{function_name}` id=`{id(funcBody)}`")
            DSLPreprocessor.print_ast(transformed_ast)
        funcBody._preprocessed = True
        return transformed_ast
    return None
```

**What Happens:**
1. Check if function already preprocessed (avoid duplicate transformation)
2. Get global scope for imports/symbols
3. Call `DSLPreprocessor.transform()`
4. Mark function as preprocessed
5. Return transformed AST

#### Preprocessor Transform: `DSLPreprocessor.transform()` [ast_preprocessor.py:890-955]

```python
def transform(self, func: Callable, exec_globals: dict) -> ast.Module:
    """
    Main entry point for transforming a function's AST.

    Steps:
    1. Extract source code and parse to AST
    2. Extract imports from function scope
    3. Transform control flow (for/if)
    4. Generate yield operations
    5. Compile and execute in isolated namespace
    """
    # Get source and parse to AST
    source = inspect.getsource(func)
    tree = ast.parse(textwrap.dedent(source))

    # Walk AST and transform
    transformed_tree = self.visit(tree)

    # Fix missing locations
    ast.fix_missing_locations(transformed_tree)

    return transformed_tree
```

#### For Loop Transformation

**Before:**
```python
for i in range(10):
    result[i] = a[i] + b[i]
```

**After (conceptual):**
```python
@loop_selector(range(10))
def loop_body(i):
    result[i] = a[i] + b[i]
    yield result  # Auto-generated yield for modified vars
```

#### If Statement Transformation

**Before:**
```python
if condition:
    x = a
else:
    x = b
```

**After (conceptual):**
```python
@if_selector(condition)
def if_branch():
    x = a
    yield x

@else_selector()
def else_branch():
    x = b
    yield x

x = merge_results()
```

### Scope Management: `ScopeManager` [ast_preprocessor.py:134-180]

```python
@dataclass
class ScopeManager:
    """
    Manages symbol scopes during AST traversal.
    Tracks:
    - Variables defined in each scope
    - Read vs write accesses
    - Which variables need yield operations
    """
    scopes: List[Set[str]]

    def add_to_scope(self, name: str) -> None:
        """Add variable to current scope"""
        if name == "_":
            return
        assert len(self.scopes) > 0, "No active scope"
        self.scopes[-1].add(name)

    def is_in_scope(self, name: str) -> bool:
        """Check if variable in any scope"""
        return any(name in scope for scope in self.scopes)
```

**Why Scope Tracking Matters:**
- Determines which variables need yield operations
- Tracks variable lifetime across control flow boundaries
- Ensures proper SSA (Static Single Assignment) form in MLIR

### When Preprocessing is Active

Controlled by `_can_preprocess()` check in `BaseDSL._preprocess_and_execute()` [dsl.py:480-502]:

```python
def _can_preprocess(self):
    return (
        self.preprocess and
        self.preprocessor is not None and
        not self.device_compilation_only
    )

def _preprocess_and_execute(self, func):
    """Run preprocessor if enabled and not already preprocessed"""
    if self._can_preprocess():
        transformed_ast = self.run_preprocessor(func)
        if transformed_ast is not None:
            func._transformed_ast = transformed_ast
            return self.get_function_ptr(func)
    return func
```

**Conditions:**
1. `preprocess=True` in DSL initialization
2. Preprocessor object exists
3. Not in device-only compilation mode

---

## Stage 2: Function Tracing & Argument Processing

### Entry Point: `BaseDSL._func()` [dsl.py:1498-1567]

This is the heart of the compilation pipeline. Every decorated function call flows through here.

### Frame 1: Initial Setup

```python
def _func(self, funcBody, *args, **kwargs):
    """Decorator for MLIR functions.
    It cuts the boilerplate code, does the following:
        1. Generates `func.func`
        2. Types translation (numpy arrays -> cute.memref, float -> <f32>, etc.)
        3. Compiles and JITs the MLIR module
        4. Invokes the generated function
        5. Operator overloading (a + b --> arith.addi a, b)
        6. Generates GPU kernel function with GPU module and kernel attributes baked
    """
    # Check if we're already inside an MLIR context
    if ir.Context.current is None:
        pass
    elif ir.InsertionPoint.current is not None:
        # Already inside MLIR generation - just call function directly
        return funcBody(*args, **kwargs)

    function_name = funcBody.__name__
    self.funcBody = funcBody
```

**What's Happening:**
- **Context Check:** If already inside MLIR generation (nested function call), execute directly without re-JIT
- **Bootstrapping:** Store function name and body for later stages

### Frame 2: Extract Compilation Options

```python
    pipeline = kwargs.pop("pipeline", None)
    gpu_module_attrs = kwargs.pop("gpu_module_attrs", {})
    decorator_frame = kwargs.pop("_decorator_frame", None)

    # Disable cache
    no_cache = kwargs.pop("no_cache", False)

    # Always compile(disable cache) and return the result jit_executor
    compile_only = kwargs.pop("compile_only", False)

    if not no_cache and (self.envar.keep_ptx or self.envar.keep_cubin):
        no_cache = True
        self.print_warning("Cache is disabled as user wants to generate PTX/ASM.")

    if not no_cache and compile_only:
        no_cache = True
        self.print_warning("Cache is disabled as user wants to compile only.")
```

**Options Extracted:**
- `pipeline`: Custom MLIR pass pipeline
- `gpu_module_attrs`: GPU module attributes
- `decorator_frame`: Source location for line info
- `no_cache`: Force recompilation
- `compile_only`: Don't execute, just return JIT executor

### Frame 3: Argument Validation

```python
    # Check the number of arguments
    sig = self._check_arg_count(*args, **kwargs)
```

#### `_check_arg_count()` [dsl.py:1451-1496]

```python
def _check_arg_count(self, *args, **kwargs):
    """Validates function signature matches provided arguments"""
    sig = inspect.signature(self.funcBody)
    function_name = self.funcBody.__name__

    try:
        bound_args = sig.bind_partial(*args, **kwargs)
    except TypeError as e:
        raise DSLRuntimeError(f"Argument binding failed for `{function_name}`: {e}")

    # Check required parameters are provided
    for param_name, param in sig.parameters.items():
        if (
            param.default is inspect.Parameter.empty
            and param.kind != inspect.Parameter.VAR_POSITIONAL
            and param.kind != inspect.Parameter.VAR_KEYWORD
            and param_name not in bound_args.arguments
        ):
            raise DSLRuntimeError(
                f"Missing required argument in `{function_name}`: '{param_name}'"
            )

    return sig
```

**Validation Steps:**
1. Get function signature via introspection
2. Bind provided arguments to signature
3. Check all required parameters present
4. Raise descriptive error if mismatch

### Frame 4: Argument Canonicalization

```python
    args_spec = inspect.getfullargspec(funcBody)

    # Canonicalize the input arguments
    canonicalized_args, canonicalized_kwargs = self._canonicalize_args(
        sig, *args, **kwargs
    )
```

#### `_canonicalize_args()` [dsl.py:1141-1199]

```python
def _canonicalize_args(self, sig, *args, **kwargs):
    """
    Convert arguments to canonical form:
    - Positional args with defaults filled in
    - Keyword args in consistent order
    """
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    canonicalized_args = []
    canonicalized_kwargs = {}

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            canonicalized_args.append(bound_args.arguments[param_name])
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            canonicalized_kwargs[param_name] = bound_args.arguments[param_name]

    return canonicalized_args, canonicalized_kwargs
```

**Why Canonicalize?**
- Ensures consistent representation regardless of how user called function
- Fills in default values
- Separates positional from keyword arguments
- Critical for name mangling and caching

### Frame 5: Name Mangling

```python
    # Simple name mangling
    function_name = self.mangle_name(function_name, canonicalized_args, args_spec)
    self.compile_options.apply_envar_settings(self.envar, function_name)
```

#### `mangle_name()` [dsl.py:557-594]

```python
def mangle_name(self, function_name, args, args_spec: inspect.FullArgSpec):
    """Does simple name mangling based on argument types/values"""

    for spec_arg, arg in zip(args_spec.args, args):
        spec_ty = args_spec.annotations.get(spec_arg, None)

        # Skip MLIR types
        if spec_ty != None:
            if issubclass(type(spec_ty), (t.IRValue, t.IRVariadic)):
                continue
            if isinstance(spec_ty, (ir.Type, ir.Value)):
                continue
        if isinstance(arg, (ir.Type, ir.Value, ir.OpResult)):
            continue
        if self._is_tensor_descriptor(arg):
            continue

        # Append type/value info to name
        if inspect.isclass(spec_ty):
            class_name = str(arg).replace("class", "")
            class_name = class_name.replace(" ", "")
            function_name = f"{function_name}_{class_name}"
        elif isinstance(arg, (list, tuple)):
            function_name = f"{function_name}_{'_'.join(map(str, arg))}"
        else:
            function_name = f"{function_name}_{arg}"

    # Clean unwanted characters
    unwanted_chars = r"'-![]#,.<>()\":{}=%?@;"
    translation_table = str.maketrans("", "", unwanted_chars)
    function_name = function_name.translate(translation_table)

    # Remove hex addresses
    function_name = re.sub(r"0x[a-f0-9]{8,16}", "", function_name)
    function_name = re.sub(r"\s+", " ", function_name)
    function_name = function_name.replace(" ", "_")
    function_name = function_name.replace("\n", "_")
    function_name = function_name.replace("/", "_")

    # Truncate to 180 chars (max is 256, leave space)
    function_name = function_name[:180]

    log().info(f"Final mangled function name: {function_name}")
    return function_name
```

**Example:**
```python
@cute.jit
def add(a: Tensor, b: Tensor, alpha: int):
    return a + alpha * b

# Call: add(tensor_a, tensor_b, 5)
# Mangled: add_Tensorgmemf321024_Tensorgmemf321024_5
```

**Why Name Mangling?**
- Enables function overloading (same function, different args)
- Cache key generation
- Debugging (readable kernel names)

### Frame 6: Dispatch to MLIR Generation

```python
    if not self.compile_options.generate_line_info:
        decorator_frame = None

    # Generate MLIR Context and start generating IR
    log().debug(f"Generating MLIR for function '{function_name}'")
    result = self.generate_mlir(
        funcBody,
        canonicalized_kwargs,
        function_name,
        gpu_module_attrs,
        canonicalized_args,
        args_spec,
        pipeline,
        no_cache,
        compile_only,
        frame=decorator_frame,
    )
    return result
```

**Transition:** Now entering MLIR generation phase...

---

## Stage 3: MLIR Generation

### Entry Point: `BaseDSL.generate_mlir()` [dsl.py:1332-1414]

This is where Python code transforms into MLIR IR.

### Frame 1: Context Setup

```python
def generate_mlir(
    self,
    funcBody,
    kwargs,
    function_name,
    gpu_module_attrs,
    args,
    args_spec,
    pipeline,
    no_cache,
    compile_only,
    loc=None,
    frame=None,
):
    """Generate MLIR module and compile it."""
    with ir.Context(), self.get_location(frame):
        try:
            # [Main generation logic]
        finally:
            self.post_compilation_cleanup()
```

**What's Happening:**
- Create fresh MLIR context
- Set source location for debug info
- Ensure cleanup even if compilation fails

### Frame 2: Type Conversion

```python
            # Convert input arguments to MLIR arguments
            exe_args, func_types, adapted_args = self.generate_mlir_function_types(
                funcBody, function_name, args, kwargs, args_spec, compile_only
            )
```

#### `generate_mlir_function_types()` [dsl.py:1201-1272]

```python
def generate_mlir_function_types(
    self, funcBody, function_name, args, kwargs, args_spec, compile_only
):
    """
    Convert Python types to MLIR types
    Returns: (exe_args, func_types, adapted_args)
    """
    exe_args = []
    func_types = []
    adapted_args = []

    for i, (spec_arg, arg) in enumerate(zip(args_spec.args, args)):
        spec_ty = args_spec.annotations.get(spec_arg, None)

        # Use JitArgAdapter registry to convert
        adapter = JitArgAdapterRegistry.find_adapter(arg, spec_ty)

        if adapter:
            # Adapter found - use it
            mlir_types = adapter.get_mlir_type(arg, spec_ty)
            exe_arg = adapter.get_execution_arg(arg, spec_ty)
            adapted = adapter.get_adapted_arg(arg, spec_ty)

            func_types.extend(mlir_types)
            exe_args.extend(exe_arg)
            adapted_args.append(adapted)
        else:
            # No adapter - try default conversion
            mlir_type = self._python_type_to_mlir_type(arg, spec_ty)
            func_types.append(mlir_type)
            exe_args.append(arg)
            adapted_args.append(arg)

    return exe_args, func_types, adapted_args
```

**Type Conversion Examples:**

| Python Type | MLIR Type | Adapter |
|-------------|-----------|---------|
| `int` | `i32` or `i64` | Default |
| `float` | `f32` or `f64` | Default |
| `np.ndarray` | `memref<?xf32>` | NumpyAdapter |
| `Tensor[float32, (M, N)]` | `!cute.tensor<...>` | TensorAdapter |
| `Layout` | `!cute.layout<...>` | LayoutAdapter |
| Custom dataclass | Flattened fields | TreeAdapter |

### Frame 3: Extract Dynamic Arguments

```python
            dynamic_args, dynamic_kwargs = self.extract_dynamic_args(
                funcBody, args, kwargs, args_spec
            )
```

#### `extract_dynamic_args()` [dsl.py:1305-1330]

```python
def extract_dynamic_args(self, funcBody, args, kwargs, args_spec):
    """
    Separate compile-time (constexpr) from runtime (dynamic) arguments.

    Constexpr args are baked into the compiled module.
    Dynamic args can vary between executions without recompilation.
    """
    dynamic_args = []
    dynamic_kwargs = {}

    for i, (spec_arg, arg) in enumerate(zip(args_spec.args, args)):
        if not is_argument_constexpr(arg, args_spec.args, spec_arg, i, funcBody):
            try:
                # Use weak reference to avoid keeping objects alive
                dynamic_args.append(weakref.proxy(arg))
            except TypeError:
                # If arg cannot be weakly referenced (e.g., int, float)
                dynamic_args.append(arg)

    for i, (k, v) in enumerate(kwargs.items()):
        if not is_argument_constexpr(v, args_spec.kwonlyargs[i], k, i, funcBody):
            try:
                dynamic_kwargs[k] = weakref.proxy(v)
            except TypeError:
                dynamic_kwargs[k] = v

    return dynamic_args, dynamic_kwargs
```

**Why Separate Dynamic Args?**
- Dynamic args don't affect compilation hash
- Allows reusing compiled module for different runtime values
- Example: Kernel compiled once, executed with different input tensors

### Frame 4: Generate Original IR Module

```python
            original_function_name = funcBody.__name__

            # Generate original ir module and its hash value.
            module, module_hash, result = self.generate_original_ir(
                ir,
                func,
                funcBody,
                kwargs,
                function_name,
                func_types,
                gpu_module_attrs,
                args,
                args_spec,
                frame=frame,
            )
```

#### `generate_original_ir()` [dsl.py:1861-2020]

This is the most complex function - it builds the actual MLIR module.

```python
def generate_original_ir(
    self,
    ir,
    func,
    funcBody,
    kwargs,
    function_name,
    func_types,
    gpu_module_attrs,
    args,
    args_spec,
    frame=None,
):
    """Generate the MLIR IR module from Python function"""

    # Step 1: Create module with GPU container attribute
    module = ir.Module.create()
    module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

    # Step 2: Build GPU module for kernels
    with self.enter_gpu_module(module), self.get_location(frame):
        self._build_gpu_module(gpu_module_attrs)

    # Step 3: Create function op
    with ir.InsertionPoint(module.body):
        # Create function type
        func_ty = ir.FunctionType.get(func_types, [])

        # Create function operation
        fop = func.FuncOp(
            function_name,
            func_ty,
            ip=ir.InsertionPoint.at_block_begin(module.body)
        )

        # Mark as public
        fop.sym_visibility = ir.StringAttr.get("public")

        # Add entry block
        entry_block = fop.add_entry_block()

        # Step 4: Generate function body
        with ir.InsertionPoint(entry_block):
            # Convert Python args to MLIR Values
            ir_args, ir_kwargs = self.generate_execution_arguments(
                args, kwargs, fop, args_spec
            )

            # Execute user's Python function
            # This is where the magic happens!
            result = funcBody(*ir_args, **ir_kwargs)

            # Add return operation
            func.ReturnOp([])

    # Step 5: Verify module
    if not module.operation.verify():
        raise DSLRuntimeError("Generated MLIR module is invalid")

    # Step 6: Compute hash for caching
    module_hash = self.get_module_hash(module, function_name, args, kwargs)

    return module, module_hash, result
```

**Critical Insight: Python Execution Generates MLIR**

When `funcBody(*ir_args, **ir_kwargs)` executes:
- Python operators are overloaded
- `a + b` → calls `__add__` → generates `arith.addi` op
- `tensor[i]` → calls `__getitem__` → generates `cute.load` op
- Control flow (`if`, `for`) generates MLIR ops via preprocessor

Example:
```python
@cute.jit
def add(a: Tensor, b: Tensor):
    return a + b

# When called with Tensor arguments:
# 1. `a` and `b` become MLIR Values
# 2. `a + b` calls Tensor.__add__(a, b)
# 3. __add__ generates: %result = arith.addf %a, %b
# 4. Return generates: func.return %result
```

#### `generate_execution_arguments()` [dsl.py:747-825]

```python
def generate_execution_arguments(self, args, kwargs, fop, args_spec):
    """
    Convert function arguments to MLIR block arguments.

    This maps Python arguments to MLIR SSA values.
    """
    fop_args = list(fop.arguments)
    iv_block_args = []  # Variadic args

    ir_args = []
    for i, (spec_arg, arg) in enumerate(zip(args_spec.args, args)):
        spec_ty = args_spec.annotations.get(spec_arg, None)

        # Check if argument has special handling
        ir_arg = self._generate_execution_arguments_for_known_types(
            arg, spec_ty, spec_arg, i, fop_args, iv_block_args
        )

        if ir_arg is not None:
            ir_args.append(ir_arg)
        else:
            # Default: use block argument directly
            ir_args.append(fop_args[i])

    # Handle kwargs similarly
    ir_kwargs = {}
    for k, v in kwargs.items():
        # ... similar processing

    return ir_args, ir_kwargs
```

### Frame 5: Caching and Compilation

```python
            # dryrun is used to only generate IR
            if self.envar.dryrun:
                return result

            if (
                no_cache
                or module_hash not in self.jit_cache
                or self.jit_cache[module_hash].capi_func is None
            ):
                # no cache or cache miss, do ir generation/compilation/jit engine
                jit_function = self.compile_and_cache(
                    module,
                    module_hash,
                    function_name,
                    pipeline,
                    args_spec,
                    no_cache,
                    full_args=args,
                    full_kwargs=kwargs,
                    dynamic_args=dynamic_args,
                    dynamic_kwargs=dynamic_kwargs,
                    original_function_name=original_function_name,
                )
            else:
                # cache hit
                log().info(
                    "JIT cache hit IN-MEMORY function=[%s] module_hash=[%s]",
                    function_name,
                    module_hash,
                )
                jit_function = self.jit_cache[module_hash]
```

**Caching Strategy:**
1. Compute hash of module IR + compile options + constexpr args
2. Check in-memory cache (`self.jit_cache`)
3. If miss, compile and cache result
4. If hit, reuse compiled function

**Cache Key Components:**
- MLIR module IR (string representation)
- Compile options (opt level, arch, etc.)
- Constexpr argument values
- Environment variables

### Frame 6: Execution or Return

```python
        # If compile_only is set, bypass execution return the jit_executor directly
        if compile_only:
            return jit_function

        # Run the compiled program
        jit_function.run_compiled_program(exe_args)

        return result
```

**Two Paths:**
1. **compile_only=True**: Return `JitExecutor` without running (used by `cute.compile`)
2. **compile_only=False**: Execute immediately and return result

---

## Stage 4: Kernel Generation (for @cute.kernel)

### Entry Point: `CutlassBaseDSL._kernel_helper()` [cutlass.py:652-747]

When `@cute.kernel` is used instead of `@cute.jit`, a specialized kernel generation path is taken.

### Frame 1: Kernel Generation Helper Class

```python
def _kernel_helper(self, funcBody, *args, **kwargs):
    class _CutlassIrKernelGenHelper(BaseDSL._KernelGenHelper):
        def __init__(self, dsl: CutlassBaseDSL):
            super().__init__()
            self.dsl = dsl
            self.dsl._reset_smem_tracking()
```

**Design Pattern:**
- Inner class defines kernel-specific IR generation
- Implements abstract methods from `_KernelGenHelper`
- Allows different DSLs to customize kernel generation

### Frame 2: Kernel Function Op Generation

```python
        def generate_func_op(self, arg_types, arg_attrs, kernel_name, loc=None):
            super().generate_func_op(arg_types, arg_attrs, kernel_name)

            # Create CUDA kernel op (not standard func op)
            self.func_op = cuda_dialect.KernelOp(
                kernel_name,
                ir.FunctionType.get(arg_types, []),
                loc=loc
            )

            self.arg_types = arg_types

            # Set CUDA-specific attributes
            self.func_op.attributes["cu_attrs"] = ir.DictAttr.get(
                {
                    str(cuda_dialect.CUFunctionAttribute.non_portable_cluster_size_allowed):
                        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1),
                    str(cuda_dialect.CUFunctionAttribute.max_dynamic_shared_size_bytes):
                        cuda_dialect.DevMaxSharedMemoryOptinAttr.get(),
                }
            )

            if arg_attrs is not None:
                log().debug(arg_attrs)
                self.func_op.arg_attrs = arg_attrs

            return self.func_op
```

**Key Differences from @cute.jit:**
- Uses `cuda_dialect.KernelOp` instead of `func.FuncOp`
- Sets CUDA-specific attributes:
  - `non_portable_cluster_size_allowed`: Enable flexible cluster sizing
  - `max_dynamic_shared_size_bytes`: Opt-in to max shared memory
- Kernel ops live in GPU module, not top-level module

### Frame 3: Kernel Return Op

```python
        def generate_func_ret_op(self, loc=None, ip=None):
            return cuda_dialect.ReturnOp([], loc=loc, ip=ip)
```

**Why Different Return Op?**
- `cuda_dialect.ReturnOp` has different semantics than `func.ReturnOp`
- Kernels cannot return values (void return)
- Return op triggers synchronization semantics

### Frame 4: Entry Block Generation

```python
        def get_func_body_start(self):
            assert self.func_op is not None, "Invalid func_op is not expected!"
            arg_locs = [self.func_op.operation.location for _ in self.arg_types]
            return self.func_op.add_entry_block(arg_locs=arg_locs)
```

**What's Happening:**
- Create entry block for kernel body
- Assign source locations to each argument
- Return insertion point for body generation

### Frame 5: Launch Op Generation

This is the most complex part - generating the kernel launch operation.

```python
        def generate_launch_op(self, *args, **kwargs):
            # Extract args and do validation
            kernelSym = kwargs.get("kernelSym", None)
            kernelOperands = kwargs.get("kernelOperands", None)
            requiredArgs = kwargs.get("requiredArgs", None)
            loc = kwargs.get("loc", None)

            assert kernelSym is not None, "kernelSym being None is not expected!"
            assert requiredArgs is not None, "requiredArgs being None is not expected!"
            assert kernelOperands is not None, "kernelOperands being None is not expected!"
            assert isinstance(requiredArgs.config, BaseDSL.LaunchConfig), (
                f"Expect LaunchConfig for @kernel, but got {type(requiredArgs.config)}"
            )

            cfg = requiredArgs.config

            # Auto-calculate shared memory usage
            smem_usage = self.dsl._get_smem_usage()
            if any(not isinstance(x, int) for x in [cfg.smem, smem_usage]):
                pass  # cannot compare dynamic value inside kernel to launch op in py
            elif cfg.auto_smem:
                cfg.smem = smem_usage
            elif smem_usage > cfg.smem:
                warnings.warn(
                    f"Potential error: specified kernel launch smem bytes "
                    f"({cfg.smem}) is smaller than kernel usage ({smem_usage})!",
                    UserWarning,
                )
            cfg.smem = const(cfg.smem)

            # Handle async dependencies
            async_deps = cfg.async_deps
            if not isinstance(cfg.async_deps, (list, tuple)):
                async_deps = [cfg.async_deps]

            # Generate launch operation
            CutlassBaseDSL.cuda_launch_func(
                async_deps,
                kernelSym,
                *cfg.grid,
                *cfg.block,
                kernelOperands,
                **dict(
                    zip(
                        ("cluster_size_x", "cluster_size_y", "cluster_size_z"),
                        tuple(cfg.cluster),
                    )
                ),
                dynamic_shared_memory_size=cfg.smem,
                use_pdl=cfg.use_pdl,
                loc=loc,
            )
            return None
```

#### Launch Configuration: `BaseDSL.LaunchConfig` [dsl.py:887-936]

```python
@dataclass
class LaunchConfig:
    """
    Configuration for CUDA kernel launch.

    Specifies grid/block/cluster dimensions, shared memory, and streams.
    """
    # Grid dimensions (number of thread blocks)
    grid: tuple[int, int, int] = (1, 1, 1)

    # Block dimensions (threads per block)
    block: tuple[int, int, int] = (1, 1, 1)

    # Cluster dimensions (blocks per cluster) - Hopper+
    cluster: tuple[int, int, int] = (1, 1, 1)

    # Dynamic shared memory size (bytes)
    smem: int = 0

    # Auto-calculate shared memory based on kernel usage
    auto_smem: bool = True

    # Async dependencies (CUDA streams/events)
    async_deps: Union[list, tuple, Any] = None

    # Use PDL (Programmatic Dependent Launch)
    use_pdl: bool = False

    def __post_init__(self):
        # Ensure dimensions are tuples
        self.grid = self._ensure_3d_tuple(self.grid)
        self.block = self._ensure_3d_tuple(self.block)
        self.cluster = self._ensure_3d_tuple(self.cluster)
```

**Usage Example:**
```python
@cute.kernel
def my_kernel(a: Tensor, b: Tensor):
    # Kernel body
    pass

# Launch with configuration
config = cute.LaunchConfig(
    grid=(128, 1, 1),      # 128 thread blocks
    block=(256, 1, 1),     # 256 threads per block
    cluster=(1, 1, 1),     # No clustering
    smem=48 * 1024,        # 48KB shared memory
)
my_kernel(a, b, config=config)
```

#### `cuda_launch_func()` [cutlass.py:512-606]

```python
@staticmethod
def cuda_launch_func(
    async_deps,
    kernel,
    gridSizeX,
    gridSizeY,
    gridSizeZ,
    blockSizeX,
    blockSizeY,
    blockSizeZ,
    args,
    cluster_size_x=1,
    cluster_size_y=1,
    cluster_size_z=1,
    dynamic_shared_memory_size=0,
    use_pdl=False,
    loc=None,
):
    """Generate CUDA launch operation in MLIR"""

    # Create launch config type
    launch_config_type = cuda_dialect.LaunchConfigType.get()

    # Build config operands
    config_operands = [
        const(gridSizeX), const(gridSizeY), const(gridSizeZ),
        const(blockSizeX), const(blockSizeY), const(blockSizeZ),
        const(cluster_size_x), const(cluster_size_y), const(cluster_size_z),
        const(dynamic_shared_memory_size),
    ]

    # Create config value
    launch_config = cuda_dialect.CreateLaunchConfigOp(
        launch_config_type,
        config_operands,
        loc=loc,
    ).result

    # Generate launch_ex operation
    launch_op = cuda_dialect.launch_ex(
        async_deps if async_deps else [],
        launch_config,
        kernel,
        args,
        use_pdl=use_pdl,
        loc=loc,
    )

    return launch_op
```

**Generated MLIR (conceptual):**
```mlir
%config = cuda.create_launch_config(
    grid: (128, 1, 1),
    block: (256, 1, 1),
    cluster: (1, 1, 1),
    smem: 49152
)

cuda.launch_ex %config, @my_kernel(%arg0, %arg1) : (!cute.tensor<...>, !cute.tensor<...>)
```

### Frame 6: Kernel Launcher Wrapper

```python
    return KernelLauncher(
        self,
        lambda: _CutlassIrKernelGenHelper(self),
        funcBody,
        *args,
        **kwargs,
    )
```

#### `KernelLauncher` Class [dsl.py:1658-1886]

```python
class KernelLauncher:
    """
    Wrapper that encapsulates kernel generation and launch logic.

    Allows kernels to be called like regular functions:
        my_kernel(args, config=LaunchConfig(...))
    """

    def __init__(self, dsl, kernel_gen_helper_factory, funcBody, *args, **kwargs):
        self.dsl = dsl
        self.kernel_gen_helper_factory = kernel_gen_helper_factory
        self.funcBody = funcBody
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """
        When kernel is called, this:
        1. Extracts launch config from kwargs
        2. Validates config
        3. Generates kernel IR
        4. Generates launch op
        5. Returns launch op result
        """
        # Extract config
        config = kwargs.pop("config", None)
        if config is None:
            raise DSLRuntimeError("Kernel requires 'config' parameter")

        # Validate config
        if not isinstance(config, BaseDSL.LaunchConfig):
            raise DSLRuntimeError(f"Expected LaunchConfig, got {type(config)}")

        # Store config for kernel generation
        self.kwargs["requiredArgs"] = SimpleNamespace(config=config)

        # Generate kernel and launch op
        return self.dsl._generate_kernel_and_launch(
            self.funcBody,
            self.kernel_gen_helper_factory,
            *args,
            **self.kwargs,
        )
```

**Key Insight:**
- Kernel functions return `KernelLauncher`, not direct result
- `KernelLauncher.__call__` handles actual IR generation
- Allows natural syntax: `kernel(args, config=cfg)`

---

## Stage 5: MLIR Compilation Pipeline

### Entry Point: `Compiler.compile_and_jit()` [compiler.py:176-193]

After MLIR module is generated, it must be compiled to executable code.

### Frame 1: Compile Pass Pipeline

```python
def compile_and_jit(
    self,
    module,
    pipeline: str,
    shared_libs: Sequence[str] = (),
    opt_level: int = 2,
    cuda_toolkit: str = "",
    arch: str = "",
):
    """Compiles and jits the module."""

    # Step 1: Run compilation passes
    self.compile(
        module,
        pipeline,
        cuda_toolkit,
        arch,
    )

    # Step 2: Create JIT execution engine
    return self.jit(module, opt_level, shared_libs)
```

### Frame 2: Pass Manager Execution

#### `Compiler.compile()` [compiler.py:136-164]

```python
def compile(
    self,
    module,
    pipeline: str,
    cuda_toolkit: str = "",
    arch: str = "",
    enable_verifier=False,
):
    """Compiles the module by invoking the pipeline."""
    try:
        # Parse and run pass pipeline
        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)
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

    # Run post-compile hook if registered
    if self._post_compile_hook:
        self._post_compile_hook(module)
```

**What's Happening:**
1. Parse pipeline string into pass sequence
2. Enable verifier if requested
3. Run passes on module
4. Catch compilation errors and enhance with context
5. Run post-compile hooks (e.g., TVM FFI attachment)

### Pipeline Structure

#### Default Pipeline [compiler.py:262]

```python
def _get_default_pipeline(self):
    return f"builtin.module(cute-to-nvvm{{cubin-format=bin cubin-chip={self.arch} ...}})"
```

**Pipeline Breakdown:**
```
builtin.module(
    cute-to-nvvm{
        cubin-format=bin           # Output CUBIN binary
        cubin-chip=sm_90a          # Target architecture
        cuda-toolkit=/path/cuda    # CUDA toolkit path
        opt-level=3                # Optimization level
        cubin-name=kernel.cubin    # Output filename
        ptx-name=kernel.ptx        # PTX filename (if enabled)
        enable-assertions=false    # Runtime assertions
        generate-line-info=false   # Debug line info
    }
)
```

### Pass Sequence (Inside cute-to-nvvm)

The `cute-to-nvvm` pass is a mega-pass that runs multiple transformations:

1. **Cute Dialect Lowering**
   - `cute-to-llvm`: Lower CuTe operations to LLVM
   - Layout calculations → load/store operations
   - Tensor operations → memory operations

2. **GPU Dialect Lowering**
   - `gpu-to-nvvm`: Lower GPU dialect to NVVM
   - Thread indexing → `%tid = nvvm.read.ptx.sreg.tid.x`
   - Block indexing → `%bid = nvvm.read.ptx.sreg.ctaid.x`
   - Barriers → `nvvm.barrier0`

3. **Standard Dialect Lowering**
   - `arith-to-llvm`: Arithmetic operations to LLVM
   - `math-to-llvm`: Math functions to LLVM intrinsics
   - `func-to-llvm`: Function ops to LLVM functions

4. **NVVM to PTX**
   - `nvvm-to-ptx`: Translate NVVM dialect to PTX assembly
   - Uses NVVM compiler library
   - Performs register allocation, instruction scheduling

5. **PTX to CUBIN**
   - `ptx-to-cubin`: Assemble PTX to CUBIN binary
   - Uses `ptxas` from CUDA toolkit
   - Final machine code for GPU

### Error Handling

#### `_process_error()` [compiler.py:108-134]

```python
def _process_error(self, error_msg):
    """
    Extract NVVM errors and IR context from compilation errors.
    """
    nvvm_error = None
    ir_msg = None

    # Look for NVVM error section
    if "NVVM compile failed:" in error_msg:
        # Extract NVVM error
        nvvm_start = error_msg.find("NVVM compile failed:")
        nvvm_section = error_msg[nvvm_start:]
        nvvm_lines = nvvm_section.split("\n")
        nvvm_error = "\n".join(nvvm_lines[:20])  # First 20 lines

    # Look for IR section
    if "IR:" in error_msg:
        ir_start = error_msg.find("IR:")
        ir_section = error_msg[ir_start + 3:]
        ir_lines = ir_section.split("\n")

        # Truncate if too long
        if len(ir_lines) > 10:
            ir_msg = "\n".join(ir_lines[:5] + ["  ..."] + ir_lines[-5:])
        else:
            ir_msg = ir_section

    return nvvm_error, ir_msg
```

**Enhanced Error Example:**
```
CompilationError: NVVM compilation failed

NVVM Error:
  ptxas error: Entry function '_cuda_kernel_add_42' uses too many registers

IR Context:
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  %1 = llvm.load %arg1 : !llvm.ptr -> f32
  %2 = llvm.fadd %0, %1 : f32
  ...

Suggestions:
  - Reduce register pressure by:
    - Decreasing thread block size
    - Simplifying computation
    - Using shared memory instead of registers
```

---

## Stage 6: PTX/CUBIN Generation

### PTX Generation

#### Entry Point: NVVM Compiler in cute-to-nvvm pass

The NVVM compiler translates NVVM IR to PTX assembly.

### Frame 1: NVVM IR to PTX

**Input: NVVM IR**
```llvm
define void @kernel_add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bdim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

  %block_offset = mul i32 %bid, %bdim
  %idx = add i32 %block_offset, %tid

  %ptr0 = getelementptr float, ptr %arg0, i32 %idx
  %ptr1 = getelementptr float, ptr %arg1, i32 %idx
  %ptr2 = getelementptr float, ptr %arg2, i32 %idx

  %val0 = load float, ptr %ptr0
  %val1 = load float, ptr %ptr1
  %result = fadd float %val0, %val1
  store float %result, ptr %ptr2

  ret void
}
```

**Output: PTX Assembly**
```ptx
.version 8.0
.target sm_90a
.address_size 64

.visible .entry kernel_add(
    .param .u64 kernel_add_param_0,
    .param .u64 kernel_add_param_1,
    .param .u64 kernel_add_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;

    // Calculate global thread index
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;

    // Load parameters
    ld.param.u64 %rd1, [kernel_add_param_0];
    ld.param.u64 %rd2, [kernel_add_param_1];
    ld.param.u64 %rd3, [kernel_add_param_2];

    // Calculate addresses
    mul.wide.s32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    // Perform computation
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

    ret;
}
```

**Optimizations Performed:**
- Register allocation (`.reg` declarations)
- Instruction scheduling for throughput
- Coalesced memory access patterns
- Predicate optimization for branching

### Frame 2: PTX to CUBIN

#### Entry Point: `ptxas` (CUDA Toolkit)

The PTX assembler converts PTX to binary CUBIN.

**Command:**
```bash
ptxas \
  --gpu-name=sm_90a \
  --output-file=kernel.cubin \
  kernel.ptx
```

**CUBIN Structure:**
```
ELF Header
  ├─ .text.kernel_add      # Machine code
  ├─ .nv.info.kernel_add   # Kernel metadata
  │   ├─ Register count: 24
  │   ├─ Shared memory: 0 bytes
  │   ├─ Constant memory: 0 bytes
  │   └─ Max threads: 1024
  ├─ .nv.constant0         # Constant data
  └─ .nv.shared            # Shared memory layout
```

**Binary Embedding in MLIR:**

After CUBIN generation, the binary is embedded back into the MLIR module:

```mlir
gpu.module @kernels {
  gpu.binary @kernel_add [
    #gpu.object<
      #nvvm.target<chip = "sm_90a">,
      "\x7fELF\x02\x01\x01\x00..."  // Raw CUBIN bytes
    >
  ]
}
```

### PTX/CUBIN Dump Options

#### `KeepPTX` and `KeepCUBIN` Options [compiler.py:38-68]

```python
@dataclass
class KeepPTX(BooleanBasedFileDumpOption):
    """Dump PTX assembly to file"""
    pass

@dataclass
class KeepCUBIN(BooleanBasedFileDumpOption):
    """Dump CUBIN binary to file"""
    pass
```

**Usage:**
```python
@cute.compile[cute.KeepPTX, cute.KeepCUBIN]
def my_kernel(...):
    pass
```

**Output Files:**
```
/tmp/cute_dumps/
├─ my_kernel_hash123.ptx    # PTX assembly
└─ my_kernel_hash123.cubin  # CUBIN binary
```

---

## Stage 7: Kernel Binding & Launching

After compilation, the kernel must be loaded and launched. CuTeDSL supports two modes:

### Mode 1: Legacy JIT Executor (MLIR ExecutionEngine)

#### Location: [python/CuTeDSL/cutlass/base_dsl/jit_executor.py](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py)

### Frame 1: Create JIT Execution Engine

#### `Compiler.jit()` [compiler.py:166-174]

```python
def jit(self, module, opt_level: int = 2, shared_libs: Sequence[str] = ()):
    """Wraps the module in a JIT execution engine."""

    # Check CUDA dependencies once per process
    self._check_cuda_dependencies_once(shared_libs)

    # Create ExecutionEngine
    return self.execution_engine.ExecutionEngine(
        module,
        opt_level=opt_level,
        shared_libs=shared_libs
    )
```

**What's an ExecutionEngine?**
- MLIR's JIT compiler
- Uses LLVM's ORC JIT infrastructure
- Loads compiled module into memory
- Resolves function symbols
- Provides C API for execution

### Frame 2: Extract CUBIN from Module

#### `load_kernels_from_ir_module()` [jit_executor.py:100-132]

```python
def load_kernels_from_ir_module(module, kernel_info) -> list[CudaModuleAndKernel]:
    """Loads all kernels from the IR module that match the given set of symbols."""

    if not kernel_info:
        return []  # no modules

    kernel_symbols = tuple(kernel_info.keys())
    kernel_modules = collections.OrderedDict()

    for sym in kernel_symbols:
        log().debug(f"Loading CUDA module for symbol: {sym}")

        def walk_callback(sym, func_sym, cubin_data):
            if sym in kernel_modules:
                log().debug(f"Skipping already loaded symbol: {sym}")
                return

            # Load CUBIN via CUDA driver
            cubin_module = cuda_helpers.load_library_data(cubin_data)

            # Get kernel function pointer
            kernel = cuda_helpers.get_library_kernel(cubin_module, func_sym)

            # Setup kernel attributes
            attrs = dict(kernel_info[sym])
            if cuda_helpers.get_driver_version() >= 11080:
                attrs[
                    cuda_helpers.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
                ] = 1

            kernel_modules[sym] = CudaModuleAndKernel(
                sym, cubin_module, kernel, attrs
            )

        walk_module_and_get_cubin_data(module, sym, walk_callback)

    return list(kernel_modules.values())
```

#### `walk_module_and_get_cubin_data()` [jit_executor.py:66-98]

```python
def walk_module_and_get_cubin_data(module, sym, callback):
    """Walk IR module and extract CUBIN data for given symbol"""

    def walk_gpu_binary_op(op):
        # Look for gpu.binary operations
        if op.operation.name == "gpu.binary":
            binary_sym = op.attributes["sym_name"]

            if str(binary_sym) == f'"{sym}"':
                # Found matching binary
                objects = op.attributes["objects"]

                for obj in objects:
                    # Extract CUBIN data
                    cubin_data = obj.properties["object"]

                    # Unescape special characters
                    cubin_data = cubin_data.replace("\\22", '"')
                    cubin_data = cubin_data.replace("\\5C", "\\")

                    # Call callback with data
                    callback(sym, binary_sym, cubin_data)

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_gpu_binary_op)
```

**What's Happening:**
1. Walk MLIR module IR
2. Find `gpu.binary` operations
3. Extract embedded CUBIN bytes
4. Unescape string encoding
5. Pass to callback for loading

### Frame 3: Load CUBIN via CUDA Driver

#### `cuda_helpers.load_library_data()` [runtime/cuda.py]

```python
def load_library_data(cubin_data: str) -> cuda.CUmodule:
    """Load CUBIN data into CUDA module"""

    # Convert string to bytes
    cubin_bytes = cubin_data.encode('latin-1')

    # Load module
    module = cuda.CUmodule()
    cuda.cuModuleLoadData(module, cubin_bytes)

    return module
```

#### `cuda_helpers.get_library_kernel()` [runtime/cuda.py]

```python
def get_library_kernel(module: cuda.CUmodule, kernel_name: str) -> cuda.CUfunction:
    """Get kernel function pointer from module"""

    kernel = cuda.CUfunction()
    cuda.cuModuleGetFunction(kernel, module, kernel_name.encode())

    return kernel
```

**CUDA Driver API Calls:**
- `cuModuleLoadData`: Load CUBIN into GPU memory
- `cuModuleGetFunction`: Get function pointer by name

### Frame 4: Create Device Context

#### `JitModule.get_device_execute_context()` [jit_executor.py:397-432]

```python
def get_device_execute_context(self, device_id=None):
    """Get or create execution context for device"""

    if device_id is None:
        # Get current device
        device_id = cuda_helpers.get_current_device()

    if device_id not in self.execute_contexts:
        # Create primary context for device
        device_ctx = cuda_helpers.DevicePrimaryContext(device_id)

        # Set kernel attributes for all kernels
        for kernel_info in self.cuda_modules:
            cuda_helpers.set_kernel_attributes(
                kernel_info.kernel,
                kernel_info.attrs
            )

        # Create execution context
        self.execute_contexts[device_id] = JitExecuteContext(
            device=device_id,
            device_ctx=device_ctx,
            kernel_ptrs=[k.kernel for k in self.cuda_modules],
        )

    return self.execute_contexts[device_id]
```

**Context Management:**
- One context per device
- Lazy creation on first use
- Primary context (default CUDA context)
- Kernel attributes set once per context

### Frame 5: Kernel Launch

#### `JitExecutor.run_compiled_program()` [jit_executor.py:492-508]

```python
def run_compiled_program(self, exe_args):
    """Execute the compiled kernel"""

    # Get device context
    execute_ctx = self.jit_module.get_device_execute_context()

    # Pack arguments into C array
    packed_args = (ctypes.c_void_p * len(exe_args))()
    for i, arg in enumerate(exe_args):
        packed_args[i] = ctypes.cast(
            ctypes.pointer(arg),
            ctypes.c_void_p
        )

    # Call main function via C API
    # This invokes the host wrapper which launches kernels
    result = self.jit_module.capi_func(packed_args)

    # Check for CUDA errors
    if result != 0:
        error_name = cuda_helpers.get_error_name(result)
        raise DSLCudaRuntimeError(result, error_name)

    return result
```

**Execution Flow:**
1. Get device context (creates if needed)
2. Pack arguments as C void pointers
3. Call host function via ExecutionEngine
4. Host function launches kernel(s) via `cudaLaunchKernel`
5. Check CUDA error code
6. Synchronize if needed

### Host Function (Generated by MLIR)

The MLIR compilation generates a host wrapper function:

```c
// Generated by MLIR
extern "C" int _host_wrapper(void** args) {
    // Unpack arguments
    float* arg0 = *(float**)args[0];
    float* arg1 = *(float**)args[1];
    float* arg2 = *(float**)args[2];

    // Get launch config
    dim3 grid(128, 1, 1);
    dim3 block(256, 1, 1);
    size_t smem = 0;

    // Launch kernel
    void* kernel_args[] = {&arg0, &arg1, &arg2};
    cudaError_t result = cudaLaunchKernel(
        _cuda_kernel_add,  // Kernel function
        grid, block, smem,
        0,  // Stream
        kernel_args
    );

    return (int)result;
}
```

---

### Mode 2: CUDA Dialect Executor (Modern)

#### Location: [python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py](../python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py)

This is the modern, context-free execution path.

### Frame 1: CUDA Library Loading

#### `CudaDialectJitCompiledFunction._load_cuda_library()` [cuda_jit_executor.py:213-261]

```python
def _load_cuda_library(self):
    """Loads the CUDA library from the engine."""

    # Get init and load functions from execution engine
    cuda_init, cuda_load_to_device = self._get_cuda_init_and_load()

    # Allocate library handle
    library = ctypes.c_void_p()
    pointer_to_library = ctypes.pointer(library)
    pointer_to_pointer_to_library = ctypes.pointer(pointer_to_library)
    err = ctypes.c_int32(0)
    pointer_to_err = ctypes.pointer(err)

    # Call cuda_init
    cuda_init_args = [pointer_to_pointer_to_library, pointer_to_err]
    packed_args = (ctypes.c_void_p * len(cuda_init_args))()
    for i in range(len(cuda_init_args)):
        packed_args[i] = ctypes.cast(cuda_init_args[i], ctypes.c_void_p)
    cuda_init(packed_args)

    if err.value != 0:
        error_code = err.value
        error_name = cuda_runtime.cudaGetErrorName(
            cuda_runtime.cudaError_t(error_code)
        )
        raise DSLCudaRuntimeError(error_code, error_name)

    # Load to each device
    device_id = ctypes.c_int32(0)
    pointer_to_device_id = ctypes.pointer(device_id)

    cuda_load_args = [pointer_to_library, pointer_to_device_id, pointer_to_err]
    packed_args = (ctypes.c_void_p * len(cuda_load_args))()
    for i, arg in enumerate(cuda_load_args):
        packed_args[i] = ctypes.cast(arg, ctypes.c_void_p)

    for dev in range(self.num_devices):
        device_id.value = dev
        cuda_load_to_device(packed_args)
        if err.value != 0:
            raise DSLCudaRuntimeError(
                err.value,
                cuda_runtime.cudaGetErrorName(cuda_runtime.cudaError_t(err.value)),
            )

    return [cuda_runtime.cudaLibrary_t(library.value)]
```

**CUDA Dialect Runtime Functions:**

These are generated by MLIR's CUDA dialect lowering:

```c
// Generated by MLIR
extern "C" void cuda_init(void** library, int* error) {
    // Initialize CUDA library from embedded CUBIN
    cudaLibrary_t lib;
    *error = cudaLibraryLoadFromFile(&lib, "embedded.cubin");
    *library = lib;
}

extern "C" void cuda_load_to_device(void* library, int* device, int* error) {
    // Load library to specific device
    *error = cudaLibraryLoadToDevice(library, *device);
}
```

### Frame 2: Context-Free Execution

#### `CudaDialectJitCompiledFunction.to()` [cuda_jit_executor.py:263-289]

```python
def to(self, device=None) -> JitExecutor:
    """Returns an executable function bound to the given device.

    Since CudaJitCompiledFunction uses CUDA libraries, which are context free,
    binding to a device is not necessary and the device is ignored.
    """
    super()._validate_engine()

    with self._executor_lock:
        # Ensure modules are loaded
        if self.jit_module is None or self.jit_module.is_unloaded():
            cuda_library = self._load_cuda_library()
            self.jit_module = CudaDialectJitModule(
                self.engine,
                self.capi_func,
                self.args_spec,
                cuda_library,
            )

        # Return executor (no device binding needed)
        return JitExecutor(
            self.jit_module,
            self.function_name,
            device,
            self.jit_time_profiling,
        )
```

**Key Advantage:**
- No explicit device context management
- CUDA runtime handles device selection
- Can run on any device without rebinding

### Mode Comparison

| Feature | Legacy (JIT Executor) | Modern (CUDA Dialect) |
|---------|----------------------|----------------------|
| **Context** | Device-specific | Context-free |
| **Module Loading** | `cuModuleLoadData` | `cudaLibraryLoad` |
| **Kernel Launch** | Manual via driver API | CUDA dialect runtime |
| **Multi-Device** | Requires per-device contexts | Automatic |
| **Performance** | Slightly lower overhead | Slightly higher overhead |
| **Flexibility** | Fine-grained control | Simpler API |

---

## cute.compile Function

### Location: [python/CuTeDSL/cutlass/base_dsl/compiler.py](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L556-L653)

The `cute.compile` function allows explicit compilation without immediate execution.

### Frame 1: CompileCallable Class

```python
class CompileCallable:
    def __init__(self, options=None):
        def preprocess_options(option):
            if type(option) is type and issubclass(
                option, (BooleanCompileOption, BooleanBasedFileDumpOption, EnableTVMFFI)
            ):
                # Automatically creates a True instance of the option
                return option(True)
            elif isinstance(option, tuple):
                return tuple(preprocess_options(opt) for opt in option)
            return option

        self._compile_options = CompileOptions(preprocess_options(options))

    def __getitem__(self, options):
        """
        Get a new CompileCallable object with the specified options.

        Enables syntax: cute.compile[option1, option2]
        """
        new_callable_with_options = CompileCallable(options)
        return new_callable_with_options

    def __call__(self, *args, **kwargs):
        return self._compile(*args, **kwargs)
```

**Usage Examples:**

```python
# Option 1: Compile with options
jit_fn = cute.compile[cute.OptLevel(3), cute.KeepPTX](my_func, arg1, arg2)

# Option 2: Compile without options
jit_fn = cute.compile(my_func, arg1, arg2)

# Option 3: Chaining syntax
compiler = cute.compile[cute.OptLevel(3)]
jit_fn = compiler(my_func, arg1, arg2)
```

### Frame 2: Compile Implementation

```python
def _compile(self, func, *args, **kwargs):
    """
    This function is used to compile a `cute.jit` decorated function.
    It will process the compile options and input parameters, do explicit compilation
    and return the jit executor.
    """
    if func is None:
        raise DSLRuntimeError("Function is not set or invalid.")

    if not callable(func):
        raise DSLRuntimeError("Object is not callable.")

    # Set compile-only mode
    kwargs["compile_only"] = True
    kwargs["no_cache"] = True

    # Handle different function types
    if inspect.isfunction(func):
        # regular function
        pass
    elif inspect.ismethod(func):
        # if it's a method, add the instance to the first argument
        args = [func.__self__] + list(args)
        func = func.__func__
    elif (
        inspect.isclass(type(func))
        and hasattr(func, "__call__")
        and hasattr(func.__call__, "__func__")
    ):
        # If it's a class instance, get the class's __call__ method
        args = [func] + list(args)
        func = func.__call__.__func__
    else:
        raise DSLRuntimeError(
            "Invalid function type, only function, method and module are supported"
        )

    # If it's a wrapped function created by jit decorator, get the original function
    if hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    # Lazy initialization of DSL object
    from .dsl import BaseDSL
    BaseDSL._lazy_initialize_dsl(func)

    if not hasattr(func, "_dsl_object"):
        raise DSLRuntimeError("Function is not decorated with jit decorator.")

    # Process compile options
    options = kwargs.pop("options", None)
    if isinstance(options, str) and len(options) == 0:
        options = None

    if options is not None and isinstance(options, str):
        compile_options = _parse_compile_options_from_str(options)
    else:
        compile_options = self._compile_options

    func._dsl_object.compile_options = compile_options

    # Run preprocessor if enabled
    fcn_ptr = func._dsl_object._preprocess_and_execute(func)

    # Pass decorator frame for line info
    if hasattr(func, "_decorator_frame"):
        kwargs["_decorator_frame"] = func._decorator_frame

    # Call _func with compile_only=True
    return func._dsl_object._func(fcn_ptr, *args, **kwargs)
```

**Key Points:**
- Sets `compile_only=True` to skip execution
- Disables caching (always recompile)
- Returns `JitExecutor` object
- Can be called later with different arguments

### Frame 3: JitExecutor Usage

```python
# Compile once
add_jit = cute.compile(add_func, tensor_a, tensor_b)

# Execute multiple times
result1 = add_jit.run(tensor_c, tensor_d)
result2 = add_jit.run(tensor_e, tensor_f)

# Or use as callable
result3 = add_jit(tensor_g, tensor_h)
```

---

## Complete Call Graph Summary

### For @cute.jit

```
User Code: add(a, b)
    ↓
@cute.jit decorator intercepts
    ↓
BaseDSL.jit_runner()
    ↓
func.__wrapped__ stored
    ↓
[On call] func(*args)
    ↓
BaseDSL._preprocess_and_execute()
    ├─ [If enabled] DSLPreprocessor.transform()
    │   ├─ Parse AST
    │   ├─ Transform for/if
    │   ├─ Generate yield ops
    │   └─ Return transformed function
    └─ Return function pointer
    ↓
BaseDSL._func(funcBody, *args, **kwargs)
    ├─ Check if in MLIR context
    ├─ Extract options
    ├─ Validate arguments (_check_arg_count)
    ├─ Canonicalize arguments (_canonicalize_args)
    ├─ Mangle name (mangle_name)
    └─ Call generate_mlir()
        ↓
BaseDSL.generate_mlir()
    ├─ Create MLIR Context
    ├─ Convert types (generate_mlir_function_types)
    ├─ Extract dynamic args (extract_dynamic_args)
    └─ Generate IR (generate_original_ir)
        ├─ Create ir.Module
        ├─ Build GPU module (_build_gpu_module)
        ├─ Create func.FuncOp
        ├─ Generate entry block
        ├─ Generate execution arguments
        ├─ Execute funcBody(*ir_args) → emits MLIR ops
        ├─ Add func.ReturnOp
        ├─ Verify module
        └─ Compute hash
        ↓
    Check cache (self.jit_cache)
    ├─ If miss: compile_and_cache()
    │   ↓
    │   Compiler.compile_and_jit()
    │       ├─ Compiler.compile(module, pipeline)
    │       │   ├─ PassManager.parse(pipeline)
    │       │   ├─ PassManager.run(module)
    │       │   │   ├─ cute-to-llvm passes
    │       │   │   ├─ gpu-to-nvvm passes
    │       │   │   ├─ nvvm-to-ptx
    │       │   │   └─ ptx-to-cubin (ptxas)
    │       │   └─ Post-compile hooks
    │       └─ Compiler.jit(module)
    │           ├─ Check CUDA dependencies
    │           └─ Create ExecutionEngine
    │               ↓
    │   load_kernels_from_ir_module()
    │       ├─ Walk module for gpu.binary ops
    │       ├─ Extract CUBIN data
    │       ├─ cuModuleLoadData(cubin)
    │       ├─ cuModuleGetFunction(module, kernel_name)
    │       └─ Set kernel attributes
    │           ↓
    │   Create JitCompiledFunction
    │       └─ Store in cache
    └─ If hit: use cached JitCompiledFunction
        ↓
[If compile_only] Return JitExecutor
[Else] JitExecutor.run_compiled_program(exe_args)
    ├─ Get device context (get_device_execute_context)
    ├─ Pack arguments as C array
    ├─ Call capi_func(packed_args)
    │   ↓
    │   [In ExecutionEngine] _host_wrapper(args)
    │       ├─ Unpack arguments
    │       ├─ Setup launch config
    │       └─ cudaLaunchKernel(kernel, grid, block, args)
    └─ Check CUDA error
        ↓
Return result to user
```

### For @cute.kernel

```
User Code: kernel(a, b, config=LaunchConfig(...))
    ↓
@cute.kernel decorator intercepts
    ↓
BaseDSL.jit_runner(executor="_kernel_helper")
    ↓
[On call] kernel(*args, config=cfg)
    ↓
KernelLauncher.__call__(*args, **kwargs)
    ├─ Extract config from kwargs
    ├─ Validate LaunchConfig
    └─ Call _generate_kernel_and_launch()
        ↓
CutlassBaseDSL._kernel_helper(funcBody, *args, **kwargs)
    ├─ Create _CutlassIrKernelGenHelper
    └─ Call _generate_kernel_and_launch()
        ↓
BaseDSL._generate_kernel_and_launch()
    ├─ Extract config and optional args
    ├─ Mangle kernel name
    ├─ Check and canonicalize arguments
    ├─ Generate kernel operands and types
    ├─ Enter GPU module context
    └─ Generate kernel
        ├─ helper.generate_func_op()
        │   ├─ Create cuda_dialect.KernelOp
        │   └─ Set cu_attrs (cluster size, smem)
        ├─ Add entry block
        ├─ Generate execution arguments
        ├─ Execute funcBody(*ir_args) → emits kernel IR
        ├─ helper.generate_func_ret_op()
        │   └─ Create cuda_dialect.ReturnOp
        └─ helper.generate_launch_op()
            ├─ Extract launch config
            ├─ Calculate shared memory
            ├─ Create SymbolRefAttr for kernel
            └─ CutlassBaseDSL.cuda_launch_func()
                ├─ Create LaunchConfigType
                ├─ Build config operands
                ├─ cuda.CreateLaunchConfigOp
                └─ cuda.launch_ex(config, kernel, operands)
                    ↓
[Rest of compilation same as @cute.jit]
    ↓
Generated MLIR contains:
    ├─ gpu.module @kernels {
    │   └─ cuda.kernel @my_kernel_hash123(...) {
    │         // Kernel body
    │         cuda.return
    │       }
    └─ func.func @host_wrapper(...) {
          %config = cuda.create_launch_config(...)
          cuda.launch_ex %config, @my_kernel_hash123(...)
          func.return
        }
    ↓
[Compilation continues through passes]
    ↓
[At execution, cuda.launch_ex lowers to cudaLaunchKernel]
```

---

## Key Insights

### 1. Python Execution Generates MLIR
- User function executes during compilation
- Operator overloading emits MLIR operations
- Control flow handled by preprocessor or decorators

### 2. Two-Phase Compilation
- **Phase 1**: Python → MLIR (during first call)
- **Phase 2**: MLIR → CUBIN (via pass pipeline)
- Cached compiled modules reused for same signature

### 3. Constexpr vs Dynamic Arguments
- Constexpr args affect compilation (part of cache key)
- Dynamic args don't trigger recompilation
- Allows efficient reuse with varying data

### 4. @cute.jit vs @cute.kernel
- @cute.jit: Simple functions, no launch config
- @cute.kernel: Explicit launch, grid/block/smem control
- Same compilation pipeline, different IR generation

### 5. Execution Modes
- Legacy: Direct CUDA driver calls, device contexts
- Modern: CUDA dialect runtime, context-free
- Both support multi-device execution

### 6. Error Handling
- Compilation errors enhanced with IR context
- NVVM errors include suggestions
- Runtime CUDA errors with descriptive names

### 7. Extensibility
- JitArgAdapter for custom types
- Post-compile hooks for transformations (TVM FFI)
- Custom preprocessor for new constructs
- Pluggable pass pipelines

---

## Source Files Reference

### Core Compilation
- [cutlass/cute/__init__.py](../python/CuTeDSL/cutlass/cute/__init__.py) - Public API
- [cutlass/base_dsl/dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py) - Base DSL class
- [cutlass/cutlass_dsl/cutlass.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py) - CuTe DSL
- [cutlass/base_dsl/compiler.py](../python/CuTeDSL/cutlass/base_dsl/compiler.py) - Compiler & options

### AST & Preprocessing
- [cutlass/base_dsl/ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py) - AST transformation

### Execution & Runtime
- [cutlass/base_dsl/jit_executor.py](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py) - Legacy execution
- [cutlass/cutlass_dsl/cuda_jit_executor.py](../python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py) - CUDA dialect execution
- [cutlass/base_dsl/runtime/cuda.py](../python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py) - CUDA helpers

### Type System
- [cutlass/base_dsl/typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py) - Type definitions
- [cutlass/cute/typing.py](../python/CuTeDSL/cutlass/cute/typing.py) - CuTe types
- [cutlass/base_dsl/runtime/jit_arg_adapters.py](../python/CuTeDSL/cutlass/base_dsl/runtime/jit_arg_adapters.py) - Argument adapters

### TVM FFI (Optional)
- [cutlass/cutlass_dsl/tvm_ffi_provider.py](../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py) - TVM integration

---

## Debugging Tips

### 1. Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Dump IR

```python
import os
os.environ["CUTE_DRYRUN"] = "1"  # Only generate IR, don't compile
os.environ["CUTE_PRINT_AFTER_PREPROCESSOR"] = "1"  # Print AST
```

### 3. Keep PTX/CUBIN

```python
@cute.compile[cute.KeepPTX, cute.KeepCUBIN]
def my_func(...):
    pass
```

### 4. Disable Cache

```python
add(a, b, no_cache=True)  # Force recompilation
```

### 5. Check Compilation Hash

```python
import logging
logging.getLogger("cutlass.base_dsl.dsl").setLevel(logging.DEBUG)
# Will print: "Module hash: abc123..."
```

---

## Conclusion

The CuTeDSL compilation pipeline is a sophisticated multi-stage system that:
1. Transforms Python code via AST preprocessing
2. Generates MLIR IR through operator overloading
3. Compiles to PTX/CUBIN via MLIR passes
4. Loads and executes kernels via CUDA runtime

Understanding this pipeline enables:
- Writing efficient GPU kernels in Python
- Debugging compilation issues
- Extending the DSL with custom operations
- Optimizing kernel performance
- Integrating with other frameworks (TVM, JAX, etc.)

This documentation provides a complete reference for understanding and working with CuTeDSL's compilation infrastructure.
