# MLIR Context and `site_initialize` in CuTeDSL

## Overview

`ir.Context` in CuTeDSL is **not** the raw C++ pybind11 class. It's a Python subclass created at import time by `_site_initialize()`, which monkey-patches `ir.Context` with dialect registration, multithreading config, and LLVM translation setup.

---

## The Three Layers

### Layer 1: C++ pybind11 binding

**Module:** `cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir`

Exports a raw `Context` class (renamed to `_BaseContext` by the site init layer). Provides core MLIR functionality: create modules, operations, types, attributes. No dialects registered, no translations loaded.

### Layer 2: `_site_initialize()` — import-time subclass injection

**File:** `python/CuTeDSL/cutlass/_mlir/_mlir_libs/__init__.py:53-179`

Runs at import time (line 179: `_site_initialize()`). Performs:

1. **Plugin discovery** — imports `_mlirRegisterEverything` and `_site_initialize_{0,1,2,...}` modules. These are compiled C extensions that can provide:
   - `register_dialects(registry)` — populate the shared `DialectRegistry`
   - `context_init_hook(ctx)` — post-creation callback
   - `disable_multithreading` — boolean flag
   - `register_llvm_translations(ctx)` — LLVM backend registration

2. **Defines enriched `Context`** subclass (line 107):
   ```python
   class Context(ir._BaseContext):
       def __init__(self, ...):
           super().__init__()
           self.append_dialect_registry(get_dialect_registry())  # all discovered dialects
           for hook in post_init_hooks:
               hook(self)                                         # custom init hooks
           if not disable_multithreading:
               self.enable_multithreading(True)                   # default: MT on
           self.load_all_available_dialects()                     # eager load
           init_module.register_llvm_translations(self)           # LLVM/NVVM backend
   ```

3. **Monkey-patches** `ir.Context = Context` (line 143) — replaces the C++ class attribute.

### Layer 3: Public re-export

**File:** `python/CuTeDSL/cutlass/_mlir/ir.py:5`
```python
from ._mlir_libs._cutlass_ir._mlir.ir import *
```

By the time this star-import runs, `_site_initialize()` has already replaced `ir.Context`, so all downstream code gets the enriched version.

---

## Why This Design?

From the [upstream MLIR source](https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/_mlir_libs/__init__.py) (lines 38-40):

> "Aside from just being far more convenient to do this at the Python level,
> it is actually quite hard/impossible to have such `__init__` hooks, given
> the pybind memory model (i.e. there is not a Python reference to the object
> in the scope of the base class `__init__`)."

Key reasons:

1. **pybind11 limitation** — C++ object construction happens before Python `__init__`, so you can't inject Python-level logic into the C++ constructor.
2. **Decoupling** — Different MLIR-based projects (CuTeDSL, CIRCT, IREE, torch-mlir) need different dialect sets. Can't hardcode in C++.
3. **Plugin architecture** — Drop a `_site_initialize_N.so` into the package and it's auto-discovered. No central registry needed.
4. **Consistency** — Every `ir.Context()` anywhere in the process gets identical configuration.

---

## Is Context a Singleton?

**No.** Multiple independent contexts can coexist. It uses a **thread-local stack** for `Context.current`:

```python
ctx1 = ir.Context()   # creates context, does NOT set as current
ctx2 = ir.Context()   # another independent context

ir.Context.current    # → None

with ctx1:
    ir.Context.current  # → ctx1
    with ctx2:
        ir.Context.current  # → ctx2 (nested push)
    ir.Context.current  # → ctx1 (restored)

ir.Context.current    # → None
```

CuTeDSL creates a **fresh Context per compilation** at `dsl.py:1345`:
```python
with ir.Context(), self.get_ir_location(location):
    # all MLIR ops created here belong to this context
```

---

## Introspecting a Context

```python
from cutlass._mlir import ir

ctx = ir.Context()
with ctx:
    # --- Type hierarchy ---
    type(ctx)                   # <class '...Context'> (the site_initialize'd subclass)
    type(ctx).__mro__           # Context → _BaseContext → pybind11_object

    # --- Liveness ---
    ir.Context._get_live_count()          # number of Context objects alive
    ctx._get_live_operation_count()       # ops in this context
    ctx._get_live_module_count()          # modules in this context
    ctx._get_live_operation_objects()     # list of live Operation objects
    ctx._clear_live_operations()          # force cleanup, returns count cleared

    # --- Dialect inspection ---
    ctx.dialects                                    # Dialects container
    ctx.is_registered_operation("func.func")        # True
    ctx.is_registered_operation("cute.some_op")     # check custom ops
    ctx.allow_unregistered_dialects                 # False by default

    # --- Threading ---
    ctx.enable_multithreading(False)    # required before enable_ir_printing

    # --- Diagnostics ---
    def handler(diag):
        print(f"[{diag.severity}] {diag.message}")
        return True   # True = handled
    ctx.attach_diagnostic_handler(handler)
```

---

## CuTeDSL vs Upstream Difference

| Aspect | Upstream MLIR | CuTeDSL |
|--------|--------------|---------|
| Original saved as | `ir._Context = ir.Context` | Uses `ir._BaseContext` (renamed at C++ level) |
| Dialect registration | `m.register_dialects(registry)` | Calls C++ `register_dialects(registry)` from `_cutlass_ir` |
| LLVM translations | `init_module.register_llvm_translations(ctx)` | Same pattern |
| Thread pool support | `ctx.set_thread_pool(pool)` (upstream main) | Not present in CuTeDSL's version |

---

## References

- Upstream source (authoritative): [`mlir/python/mlir/_mlir_libs/__init__.py`](https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/_mlir_libs/__init__.py)
- MLIR Python Bindings docs: [`mlir/docs/Bindings/Python.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Bindings/Python.md)
- CuTeDSL local copy: `python/CuTeDSL/cutlass/_mlir/_mlir_libs/__init__.py`
- Context stub: `python/CuTeDSL/cutlass/_mlir/_mlir_libs/_cutlass_ir/_mlir/ir.pyi:978-1017`
