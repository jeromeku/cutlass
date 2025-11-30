# Why Breakpoints Don't Work in Compiler.compile()

## TL;DR

**Breakpoints fail inside `Compiler.compile()` because the critical line `pm.run(module.operation)` calls into compiled C++ code (MLIR's PassManager), which executes outside the Python debugger's control.**

---

## The Problem

When debugging CuTeDSL compilation:

```python
# ✅ Breakpoint works here
def compile_and_jit(self, module, pipeline, ...):
    breakpoint()  # STOPS ✓

    self.compile(module, pipeline, ...)  # ← Calls compile()

    return self.jit(module, ...)

# ⚠️ Breakpoint doesn't work here
def compile(self, module, pipeline, ...):
    breakpoint()  # DOESN'T STOP ✗ (or stops then immediately continues)

    pm = self.passmanager.PassManager.parse(pipeline)
    pm.enable_verifier(enable_verifier)
    pm.run(module.operation)  # ← THIS is the problem!
    # ... rest never reached with breakpoint
```

**What you observe:**
1. Breakpoint in `compile_and_jit()` works fine
2. Step into `compile()` - breakpoint seems to be skipped
3. Function returns immediately (or so it appears)
4. Execution continues after `compile()` call

---

## Root Cause: C++ Extension Call

### The Smoking Gun

[compiler.py:146-148](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L146-L148):
```python
pm = self.passmanager.PassManager.parse(pipeline)
pm.enable_verifier(enable_verifier)
pm.run(module.operation)  # ← Calls into C++
```

### What `passmanager` Actually Is

[_mlir/passmanager.py:5](../python/CuTeDSL/cutlass/_mlir/passmanager.py#L5):
```python
from ._mlir_libs._cutlass_ir._mlir.passmanager import *
```

This imports from: `/home/jeromeku/cutlass/python/CuTeDSL/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-312-x86_64-linux-gnu.so`

**This is a compiled C++ extension!**

### The Call Chain

```
Python                          C++ (compiled .so)
------                          ------------------
pm.run(module.operation)
    ↓
[Python/C boundary]
    ↓
    PassManager::run()           ← MLIR C++ code
        ↓
        Pass execution
        NVVM compilation
        PTX generation
        CUBIN assembly
        (takes seconds to minutes)
    ↓
[C++/Python boundary]
    ↓
return to Python
```

**The problem:** While execution is in C++, the Python debugger has no visibility or control.

---

## Why This Happens

### 1. **Python Debugger Scope**

Python debuggers (pdb, ipdb, VS Code debugger) work by:
- Instrumenting Python bytecode
- Hooking into Python's tracing facility
- Setting breakpoints in Python source

**They cannot:**
- See inside compiled C/C++ extensions
- Step through machine code
- Control execution in native libraries

### 2. **MLIR is Native Code**

MLIR (Multi-Level Intermediate Representation) is:
- Written in C++ (part of LLVM project)
- Compiled to native machine code (.so files)
- Exposed to Python via pybind11 bindings

When you call `pm.run()`:
- Python calls a C++ function
- Execution transfers to compiled code
- Python debugger loses control
- Only returns when C++ completes

### 3. **Time Spent in C++**

The `pm.run()` call does:
- Parse pass pipeline
- Run 10+ MLIR transformation passes
- Lower CuTe → NVVM → PTX
- Call `ptxas` (CUDA assembler) via subprocess
- Assemble PTX → CUBIN

**This takes seconds to minutes**, during which Python debugger sees "nothing happening."

---

## What Actually Happens

### Timeline of Execution

```
T=0ms    ┌─ compile_and_jit() entry
         │  breakpoint()              ← STOPS: Debugger has control
         │
T=10ms   ├─ self.compile() call
         │  breakpoint()              ← MAY STOP: But immediately...
         │  pm.run()
         │      ↓
         │  [Enters C++]              ← Debugger loses control
         │      ↓
T=20ms   │  MLIR passes running...    ← Appears "frozen" to debugger
...      │  ...
T=5000ms │  CUBIN generated
         │      ↓
         │  [Returns to Python]       ← Debugger regains control
         │
T=5010ms ├─ self.jit() call           ← Breakpoint would work here
         └─ return
```

### What Debugger Sees

From debugger's perspective:
```
compile_and_jit (frame exists)
    ↓
compile (frame exists briefly)
    ↓
<black box - no visibility>
    ↓
compile returns
```

The frame for `compile()` exists, but there's no Python code actively executing inside it for most of the time.

---

## Evidence in the Code

### 1. PassManager is Binary

```bash
$ file /home/jeromeku/cutlass/python/CuTeDSL/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-312-x86_64-linux-gnu.so
_cutlass_ir.cpython-312-x86_64-linux-gnu.so: ELF 64-bit LSB shared object, x86-64
```

### 2. Import Chain

```python
# cutlass/base_dsl/compiler.py
from .._mlir import passmanager

# cutlass/_mlir/passmanager.py
from ._mlir_libs._cutlass_ir._mlir.passmanager import *
                     ↑
                     └─ Compiled C++ module (pybind11)
```

### 3. Similar Pattern in ExecutionEngine

[_mlir/execution_engine.py:5](../python/CuTeDSL/cutlass/_mlir/execution_engine.py#L5):
```python
from ._mlir_libs._cutlass_ir import _mlirExecutionEngine as _execution_engine
```

This is also C++, so `ExecutionEngine()` constructor calls are similarly opaque to Python debuggers.

---

## Why Some Breakpoints Work

### ✅ Works: Before C++ Call

```python
def compile(self, module, pipeline, ...):
    print("Before compile")  # ✓ Executes
    breakpoint()             # ✓ Stops

    pm = self.passmanager.PassManager.parse(pipeline)  # ← Last Python line
    pm.run(module.operation)  # ← Enters C++
```

### ✅ Works: After C++ Returns

```python
def compile(self, module, pipeline, ...):
    try:
        pm.run(module.operation)  # ← C++ execution
    except Exception as e:
        breakpoint()  # ✓ Stops (back in Python)
```

### ❌ Doesn't Work: Inside C++ Time

```python
def compile(self, module, pipeline, ...):
    pm.run(module.operation)  # ← Execution here is in C++
    print("After compile")    # ✗ No Python code running during pm.run()
```

---

## Workarounds for Debugging

### Option 1: Add Print Statements Before/After

```python
def compile(self, module, pipeline, ...):
    print(f"[DEBUG] Starting compilation with pipeline: {pipeline}")
    print(f"[DEBUG] Module:\n{module}")

    start = time.time()
    pm = self.passmanager.PassManager.parse(pipeline)
    pm.enable_verifier(enable_verifier)
    pm.run(module.operation)
    elapsed = time.time() - start

    print(f"[DEBUG] Compilation took {elapsed:.2f}s")

    if self._post_compile_hook:
        print(f"[DEBUG] Running post-compile hook")
        self._post_compile_hook(module)
```

### Option 2: Dump IR Before Pass Execution

```python
def compile(self, module, pipeline, ...):
    # Dump IR before compilation
    import os
    if os.getenv("DEBUG_DUMP_IR"):
        with open(f"/tmp/mlir_before_{id(module)}.mlir", "w") as f:
            f.write(str(module))

    pm.run(module.operation)

    # Dump IR after compilation
    if os.getenv("DEBUG_DUMP_IR"):
        with open(f"/tmp/mlir_after_{id(module)}.mlir", "w") as f:
            f.write(str(module))
```

### Option 3: Use Environment Variables

The CuTeDSL already supports debug environment variables:

```python
import os

# Dump MLIR IR
os.environ["CUTE_KEEP_IR"] = "1"

# Keep PTX/CUBIN
os.environ["CUTE_KEEP_PTX"] = "1"
os.environ["CUTE_KEEP_CUBIN"] = "1"

# Dry run (generate IR only, don't compile)
os.environ["CUTE_DRYRUN"] = "1"

# Print IR after preprocessing
os.environ["CUTE_PRINT_AFTER_PREPROCESSOR"] = "1"
```

### Option 4: Catch Exceptions to Inspect State

```python
def compile(self, module, pipeline, ...):
    try:
        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)
    except Exception as e:
        # Exception brings us back to Python
        print(f"[DEBUG] Compilation failed: {e}")
        print(f"[DEBUG] Module IR:\n{module}")
        print(f"[DEBUG] Pipeline: {pipeline}")

        breakpoint()  # ✓ Now this works!
        raise
```

### Option 5: Monkey-Patch for Timing

```python
# In your debugging script
import cutlass.base_dsl.compiler as compiler_module
import time

original_compile = compiler_module.Compiler.compile

def debug_compile(self, module, pipeline, **kwargs):
    print(f"\n{'='*60}")
    print(f"[COMPILE START]")
    print(f"  Pipeline: {pipeline[:100]}...")
    print(f"  Module hash: {hash(str(module))}")
    print(f"{'='*60}\n")

    start = time.time()
    try:
        result = original_compile(self, module, pipeline, **kwargs)
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"[COMPILE SUCCESS] {elapsed:.2f}s")
        print(f"{'='*60}\n")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"[COMPILE FAILED] {elapsed:.2f}s: {e}")
        print(f"{'='*60}\n")
        raise

compiler_module.Compiler.compile = debug_compile
```

### Option 6: Use C++ Debugger (Advanced)

If you need to debug inside MLIR passes:

```bash
# Build MLIR with debug symbols
# Then attach gdb/lldb to Python process

$ gdb --args python my_script.py
(gdb) break mlir::PassManager::run
(gdb) run
```

**Warning:** This requires:
- MLIR source code
- Debug build of MLIR
- C++ debugging skills
- Much more complex setup

---

## Why compile_and_jit Breakpoint Works

```python
def compile_and_jit(self, module, pipeline, ...):
    breakpoint()  # ✓ Works: Pure Python context

    self.compile(module, pipeline, ...)  # Python calls compile()
    #     ↑                                Returns after C++ completes
    #     └─ Python frame exists, but most time spent in C++

    return self.jit(module, ...)  # ✓ Breakpoint would work here too
```

**Why it works:**
- `compile_and_jit()` is pure Python
- Even though it *calls* C++, it doesn't *run in* C++
- Debugger maintains control at the Python level
- The call to `self.compile()` appears as a single step

---

## Testing the Theory

### Experiment 1: Timing

```python
import time

def compile(self, module, pipeline, ...):
    print(f"[{time.time()}] Entering compile()")
    breakpoint()  # Set here

    print(f"[{time.time()}] About to call pm.run()")
    pm.run(module.operation)

    print(f"[{time.time()}] Returned from pm.run()")
```

**Expected output:**
```
[1234567890.123] Entering compile()
> breakpoint() hit
(Pdb) continue

[1234567890.125] About to call pm.run()
[1234567895.456] Returned from pm.run()
       ↑
       └─ 5+ second gap with no output = time in C++
```

### Experiment 2: Thread Inspection

```python
import threading

def compile(self, module, pipeline, ...):
    print(f"Thread before: {threading.current_thread().name}")

    pm.run(module.operation)

    print(f"Thread after: {threading.current_thread().name}")
```

**Expected:** Same thread (MainThread), but execution was in C++.

### Experiment 3: Stack Frames

```python
import traceback

def compile(self, module, pipeline, ...):
    print("Stack before pm.run():")
    traceback.print_stack()

    pm.run(module.operation)  # While this runs, stack is different

    print("Stack after pm.run():")
    traceback.print_stack()
```

---

## Related Debugging Issues

### 1. Step Over Doesn't Work as Expected

```python
def compile_and_jit(self, ...):
    self.compile(...)  # (Pdb) next
    #                  ↓ "Hangs" for seconds
    return self.jit(...)
```

**Why:** `next` (step over) waits for the entire function to complete, including C++ execution time.

### 2. Step Into Doesn't Show Source

```python
def compile_and_jit(self, ...):
    self.compile(...)  # (Pdb) step
    #                  ↓ Shows compile() source, but then...
    pm.run(...)        # (Pdb) step
    #                  ↓ "No source available" or immediately returns
```

**Why:** Debugger can't step into compiled C++ without debug symbols.

### 3. Breakpoint in Exception Handler Works

```python
def compile(self, ...):
    try:
        pm.run(module.operation)
    except Exception as e:
        breakpoint()  # ✓ This DOES work!
```

**Why:** Exception brings execution back to Python, where debugger has control.

---

## Summary

### The Core Issue

| Where | What Happens | Debugger Works? |
|-------|--------------|-----------------|
| `compile_and_jit()` entry | Pure Python | ✅ Yes |
| `compile()` entry | Pure Python (briefly) | ⚠️ Yes, but immediately... |
| Inside `pm.run()` | **Compiled C++ code** | ❌ No - debugger blind |
| `compile()` exit | Pure Python | ✅ Yes |
| `jit()` call | Python → C++ (ExecutionEngine) | ⚠️ Similar issue |

### The Real Execution

```
Python visible:     ▪️▪️▪️ (30ms)
                     ↓
C++ invisible:       ██████████████████ (5000ms) ← MLIR passes
                                                   ↓
Python visible:                                    ▪️▪️▪️ (10ms)
```

**Total time:** 5040ms, but debugger only sees 40ms of Python execution.

### Best Practices for Debugging

1. **Use print statements** around C++ calls
2. **Dump IR** before/after pass execution
3. **Time each stage** to identify bottlenecks
4. **Catch exceptions** to inspect state when things fail
5. **Use environment variables** for MLIR debugging
6. **Avoid stepping into** C++ extension calls
7. **Set breakpoints after** C++ calls return

### Remember

**The breakpoint isn't "broken" - it's just that there's no Python code executing during the long compilation phase!**
