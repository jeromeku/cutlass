# Fix: Save CuteDSL Generated Code to `TORCH_COMPILE_DEBUG_DIR`

**Date:** 2026-02-14
**Status:** Not yet applied — instructions only

---

## Problem

When running with `TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=inductor_logs`, PyTorch saves the Triton-generated wrapper code to the debug directory but **not** the CuteDSL (Flash) generated code. The CuteDSL code lands only in `/tmp/torchinductor_<user>/` and is not copied to `TORCH_COMPILE_DEBUG_DIR`.

## Why It Happens

### Triton path (works correctly)

**File:** `thirdparty/pytorch/torch/_inductor/graph.py`, lines 2562–2566

```python
key, path = PyCodeCache.write(wrapper_code.value)   # write to /tmp/...
output_code_log.debug("Output code written to: %s", path)
V.debug.output_code(path)                           # ← copies to TORCH_COMPILE_DEBUG_DIR
V.debug.copy(os.path.splitext(path)[0] + ".debug")
```

`V.debug.output_code(path)` calls `shutil.copy(filename, self.filename("output_code.py"))` inside the active `DebugContext`, which writes to `TORCH_COMPILE_DEBUG_DIR/torchinductor/<module_name>/output_code.py`.

### CuteDSL path (broken)

**File:** `thirdparty/pytorch/torch/_inductor/autotune_process.py`, lines 1026–1027

```python
finalized_code = source_code.finalize_all()
self.module_cache_key, self.module_path = PyCodeCache.write(finalized_code)
# ← V.debug.output_code() is never called; file goes to /tmp only
```

This `PyCodeCache.write()` call happens inside `CuteDSLBenchmarkRequest.__init__()`, which is invoked from `CuteDSLTemplate.generate()` in `cutedsl_template.py`. At that call site `V` (the inductor virtual context) is active and `V.debug` is the live `DebugContext`, but nobody ever calls `V.debug.output_code()` with the resulting path.

The `log.debug("Generated CuteDSL Code:\n%s", code)` line at `cutedsl_template.py:91` only emits to the logging stream (visible when `TORCH_LOGS=output_code`); it does not write a file.

---

## The Fix

### File to edit

```
thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_template.py
```

### Context — current code (lines 89–99)

```python
            code = kernel.render(self.template, **kwargs)

            log.debug("Generated CuteDSL Code:\n%s", code)

            bmreq = CuteDSLBenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=tuple(),
                source_code=code,
            )
```

### Change required

Add **one line** immediately after the closing `)` of `CuteDSLBenchmarkRequest(...)`, at line 99:

```python
            code = kernel.render(self.template, **kwargs)

            log.debug("Generated CuteDSL Code:\n%s", code)

            bmreq = CuteDSLBenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=tuple(),
                source_code=code,
            )
            V.debug.output_code(bmreq.module_path)   # ← ADD THIS LINE
```

### Why this is safe

- `V` is already imported at line 9 of `cutedsl_template.py`:
  ```python
  from torch._inductor.virtualized import V
  ```
- `V.debug` is the null debug handler by default. When `TORCH_COMPILE_DEBUG` is not set, `V.debug.output_code()` is a no-op — it goes to `NullHandler.output_code()` which returns immediately. No files are written, no performance impact.
- When `TORCH_COMPILE_DEBUG=1`, `V.debug` is a live `DebugContext` instance and `output_code()` calls `shutil.copy(filename, self.filename("output_code.py"))`, writing into the current per-module debug subdirectory.
- `bmreq.module_path` is already set by `PyCodeCache.write()` inside `CuteDSLBenchmarkRequest.__init__()`, so the file is guaranteed to exist at this point.

---

## Step-by-Step Instructions

### Step 1 — Open the file

```
thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_template.py
```

Navigate to line 99 (the closing `)` of `CuteDSLBenchmarkRequest(...)`).

### Step 2 — Verify surrounding context

Confirm lines 93–99 look like this before editing:

```python
            bmreq = CuteDSLBenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=tuple(),
                source_code=code,
            )
```

The next line after `)` should be a blank line followed by `def make_kernel_render(...)`.

### Step 3 — Insert the new line

After the closing `)` on line 99, add (preserving 12-space indent — this is inside `with patch.object(...)` → inside `generate()`):

```python
            V.debug.output_code(bmreq.module_path)
```

The result should look like:

```python
            bmreq = CuteDSLBenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=tuple(),
                source_code=code,
            )
            V.debug.output_code(bmreq.module_path)

            def make_kernel_render(out_node, hint_override: Optional[int] = None):
```

### Step 4 — Verify no new import is needed

Check that line 9 of `cutedsl_template.py` already has:

```python
from torch._inductor.virtualized import V
```

If for any reason it is absent, add it to the imports block at the top of the file.

### Step 5 — Test the fix

Run the attention-gym example with the debug flags:

```bash
cd thirdparty/attention-gym
TORCH_COMPILE_DEBUG=1 \
TORCH_COMPILE_DEBUG_DIR=inductor_logs \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCH_LOGS=+inductor,output_code \
python examples/flex_flash_attention.py
```

### Step 6 — Confirm output

Look for CuteDSL kernel files in the debug directory:

```bash
find inductor_logs -name "output_code.py" | xargs grep -l "cutedsl\|flash_attention_cutedsl" 2>/dev/null
```

You should see at least two `output_code.py` files per compiled call — one for the Triton wrapper and one (new) for the CuteDSL kernel.

You can also diff against what was previously only visible in `/tmp`:

```bash
ls /tmp/torchinductor_$(whoami)/*.py | xargs grep -l "flash_attention_cutedsl"
```

Those files and the newly copied debug files should have the same content.

---

## Reference: Key Files

| File | Line | Role |
|------|------|------|
| [cutedsl_template.py](../thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_template.py#L89) | 89–99 | Where to add the fix |
| [autotune_process.py](../thirdparty/pytorch/torch/_inductor/autotune_process.py#L1026) | 1026–1027 | Where `module_path` is set by `PyCodeCache.write()` |
| [graph.py](../thirdparty/pytorch/torch/_inductor/graph.py#L2562) | 2562–2566 | Triton analog — the pattern being replicated |
| [debug.py](../thirdparty/pytorch/torch/_inductor/debug.py#L628) | 628–629 | `DebugContext.output_code()` implementation |
