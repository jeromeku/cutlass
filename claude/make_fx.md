# `make_fx` Internals — Deep Trace

**Date:** 2026-02-14
**Source:** `thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py`

---

## Table of Contents

1. [What `make_fx` Does](#1-what-make_fx-does)
2. [The Mode Stack — Layered Interception](#2-the-mode-stack--layered-interception)
3. [Tracing Modes: real vs fake vs symbolic](#3-tracing-modes-real-vs-fake-vs-symbolic)
4. [Module-by-Module Walkthrough](#4-module-by-module-walkthrough)
   - 4.1 [`make_fx` → `_MakefxTracer`](#41-make_fx--_makefxtracer)
   - 4.2 [`_init_modes_from_inputs`](#42-_init_modes_from_inputs)
   - 4.3 [`_trace_inner`](#43-_trace_inner)
   - 4.4 [`wrap_key`](#44-wrap_key)
   - 4.5 [`dispatch_trace`](#45-dispatch_trace)
   - 4.6 [`PythonKeyTracer`](#46-pythonkeytracer)
   - 4.7 [`ProxyTorchDispatchMode`](#47-proxytorchdispatchmode)
   - 4.8 [`proxy_call`](#48-proxy_call)
   - 4.9 [`FakeTensorMode`](#49-faketensormode)
   - 4.10 [`TorchFunctionMetadataMode`](#410-torchfunctionmetadatamode)
   - 4.11 [`track_tensor_tree`](#411-track_tensor_tree)
5. [End-to-End Data Flow](#5-end-to-end-data-flow)
6. [The Complete Context Manager Stack](#6-the-complete-context-manager-stack)
7. [Code Map](#7-code-map)

---

## 1. What `make_fx` Does

`make_fx(fn)(*args)` executes `fn` with real (or fake) tensor arguments and records **every
ATen operator call** into an `fx.Graph`, producing an `fx.GraphModule` — a self-contained
Python module whose `forward()` is the symbolically traced computation.

```python
gm = make_fx(fn)(*example_inputs)
# gm.graph       — the FX graph (nodes: placeholder, call_function, output)
# gm.code        — generated Python source
# gm(new_inputs) — executes the captured graph
```

The key insight: `make_fx` does **not** use `torch.fx.symbolic_trace` (which rewrites Python
bytecode). Instead it uses **dispatch-mode interception**: the function runs with real Python
control flow, and a `TorchDispatchMode` intercepts every `aten::*` call at the C++ dispatcher
level, creating an FX node for each one.

---

## 2. The Mode Stack — Layered Interception

`make_fx` installs a **stack of modes** (context managers) that intercept operations at
different levels of the PyTorch dispatch pipeline. From outermost (first to see a call) to
innermost (last):

```
User calls:  torch.mm(x, y)
  │
  ▼
┌──────────────────────────────────────┐
│  TorchFunctionMetadataMode           │ ← __torch_function__: records which
│  (TorchFunctionMode)                 │   torch API function was called
├──────────────────────────────────────┤
│  ProxyTorchDispatchMode              │ ← __torch_dispatch__: creates FX node
│  (TorchDispatchMode, PROXY key)      │   (proxy_call), tracks tensor↔proxy
├──────────────────────────────────────┤
│  FakeTensorMode                      │ ← __torch_dispatch__: computes output
│  (TorchDispatchMode, FAKE key)       │   metadata (shape/dtype/device) without
│                                      │   running real kernels
├──────────────────────────────────────┤
│  Decomposition table                 │ ← replaces composite ops with their
│  (CURRENT_DECOMPOSITION_TABLE)       │   decompositions before recording
└──────────────────────────────────────┘
```

When `torch.mm(x, y)` is called:

1. `TorchFunctionMetadataMode.__torch_function__` fires first — it records `torch.mm`
   as metadata on the tracer, then calls through.
2. The C++ dispatcher dispatches to `aten.mm.default`.
3. `ProxyTorchDispatchMode.__torch_dispatch__` intercepts `aten.mm.default` — it
   creates a `Proxy` node in the FX graph and then calls the real op to get the output.
4. `FakeTensorMode.__torch_dispatch__` intercepts the real op call — instead of
   running the CUDA/CPU kernel, it computes the output **shape and dtype** using
   meta-kernel logic, returning a `FakeTensor`.
5. The FakeTensor output is associated with the Proxy node via `track_tensor_tree`.

---

## 3. Tracing Modes: real vs fake vs symbolic

`make_fx` supports three `tracing_mode` values that control how tensors are handled during
tracing. They differ in what happens when the function is actually executed:

### 3.1 `tracing_mode="real"` (default)

**How it works:** No `FakeTensorMode` is installed. The function runs with **real tensors**
and **real computation** — actual CUDA/CPU kernels execute.

**Input wrapping:** Inputs are used as-is (`lambda x: x`).

**Pros:**
- Simplest mode. Always correct — no approximation.
- Works with data-dependent control flow (e.g., `if x.sum() > 0:`) because values are
  concrete.

**Cons:**
- Must have real data. Cannot trace with symbolic shapes.
- Burns in concrete shape information — the graph is specialized to the exact input shapes.
- Expensive: actually runs the computation.

**Use case:** Quick debugging, `get_isolated_graphmodule`, VJP tracing via
`create_fw_bw_graph` (where fake tensors are already set up externally).

**Source:** [proxy_tensor.py:2532–2536](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2532)

```python
# real mode: no FakeTensorMode, no ShapeEnv
else:
    assert self.tracing_mode == "real"
    # self.fake_tensor_mode stays None
```

---

### 3.2 `tracing_mode="fake"`

**How it works:** A `FakeTensorMode` is installed with `static_shapes=True`. Real tensors are
converted to `FakeTensor`s (metadata-only tensors with the same shape/dtype/device but no data).
Operations run through meta-kernels that compute output shapes without executing real kernels.

**Input wrapping:** `fake_tensor_mode.from_tensor(x)` — creates a FakeTensor with the same
shape, dtype, device.

**Pros:**
- No real computation needed — much faster for large tensors.
- No GPU required — can trace CUDA graphs on CPU.

**Cons:**
- Shapes are still **static** (concrete integers). The graph is specialized to the example
  input shapes.
- Data-dependent control flow errors out (`_error_on_data_dependent_ops=True`), since there
  is no real data to branch on.
- `allow_fallback_kernels=True` — if a meta-kernel is missing, falls back to real execution.

**Use case:** AOT Autograd tracing, inductor compilation — the standard
`torch.compile` path traces with fake tensors.

**Source:** [proxy_tensor.py:2498–2512](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2498)

```python
if self.tracing_mode == "fake":
    fake_tensor_mode = FakeTensorMode(
        allow_fallback_kernels=True,
        allow_non_fake_inputs=self._allow_non_fake_inputs,
        shape_env=ShapeEnv(),
        static_shapes=True,          # ← shapes are concrete ints
    )
```

---

### 3.3 `tracing_mode="symbolic"`

**How it works:** A `FakeTensorMode` is installed with `static_shapes=False` (the default).
Tensor dimensions become `SymInt`s — symbolic integer expressions tracked by a `ShapeEnv`.
The Python dispatcher is also enabled (to handle per-dispatch-key routing).

**Input wrapping:** Same as fake — `fake_tensor_mode.from_tensor(x)` — but dimensions are
`SymInt(s0)`, `SymInt(s1)`, etc. Integer arguments become `SymInt`s via
`shape_env.create_symintnode`.

**Pros:**
- Graph is **shape-generic** — works for different input sizes.
- Captures shape guards and symbolic constraints in `ShapeEnv`.
- Required for export and dynamic-shape compilation.

**Cons:**
- Most complex mode. More likely to hit unsupported patterns.
- `allow_fallback_kernels=False` — all ops must have meta-kernels (no fallback).
- Data-dependent ops that inspect tensor values (e.g., `.item()`, `nonzero()`) produce
  unbacked SymInts or error out.
- Requires the Python dispatcher (`enable_python_dispatcher()`).

**Use case:** `torch.export`, `torch.compile` with `dynamic=True`,
any path that needs shape-generic graphs.

**Source:** [proxy_tensor.py:2513–2531](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2513)

```python
elif self.tracing_mode == "symbolic":
    fake_tensor_mode = FakeTensorMode(
        allow_fallback_kernels=False,    # ← strict: no fallbacks
        allow_non_fake_inputs=self._allow_non_fake_inputs,
        shape_env=shape_env,
        # static_shapes defaults to False → SymInt dimensions
    )
```

---

### 3.4 Comparison table

| | `"real"` | `"fake"` | `"symbolic"` |
|---|---|---|---|
| **Tensor type during trace** | Real (`torch.Tensor`) | `FakeTensor` (static shapes) | `FakeTensor` (symbolic shapes) |
| **Kernels executed** | Yes (real CUDA/CPU) | No (meta-kernels only) | No (meta-kernels only) |
| **FakeTensorMode** | Not installed | `static_shapes=True` | `static_shapes=False` |
| **ShapeEnv** | None | Created but unused | Tracks symbolic dims + guards |
| **Python dispatcher** | Off | Off | On (`enable_python_dispatcher`) |
| **Shape generality** | Concrete only | Concrete only | Dynamic (SymInt dims) |
| **Data-dependent ops** | Work (real values) | Error | Produce unbacked SymInt or error |
| **Fallback kernels** | N/A (real) | Allowed | Disallowed |
| **Primary users** | Debugging, `create_fw_bw_graph` | `torch.compile` (static) | `torch.export`, `compile(dynamic=True)` |

---

## 4. Module-by-Module Walkthrough

### 4.1 `make_fx` → `_MakefxTracer`

**File:** [proxy_tensor.py:2779](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2779)

`make_fx` is a thin wrapper. It creates a `_MakefxTracer` with the user's configuration and
returns a closure that calls `tracer.trace(f, *args)`.

```python
def make_fx(f, decomposition_table=None, tracing_mode="real", ...):
    make_fx_tracer = _MakefxTracer(decomposition_table, tracing_mode, ...)

    def wrapped(*args):
        return make_fx_tracer.trace(f, *args)

    return wrapped
```

`_MakefxTracer` ([line 2403](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2403))
holds all configuration and the mode objects. Key state:

| Field | Type | Purpose |
|---|---|---|
| `fx_tracer` | `PythonKeyTracer` | The FX graph builder (creates nodes) |
| `fake_tensor_mode` | `FakeTensorMode` or `None` | Fake tensor dispatch (fake/symbolic modes) |
| `proxy_mode` | `ProxyTorchDispatchMode` | The TorchDispatchMode that records ops as FX nodes |
| `proxy_function_mode` | `PreDispatchTorchFunctionMode` | Pre-dispatch API recording (export only) |
| `torch_fn_metadata_mode` | `TorchFunctionMetadataMode` | Records which `torch.*` function was called |
| `python_dispatcher_mode` | `enable_python_dispatcher()` | Enables per-key Python dispatch (symbolic only) |
| `decomposition_table` | `dict[OpOverload, Callable]` | Op decompositions to apply before recording |

---

### 4.2 `_init_modes_from_inputs`

**File:** [proxy_tensor.py:2478](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2478)

Called by `trace()`. Initializes all modes based on `tracing_mode` and the input arguments:

1. Creates `PythonKeyTracer` (or `_ModuleStackTracer` if `record_module_stack=True`).
2. Creates `FakeTensorMode` with `static_shapes=True` (fake) or `False` (symbolic).
   For real mode, `fake_tensor_mode` stays `None`.
3. Calls `_construct_modes_with_fx_tracer` to create `ProxyTorchDispatchMode` and the
   other mode objects.

```python
@contextmanager
def _init_modes_from_inputs(self, f, args):
    self.fx_tracer = PythonKeyTracer()

    if self.tracing_mode == "fake":
        self.fake_tensor_mode = FakeTensorMode(static_shapes=True, ...)
    elif self.tracing_mode == "symbolic":
        self.fake_tensor_mode = FakeTensorMode(shape_env=ShapeEnv(), ...)

    self._construct_modes_with_fx_tracer(self.fx_tracer)
    yield
```

---

### 4.3 `_trace_inner`

**File:** [proxy_tensor.py:2593](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L2593)

The core tracing logic. Three phases:

**Phase 1 — Input preparation:**

```python
phs = pytree.tree_map(lambda _: PH, args)   # placeholder sentinels for FX
args = _wrap_fake(args)                       # real→FakeTensor (if fake/symbolic)
func = _wrap_func(f, phs)                     # handle varargs
```

`_wrap_fake` is the key input transformation. For each input tensor:
- **real mode:** identity — returns the tensor unchanged.
- **fake/symbolic mode:** `fake_tensor_mode.from_tensor(x, source=...)` — creates a
  `FakeTensor` with the same metadata. In symbolic mode, dimensions become `SymInt`s.
  Integers become `SymIntNode`s via `shape_env.create_symbol`.

**Phase 2 — Mode stack activation + dispatch_trace:**

```python
with ExitStack() as stack:
    stack.enter_context(decompose(self.decomposition_table))
    if self.fake_tensor_mode:
        stack.enter_context(self.fake_tensor_mode)
    stack.enter_context(self.python_dispatcher_mode)
    stack.enter_context(self.proxy_function_mode)
    stack.enter_context(self.torch_fn_metadata_mode)
    stack.enter_context(proxy_mode)
    stack.enter_context(disable_autocast_cache())
    stack.enter_context(_set_make_fx_tracer(self))

    t = dispatch_trace(
        wrap_key(func, args, self.fx_tracer, self.pre_dispatch),
        tracer=self.fx_tracer,
        concrete_args=tuple(phs),
    )
```

The `ExitStack` activates all modes in order. The activation order matters:
1. `decompose` — sets the global decomposition table
2. `fake_tensor_mode` — innermost dispatch (computes shapes)
3. `python_dispatcher_mode` — enables Python-level per-key dispatch (symbolic only)
4. `proxy_function_mode` — intercepts `__torch_function__` for pre-dispatch tracing
5. `torch_fn_metadata_mode` — records `torch.*` API metadata
6. `proxy_mode` — outermost dispatch mode that creates FX nodes

**Phase 3 — Post-processing:**

```python
# For symbolic mode, attach ShapeEnv to the GraphModule
if self.tracing_mode == "symbolic":
    t.shape_env = self.fake_tensor_mode.shape_env
return t
```

---

### 4.4 `wrap_key`

**File:** [proxy_tensor.py:1572](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L1572)

Bridges between FX's proxy world and the real/fake tensor world. Called once before
`dispatch_trace` to wrap the user function.

**Problem:** FX's `Tracer.trace()` calls the function with `Proxy` objects as arguments.
But the function calls `torch.mm(x, y)` which needs actual tensors (or FakeTensors) for the
C++ dispatcher. `wrap_key` solves this by:

1. **Before execution:** associating each `(real_tensor, proxy)` pair via `track_tensor_tree`.
   This stores the proxy inside the tensor's `proxy_slot` so that when the tensor is later
   passed to an ATen op, `ProxyTorchDispatchMode` can look up its proxy.

2. **Calling `f(*tensors)`** (not `f(*proxies)`) — the function runs with real/fake tensors.

3. **After execution:** extracting proxies from output tensors via `get_proxy_slot` and
   returning them to FX.

```python
def wrapped(*proxies):
    # Install tensor→proxy associations
    track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)
    # Run with actual tensors (ProxyTorchDispatchMode intercepts ops)
    out = f(*tensors)
    # Extract proxies from output tensors
    out = pytree.tree_map_only(Tensor, get_tensor_proxy_slot, out)
    return out
```

---

### 4.5 `dispatch_trace`

**File:** [proxy_tensor.py:1528](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L1528)

A thin wrapper around `tracer.trace(root, concrete_args)` — the standard FX tracing entry
point. After tracing completes:

1. **Dead code elimination** — removes unused nodes (with special handling: `.item()` calls
   are never eliminated since they have side effects).
2. **SymInt deduplication** — `dedupe_symints(graph)`.
3. **GraphModule construction** — `_make_graph_module(tracer.root, graph, name)`.

```python
def dispatch_trace(root, tracer, concrete_args=None):
    graph = tracer.trace(root, concrete_args)
    graph.eliminate_dead_code(impure_pred)
    dedupe_symints(graph)
    return _make_graph_module(tracer.root, graph, name)
```

---

### 4.6 `PythonKeyTracer`

**File:** [proxy_tensor.py:1318](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L1318)

Extends `torch.fx.Tracer` — the FX graph builder that creates nodes. Key customizations:

| Method | Behavior |
|---|---|
| `call_module(m, forward, args, kwargs)` | Calls `forward(*args)` directly — does NOT make modules into leaf nodes |
| `getattr(attr, attr_val, ...)` | Returns the actual value — does NOT create proxy nodes for attributes |
| `create_arg(a)` | Handles `nn.Parameter` (interns as `get_attr` node) and `SymInt` (returns constant) |
| `unwrap_proxy(e)` | Looks up the proxy for a tensor/SymInt via `get_proxy_slot` |
| `create_node(kind, target, args, kwargs)` | Creates node + saves `eager_input_vals` for stride-sensitive ops |

**Tracker dictionaries:**
- `tensor_tracker: WeakTensorKeyDictionary` — maps `Tensor → _ProxyTensor`
- `symnode_tracker: _SymNodeDict` — maps `SymInt → Proxy`
- `script_object_tracker` — maps `ScriptObject → Proxy`

These trackers are what `get_proxy_slot` / `set_proxy_slot` / `has_proxy_slot` use to
associate tensors with their FX proxy nodes.

---

### 4.7 `ProxyTorchDispatchMode`

**File:** [proxy_tensor.py:1733](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L1733)

The central piece. A `TorchDispatchMode` registered at the `PROXY` infra mode key.

**`__torch_dispatch__(func, types, args, kwargs)`:**

```python
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func == prim.device.default:
        return func(*args, **kwargs)   # device queries pass through
    return proxy_call(self, func, self.pre_dispatch, args, kwargs)
```

Every ATen operator call flows through here. The mode delegates to `proxy_call` (§4.8)
which does the actual FX node creation.

**`__sym_dispatch__(func, types, args, kwargs)`:**

Handles `SymInt`/`SymFloat`/`SymBool` operations (e.g., `operator.mul` on symbolic shapes).
Creates FX nodes for shape arithmetic so that shape expressions appear in the graph (symbolic
mode only).

**Infra mode ordering:** `ProxyTorchDispatchMode` is an *infra mode*
(`_mode_key = _TorchDispatchModeKey.PROXY`). Infra modes run **after** user-defined modes
but **before** `FakeTensorMode` (which is also infra, key=`FAKE`). This means Proxy sees the
op first, creates the FX node, then lets the op fall through to FakeTensorMode for shape
computation.

---

### 4.8 `proxy_call`

**File:** [proxy_tensor.py:1079](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L1079)

The workhorse function called by `ProxyTorchDispatchMode.__torch_dispatch__`. Three phases:

**Phase 1 — Decomposition check:**

```python
r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
if r is not NotImplemented:
    return r
```

If the op is in `CURRENT_DECOMPOSITION_TABLE`, run the decomposition instead (this
re-enters the mode, so the decomposed ops get traced individually).

**Phase 2 — Create FX node:**

```python
# Extract proxy nodes from tensor arguments
proxy_args, proxy_kwargs = ...  # via _fetch_proxies_and_all_constant_flag

# Create the FX node
proxy_out = tracer.create_proxy("call_function", func, proxy_args, proxy_kwargs)
```

**Phase 3 — Execute the real op + track outputs:**

```python
# Actually run the op (dispatches to FakeTensorMode or real kernels)
out = func(*args, **kwargs)

# Associate output tensors with their proxy nodes
track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
return out
```

The dual execution — creating an FX node AND running the real op — is what makes proxy-tensor
tracing work. The FX node captures the symbolic computation; the real execution produces
the concrete (or fake) output tensor that carries shape metadata in `node.meta["val"]`.

---

### 4.9 `FakeTensorMode`

**File:** [torch/_subclasses/fake_tensor.py:1253](../thirdparty/pytorch/torch/_subclasses/fake_tensor.py#L1253)

A `TorchDispatchMode` at the `FAKE` infra mode key. When active, **every ATen op call produces
FakeTensors** — tensors with correct metadata (shape, dtype, device, strides) but no data.

**`__torch_dispatch__` → `dispatch` → `_dispatch_impl`:**

1. Check `_DISPATCH_META_HANDLERS` for fast-path handlers.
2. Check `_DISPATCH_HANDLE_DIRECTLY` for simple attribute queries.
3. Otherwise call `_dispatch_impl` which:
   - Converts all input tensors to meta tensors
   - Runs the op's **meta kernel** (a Python function that computes output shapes)
   - Wraps the meta-tensor output as a `FakeTensor`

**Not installed in real mode.** When `tracing_mode="real"`, `fake_tensor_mode` is `None` and
the ops run on actual CUDA/CPU tensors.

**`FakeTensorConverter`:** Handles conversion between real tensors and FakeTensors. Called by
`from_tensor(real_tensor, source=...)` during `_wrap_fake`.

---

### 4.10 `TorchFunctionMetadataMode`

**File:** [proxy_tensor.py:1639](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L1639)

A lightweight `TorchFunctionMode` that intercepts `__torch_function__` (the Python-level
dispatch, before `__torch_dispatch__`). Its only job: record which user-facing torch function
was called.

```python
class TorchFunctionMetadataMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.tracer.torch_fn_metadata = func
        self.tracer.torch_fn_counts[func] += 1
        return func(*args, **kwargs)
```

This metadata is used by the FX graph to annotate nodes with the original torch API call
(e.g., `torch.mm` vs `aten.mm.default`) for debugging and source mapping.

---

### 4.11 `track_tensor_tree`

**File:** [proxy_tensor.py:847](../thirdparty/pytorch/torch/fx/experimental/proxy_tensor.py#L847)

Associates output tensors from op execution with their corresponding FX proxy nodes. This is
the glue between the "real execution" world and the "proxy/graph" world.

```python
def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):
    def wrap_with_proxy(e, proxy, constant):
        if isinstance(e, Tensor):
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)            # node.meta["val"] = FakeTensor
        elif isinstance(e, py_sym_types):
            set_meta(proxy, e)
            set_proxy_slot(e, tracer, thunkify(tracer, lambda: proxy))
        elif isinstance(e, (tuple, list)):
            for idx, ee in enumerate(e):
                wrap_with_proxy(ee, proxy[idx], ...)
    ...
```

For each output tensor `e`:
- `track_tensor(e, proxy)` stores `proxy` in `e`'s `proxy_slot` (via the tracer's
  `tensor_tracker` WeakDict).
- `set_meta(proxy, e)` stores `e` (the FakeTensor) as `proxy.node.meta["val"]` — this is
  how downstream passes (inductor, partitioner) know the shape/dtype of each node's output.

---

## 5. End-to-End Data Flow

Tracing `make_fx(fn, tracing_mode="fake")(x, W)` where `fn(x, W) = x @ W`:

```
make_fx(fn)(x, W)
│
├── _MakefxTracer.__init__()
│     decomposition_table, tracing_mode="fake", ...
│
├── _MakefxTracer.trace(fn, x, W)
│   │
│   ├── _init_modes_from_inputs(fn, (x, W))
│   │     1. fx_tracer = PythonKeyTracer()
│   │     2. fake_tensor_mode = FakeTensorMode(static_shapes=True)
│   │     3. proxy_mode = ProxyTorchDispatchMode(fx_tracer, "fake")
│   │     4. torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)
│   │
│   └── _trace_inner(fn, x, W)
│       │
│       ├── _wrap_fake((x, W))
│       │     x_fake = fake_tensor_mode.from_tensor(x)   # FakeTensor, same shape
│       │     W_fake = fake_tensor_mode.from_tensor(W)
│       │
│       ├── Enter mode stack:
│       │     decompose(table) → fake_tensor_mode → proxy_mode → metadata_mode
│       │
│       └── dispatch_trace(wrap_key(fn, (x_fake, W_fake), fx_tracer), ...)
│           │
│           ├── fx_tracer.trace(wrapped_fn, concrete_args=(PH, PH))
│           │   │
│           │   ├── Create placeholder nodes: arg0_1, arg1_1
│           │   │   Proxies: Proxy(arg0_1), Proxy(arg1_1)
│           │   │
│           │   └── Call wrapped_fn(Proxy(arg0_1), Proxy(arg1_1))
│           │       │
│           │       ├── wrap_key: track_tensor_tree(x_fake → Proxy(arg0_1))
│           │       │             track_tensor_tree(W_fake → Proxy(arg1_1))
│           │       │   x_fake.proxy_slot = Proxy(arg0_1)
│           │       │   W_fake.proxy_slot = Proxy(arg1_1)
│           │       │
│           │       ├── fn(x_fake, W_fake)  →  x_fake @ W_fake
│           │       │   │
│           │       │   ├── TorchFunctionMetadataMode.__torch_function__
│           │       │   │     tracer.torch_fn_metadata = torch.mm
│           │       │   │     → calls through
│           │       │   │
│           │       │   ├── C++ dispatcher → aten.mm.default
│           │       │   │
│           │       │   ├── ProxyTorchDispatchMode.__torch_dispatch__
│           │       │   │   → proxy_call(mode, aten.mm.default, args=(x_fake, W_fake))
│           │       │   │     │
│           │       │   │     ├── maybe_handle_decomp → NotImplemented (no decomp for mm)
│           │       │   │     │
│           │       │   │     ├── Extract proxies: x_fake → Proxy(arg0_1)
│           │       │   │     │                    W_fake → Proxy(arg1_1)
│           │       │   │     │
│           │       │   │     ├── proxy_out = tracer.create_proxy("call_function",
│           │       │   │     │       aten.mm.default, (Proxy(arg0_1), Proxy(arg1_1)))
│           │       │   │     │   → Creates FX node "mm" in graph
│           │       │   │     │
│           │       │   │     ├── out = aten.mm.default(x_fake, W_fake)
│           │       │   │     │   │
│           │       │   │     │   └── FakeTensorMode.__torch_dispatch__
│           │       │   │     │         → dispatch → _dispatch_impl
│           │       │   │     │         → meta kernel: output shape = (B, O)
│           │       │   │     │         → returns FakeTensor(shape=(B,O), dtype=float32)
│           │       │   │     │
│           │       │   │     ├── track_tensor_tree(out_fake, Proxy(mm))
│           │       │   │     │     out_fake.proxy_slot = Proxy(mm)
│           │       │   │     │     mm.meta["val"] = FakeTensor(shape=(B,O))
│           │       │   │     │
│           │       │   │     └── return out_fake
│           │       │   │
│           │       │   └── returns out_fake (the FakeTensor result)
│           │       │
│           │       └── wrap_key post: get_proxy_slot(out_fake) → Proxy(mm)
│           │           return Proxy(mm)   ← back to FX tracer
│           │
│           ├── FX tracer creates output node
│           ├── graph.eliminate_dead_code()
│           └── return GraphModule(graph)
│
└── Returns GraphModule with graph:
      arg0_1 = placeholder
      arg1_1 = placeholder
      mm = call_function[aten.mm.default](arg0_1, arg1_1)
      output = (mm,)
```

---

## 6. The Complete Context Manager Stack

When `_trace_inner` runs, the active context managers (from outermost to innermost) are:

```python
with ExitStack() as stack:
    # 1. Set the decomposition table (global variable)
    stack.enter_context(decompose(self.decomposition_table))

    # 2. FakeTensorMode — innermost dispatch, computes shapes
    #    (None for real mode)
    if self.fake_tensor_mode:
        stack.enter_context(self.fake_tensor_mode)

    # 3. Python dispatcher — enables per-dispatch-key Python dispatch
    #    (nullcontext for real/fake modes)
    stack.enter_context(self.python_dispatcher_mode)

    # 4. PreDispatchTorchFunctionMode — catches torch API calls for export
    #    (nullcontext unless pre_dispatch=True)
    stack.enter_context(self.proxy_function_mode)

    # 5. TorchFunctionMetadataMode — records torch API function name
    stack.enter_context(self.torch_fn_metadata_mode)

    # 6. ProxyTorchDispatchMode — THE mode that creates FX nodes
    stack.enter_context(proxy_mode)

    # 7. Disable autocast cache (prevents untracked tensor allocations)
    stack.enter_context(disable_autocast_cache())

    # 8. Set global _CURRENT_MAKE_FX_TRACER for nested make_fx calls
    stack.enter_context(_set_make_fx_tracer(self))
```

The **dispatch order** when an ATen op is called:

1. `TorchFunctionMetadataMode.__torch_function__` (Python-level, outermost)
2. C++ dispatcher routes to `__torch_dispatch__`
3. `ProxyTorchDispatchMode.__torch_dispatch__` (infra mode, PROXY key)
   - Creates proxy node
   - Calls `func(*args)` which re-enters dispatch
4. `FakeTensorMode.__torch_dispatch__` (infra mode, FAKE key)
   - Computes output shape/dtype via meta kernel
   - Returns FakeTensor

---

## 7. Code Map

| File | Key symbols | Role |
|---|---|---|
| `torch/fx/experimental/proxy_tensor.py` | `make_fx`, `_MakefxTracer`, `PythonKeyTracer`, `ProxyTorchDispatchMode`, `proxy_call`, `wrap_key`, `dispatch_trace`, `track_tensor_tree`, `TorchFunctionMetadataMode` | Main tracing infrastructure |
| `torch/_subclasses/fake_tensor.py` | `FakeTensorMode`, `FakeTensor`, `FakeTensorConverter` | Fake tensor dispatch (shape-only computation) |
| `torch/utils/_python_dispatch.py` | `TorchDispatchMode` | Base class for dispatch modes |
| `torch/fx/interpreter.py` | `Interpreter` | FX graph interpreter (used by `DecompositionInterpreter`) |
| `torch/fx/proxy.py` | `Proxy`, `GraphAppendingTracer` | FX proxy objects and graph builder |
| `torch/fx/_symbolic_trace.py` | `Tracer.trace()` | Base FX tracing (PythonKeyTracer inherits this) |
| `torch/_decomp/decompositions.py` | `@register_decomposition` | Operator decompositions |
| `torch/fx/experimental/symbolic_shapes.py` | `ShapeEnv` | Symbolic shape tracking (symbolic mode) |
