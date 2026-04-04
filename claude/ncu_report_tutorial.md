# NCU Report Parsing: Wafer Internals + ncu_report API Tutorial

How Wafer's `ncu_parse_report.py` parses `.ncu-rep` files, and how to use NVIDIA's `ncu_report` Python API directly for custom profiling analyses.

## Part 1: How `ncu_parse_report.py` works

**File**: `wafer-ai/wafer-ext/extension/resources/ncu-tool/ncu_parse_report.py` (1,552 lines)

### Overview

The parser takes a `.ncu-rep` file path, uses NVIDIA's `ncu_report` SWIG bindings to extract every kernel's metrics, rule results, and source correlation data, then emits a single JSON blob on stdout. The extension host captures this JSON and sends it to the React webview for rendering.

### Phase 1: Find the ncu_report module

```python
# lines 61-78
def find_ncu_python_path():
    """Search platform-specific paths for NVIDIA's ncu_report module."""
    search_paths = [
        # Linux
        "/usr/local/cuda/extras/CUPTI/ncu_python",
        "/opt/nvidia/nsight-compute/ncu_python",
        # macOS
        "/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python",
    ]
    for path in search_paths:
        if os.path.exists(os.path.join(path, "ncu_report.py")):
            return path
    return None
```

The module isn't pip-installable — it ships with Nsight Compute and requires a companion `.so` (`_ncu_report.so`) plus `libnvperf_host.so`. The parser adds the discovered path to `sys.path`, then `import ncu_report`.

### Phase 2: Safe metric extraction

GPU metrics frequently produce NaN/Infinity (unused counters, divide-by-zero in throughput ratios). Two utilities handle this:

**`SafeJSONEncoder`** (lines 32-58): Recursively walks the output dict before serialization. Replaces `float('nan')` and `float('inf')` with `None`. Without this, `json.dumps()` would produce invalid JSON (`NaN` is not valid JSON).

**`get_metric_value_safe()`** (lines 92-104):
```python
def get_metric_value_safe(action, metric_name, default=None):
    """Extract a metric value, returning default if missing or NaN."""
    try:
        metric = action.metric_by_name(metric_name)
        if metric is None or not metric.has_value():
            return default
        val = metric.value()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return default
        return val
    except Exception:
        return default
```

This is the workhorse — called hundreds of times per kernel. The `try/except` is necessary because some metric names exist in the API but throw when accessed (version mismatches, metrics not collected in the profiling pass).

### Phase 3: Metric attribute extraction

**`extract_metric_attributes()`** (lines 107-190) pulls the full metadata for a single metric:

```python
{
    "name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "value": 78.3,
    "unit": "%",
    "metric_type": "THROUGHPUT",        # from IMetric.metric_type()
    "metric_subtype": "PCT_OF_PEAK_SUSTAINED_ELAPSED",  # from IMetric.metric_subtype()
    "rollup": "AVG",                    # from IMetric.rollup_operation()
    "description": "SM throughput...",   # from IMetric.description()
    "label": "SM Throughput",           # from IMetric.label()
}
```

The function maps enum values to human-readable strings via lookup tables:
```python
TYPE_MAP = {0: "OTHER", 1: "COUNTER", 2: "RATIO", 3: "THROUGHPUT"}
SUBTYPE_MAP = {0: "NONE", 1: "PEAK_SUSTAINED", 2: "PER_CYCLE_ACTIVE", ...}
ROLLUP_MAP = {0: "AVG", 1: "MAX", 2: "MIN", 3: "SUM"}
```

### Phase 4: Rule result extraction

**`extract_rule_results()`** (lines 193-351) extracts NCU's built-in optimization recommendations:

```python
for rule_result in action.rule_results():
    result = {
        "identifier": rule_result.rule_identifier(),   # e.g. "SpeedOfLight"
        "name": rule_result.name(),                      # e.g. "GPU Speed Of Light Throughput"
        "section": rule_result.section_identifier(),
        "message": rule_result.rule_message(),           # dict: {title, message, type}
        "speedup": rule_result.speedup_estimation(),     # dict: {type, speedup}
        "focus_metrics": rule_result.focus_metrics(),     # list of {name, value, severity, info}
        "tables": rule_result.result_tables(),           # structured data tables
    }
```

The `speedup_estimation()` returns a dict like `{"type": "LOCAL", "speedup": 15.2}` — meaning "this kernel could be ~15% faster if you address this issue." The extension UI uses this to sort recommendations by impact.

`focus_metrics()` identifies which specific metrics drove the recommendation, with severity levels to highlight the most actionable ones.

### Phase 5: Main parse loop

**`parse_report()`** (lines 451-842) is the entry point:

```python
def parse_report(file_path):
    context = ncu_report.load_report(file_path)   # → IContext

    result = {"success": True, "summary": {"gpu": None, "kernels": [], ...}}

    for range_idx in range(context.num_ranges()):
        range_obj = context.range_by_idx(range_idx)

        for action_idx in range(range_obj.num_actions()):
            action = range_obj.action_by_idx(action_idx)

            kernel = {
                "name": action.name(ncu_report.IAction.NameBase_FUNCTION),
                "names": {
                    "function": action.name(ncu_report.IAction.NameBase_FUNCTION),
                    "demangled": action.name(ncu_report.IAction.NameBase_DEMANGLED),
                    "mangled": action.name(ncu_report.IAction.NameBase_MANGLED),
                },
                "duration_us": get_metric_value_safe(action, "gpu__time_duration.sum") / 1000,
                "memory_throughput_pct": get_metric_value_safe(action,
                    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
                "compute_throughput_pct": get_metric_value_safe(action,
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                "achieved_occupancy_pct": get_metric_value_safe(action,
                    "sm__warps_active.avg.pct_of_peak_sustained_active"),
                "registers_per_thread": get_metric_value_safe(action,
                    "launch__registers_per_thread"),
                "block_size": get_metric_value_safe(action, "launch__block_size"),
                "grid_size": get_metric_value_safe(action, "launch__grid_size"),
                # ... more fields
            }
```

The iteration model is: **Context → Range → Action → Metrics**.

- A **range** groups related kernel launches (e.g., from NVTX annotations)
- An **action** is a single kernel launch — the primary unit of analysis
- Each action has hundreds of **metrics** accessible by name

After extracting per-kernel data, it builds categorized metric sections and aggregates recommendations.

### Phase 6: Detail sections

**`build_details_sections()`** (lines 845-1526) constructs 8+ sections matching NCU's own UI tabs. Each section collects related metrics with fallback calculations:

```python
# GPU Speed of Light section
sol = {
    "metrics": [
        {"name": "Memory Throughput", "value": mem_throughput, "unit": "%",
         "description": "..."},
        {"name": "Compute (SM) Throughput", "value": compute_throughput, "unit": "%"},
        # ...
    ],
    "diagnostics": [...]  # text explanations when values are concerning
}
```

Example fallback: if `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` isn't available, compute L2 miss rate from `lts__t_sectors_srcunit_tex_op_read.sum` and `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`.

### Metric categorization

Metrics are binned into sections by prefix:

| Prefix | Section |
|--------|---------|
| `gpu__*` | `gpu_speed_of_light` |
| `sm__*` | `compute_workload` |
| `dram__*`, `lts__*`, `l1tex__*` | `memory_workload` |
| `launch__*` | `launch_statistics` |
| `smsp__*` | `instruction_statistics` |

### Output

The final JSON is printed to stdout. The extension host reads it, stores it in `AppState.ncuState`, and posts it to the webview via `postMessage`.

---

## Part 2: The `ncu_report` Python API

**Location**: Ships with Nsight Compute at `<ncu_install>/extras/python/`

**Files**:
- `ncu_report.py` — SWIG-generated Python wrapper (~4,000 lines)
- `_ncu_report.so` — compiled C++ bindings
- `libnvperf_host.so` — NVIDIA performance library (loaded at runtime)
- `ncu_occupancy.py` — occupancy calculator bindings
- `_ncu_occupancy.so` — compiled occupancy calculator

### Object model

```
load_report(path) → IContext
                      ├── num_ranges() → int
                      └── range_by_idx(i) → IRange
                                              ├── num_actions() → int
                                              └── action_by_idx(i) → IAction
                                                                       ├── name(base) → str
                                                                       ├── metric_by_name(name) → IMetric
                                                                       ├── metric_names() → [str]
                                                                       ├── rule_results() → [IRuleResult]
                                                                       ├── source_files() → [str]
                                                                       ├── ptx_by_pc(addr) → str
                                                                       ├── sass_by_pc(addr) → str
                                                                       └── source_info(addr) → ISourceInfo
```

### Loading a report

```python
import sys
sys.path.insert(0, "/home/jeromeku/nsight-compute-2026-01/extras/python")
import ncu_report

ctx = ncu_report.load_report("/path/to/report.ncu-rep")
```

`load_report()` is a closure-based factory (lines 1688-1813). It:
1. Finds `libnvperf_host.so` relative to the `ncu_report.py` location
2. Sets `NVPERF_HOST_PATH` environment variable
3. Calls `_ncu_report.Context_Create(file_name)` → returns an `IContext`

### IContext — the loaded report

```python
ctx.num_ranges()         # Number of profiling ranges (usually 1)
r = ctx.range_by_idx(0)  # First range → IRange
```

A range corresponds to a profiling pass. Most reports have a single range unless you used NVTX range-based profiling.

### IRange — a profiling range

```python
r.num_actions()           # Number of kernel launches in this range
a = r.action_by_idx(0)   # First kernel launch → IAction
```

### IAction — a single kernel launch

This is the primary unit of analysis. Each action represents one kernel invocation with all its collected metrics.

#### Kernel identity

```python
# Three name representations
a.name(ncu_report.IAction.NameBase_FUNCTION)    # Short name: "my_kernel"
a.name(ncu_report.IAction.NameBase_DEMANGLED)   # C++ demangled: "void my_kernel<float>(float*, ...)"
a.name(ncu_report.IAction.NameBase_MANGLED)     # Mangled: "_Z9my_kernelIfEvPT_S1_i"
```

#### Metric access

```python
# Get a specific metric
m = a.metric_by_name("gpu__time_duration.sum")
print(m.value())  # e.g., 123456.0 (nanoseconds)

# List all available metrics
names = a.metric_names()
print(len(names))  # typically 500-2000+ metrics
```

#### Rule results (optimization recommendations)

```python
for rr in a.rule_results():
    print(rr.name())                    # "GPU Speed Of Light Throughput"
    print(rr.rule_identifier())         # "SpeedOfLight"

    msg = rr.rule_message()             # dict
    print(msg["title"])                 # "Compute is more utilized..."
    print(msg["message"])              # Full recommendation text

    speedup = rr.speedup_estimation()   # dict
    print(speedup["type"])              # "LOCAL" or "GLOBAL"
    print(speedup["speedup"])           # 15.2 (percent)

    for fm in rr.focus_metrics():       # list of dicts
        print(fm["name"], fm["value"], fm["severity"])
```

#### Source correlation

```python
# List source files referenced by this kernel
files = a.source_files()     # ["/path/to/kernel.cu", ...]

# Get PTX/SASS for a specific program counter address
ptx_line = a.ptx_by_pc(address)
sass_line = a.sass_by_pc(address)

# Get source location for a PC
src = a.source_info(address)  # → ISourceInfo (file, line, column)

# Source markers (NVTX annotations in source)
markers = a.source_markers()
```

### IMetric — a single metric value

The metric object carries both the value and rich metadata:

```python
m = a.metric_by_name("sm__throughput.avg.pct_of_peak_sustained_elapsed")

# Value access
m.has_value()      # True if metric was collected
m.value()          # Auto-dispatches to correct type (float, int, or str)
m.as_double()      # Force float interpretation
m.as_uint64()      # Force integer interpretation
m.as_string()      # Force string interpretation
m.kind()           # ValueKind enum: DOUBLE, FLOAT, STRING, UINT32, UINT64

# Metadata
m.name()           # "sm__throughput.avg.pct_of_peak_sustained_elapsed"
m.label()          # "SM Throughput" (human-readable)
m.unit()           # "%"
m.description()    # Full description text

# Classification
m.metric_type()    # MetricType enum: OTHER(0), COUNTER(1), RATIO(2), THROUGHPUT(3)
m.metric_subtype() # MetricSubtype enum: NONE(0), PEAK_SUSTAINED(1),
                   #   PER_CYCLE_ACTIVE(2), PCT_OF_PEAK_SUSTAINED_ELAPSED(3), ...
m.rollup_operation()  # RollupOperation enum: AVG(0), MAX(1), MIN(2), SUM(3)
```

#### Value dispatch

`IMetric.value()` is a convenience method that checks `kind()` and calls the appropriate typed accessor:

```python
def value(self, idx=None):
    k = self.kind()
    if k == ValueKind_DOUBLE or k == ValueKind_FLOAT:
        return self.as_double(idx) if idx is not None else self.as_double()
    elif k == ValueKind_UINT64 or k == ValueKind_UINT32:
        return self.as_uint64(idx) if idx is not None else self.as_uint64()
    elif k == ValueKind_STRING:
        return self.as_string(idx) if idx is not None else self.as_string()
```

#### Instanced metrics

Some metrics have per-instance data (e.g., per-SM or per-PC values):

```python
m.num_instances()         # Number of instances (0 = scalar metric)
m.has_correlation_ids()   # True if instances map to specific PCs/SMs

# IMPORTANT: correlation_ids() returns an IMetric proxy, NOT a list.
# Access individual correlation IDs via .value(i) on the returned IMetric.
corr_ids = m.correlation_ids()  # → IMetric (not a list!)

# Access per-instance values and their correlated PCs
for i in range(m.num_instances()):
    pc = corr_ids.value(i)    # PC address for instance i
    val = m.value(i)          # Metric value for instance i
```

### Metric naming convention

NCU metric names follow a structured pattern:

```
<unit>__<counter>.<rollup>.<subtype>
```

Examples:
```
gpu__time_duration.sum                                    # GPU time, summed
sm__throughput.avg.pct_of_peak_sustained_elapsed          # SM throughput, avg, as % of peak
dram__bytes_read.sum                                      # DRAM bytes read, summed
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum           # L1 texture sectors for global loads
launch__registers_per_thread                              # Launch param (no rollup)
sm__warps_active.avg.pct_of_peak_sustained_active        # Active warps as % of peak
```

Unit prefixes and what they measure:

| Prefix | Hardware unit |
|--------|--------------|
| `gpu__` | Whole GPU (time, frequency, throughput) |
| `sm__` | Streaming multiprocessor |
| `smsp__` | SM sub-partition |
| `dram__` | Device memory (HBM/GDDR) |
| `lts__` | L2 cache |
| `l1tex__` | L1/texture cache |
| `launch__` | Kernel launch parameters |

---

## Part 3: Tutorial — custom analyses

### Example 1: Basic kernel summary

```python
import sys
sys.path.insert(0, "/home/jeromeku/nsight-compute-2026-01/extras/python")
import ncu_report

ctx = ncu_report.load_report("report.ncu-rep")

for ri in range(ctx.num_ranges()):
    r = ctx.range_by_idx(ri)
    for ai in range(r.num_actions()):
        a = r.action_by_idx(ai)

        name = a.name(ncu_report.IAction.NameBase_FUNCTION)
        dur = a.metric_by_name("gpu__time_duration.sum")
        mem = a.metric_by_name("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
        comp = a.metric_by_name("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        occ = a.metric_by_name("sm__warps_active.avg.pct_of_peak_sustained_active")

        print(f"Kernel: {name}")
        print(f"  Duration:   {dur.value() / 1000:.1f} us")
        print(f"  Memory:     {mem.value():.1f}%")
        print(f"  Compute:    {comp.value():.1f}%")
        print(f"  Occupancy:  {occ.value():.1f}%")
```

### Example 2: Memory hierarchy analysis

```python
def memory_analysis(action):
    """Analyze memory subsystem utilization for a kernel."""

    def safe_val(name, default=0.0):
        m = action.metric_by_name(name)
        if m is None or not m.has_value():
            return default
        v = m.value()
        return default if (isinstance(v, float) and (v != v or abs(v) == float('inf'))) else v

    # DRAM
    dram_read = safe_val("dram__bytes_read.sum")
    dram_write = safe_val("dram__bytes_write.sum")
    dram_throughput = safe_val("dram__throughput.avg.pct_of_peak_sustained_elapsed")

    # L2 cache
    l2_hit_rate = safe_val("lts__t_sector_hit_rate.pct")
    l2_read = safe_val("lts__t_sectors_srcunit_tex_op_read.sum")
    l2_write = safe_val("lts__t_sectors_srcunit_tex_op_write.sum")

    # L1 cache
    l1_hit_rate = safe_val("l1tex__t_sector_hit_rate.pct")

    # Shared memory
    shared_load = safe_val("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")
    shared_store = safe_val("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum")
    shared_bank_conflicts = safe_val(
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")

    print(f"DRAM: {dram_read/1e6:.1f} MB read, {dram_write/1e6:.1f} MB write, "
          f"{dram_throughput:.1f}% of peak")
    print(f"L2:   hit rate {l2_hit_rate:.1f}%")
    print(f"L1:   hit rate {l1_hit_rate:.1f}%")
    print(f"SMEM: {shared_load:.0f} load wavefronts, {shared_store:.0f} store wavefronts, "
          f"{shared_bank_conflicts:.0f} bank conflicts")
```

### Example 3: Compare two kernels (before/after optimization)

```python
def compare_kernels(report_before, report_after, kernel_name=None):
    """Compare metrics between two NCU reports."""

    key_metrics = [
        ("gpu__time_duration.sum", "Duration (ns)", 1.0),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "Compute %", 1.0),
        ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Memory %", 1.0),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "Occupancy %", 1.0),
        ("launch__registers_per_thread", "Regs/thread", 1.0),
        ("dram__bytes_read.sum", "DRAM Read (B)", 1.0),
    ]

    def get_action(report_path, name=None):
        ctx = ncu_report.load_report(report_path)
        r = ctx.range_by_idx(0)
        if name:
            for i in range(r.num_actions()):
                a = r.action_by_idx(i)
                if name in a.name(ncu_report.IAction.NameBase_DEMANGLED):
                    return a
        return r.action_by_idx(0)

    a_before = get_action(report_before, kernel_name)
    a_after = get_action(report_after, kernel_name)

    print(f"{'Metric':<40} {'Before':>12} {'After':>12} {'Change':>10}")
    print("-" * 76)

    for metric_name, label, scale in key_metrics:
        m1 = a_before.metric_by_name(metric_name)
        m2 = a_after.metric_by_name(metric_name)
        v1 = m1.value() * scale if m1 and m1.has_value() else 0
        v2 = m2.value() * scale if m2 and m2.has_value() else 0

        if v1 > 0:
            pct = ((v2 - v1) / v1) * 100
            print(f"{label:<40} {v1:>12.1f} {v2:>12.1f} {pct:>+9.1f}%")
        else:
            print(f"{label:<40} {v1:>12.1f} {v2:>12.1f}       N/A")
```

### Example 4: Extract all metrics to CSV

```python
import csv

def export_all_metrics(report_path, csv_path):
    """Export every metric from every kernel to CSV for offline analysis."""
    ctx = ncu_report.load_report(report_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "metric", "value", "unit", "type", "subtype", "rollup"])

        TYPE_MAP = {0: "OTHER", 1: "COUNTER", 2: "RATIO", 3: "THROUGHPUT"}
        ROLLUP_MAP = {0: "AVG", 1: "MAX", 2: "MIN", 3: "SUM"}

        for ri in range(ctx.num_ranges()):
            r = ctx.range_by_idx(ri)
            for ai in range(r.num_actions()):
                a = r.action_by_idx(ai)
                name = a.name(ncu_report.IAction.NameBase_FUNCTION)

                for metric_name in a.metric_names():
                    m = a.metric_by_name(metric_name)
                    if m is None or not m.has_value():
                        continue

                    try:
                        val = m.value()
                        # Sanitize NaN/Inf
                        if isinstance(val, float) and (val != val or abs(val) == float('inf')):
                            val = None
                    except Exception:
                        continue

                    writer.writerow([
                        name, metric_name, val,
                        m.unit(),
                        TYPE_MAP.get(m.metric_type(), "?"),
                        str(m.metric_subtype()),
                        ROLLUP_MAP.get(m.rollup_operation(), "?"),
                    ])
```

### Example 5: Actionable recommendations sorted by speedup

```python
def top_recommendations(report_path, top_n=10):
    """Print the top-N optimization recommendations by estimated speedup."""
    ctx = ncu_report.load_report(report_path)

    all_recs = []
    for ri in range(ctx.num_ranges()):
        r = ctx.range_by_idx(ri)
        for ai in range(r.num_actions()):
            a = r.action_by_idx(ai)
            kernel_name = a.name(ncu_report.IAction.NameBase_FUNCTION)
            dur_m = a.metric_by_name("gpu__time_duration.sum")
            dur = dur_m.value() / 1000 if dur_m and dur_m.has_value() else 0

            for rr in a.rule_results():
                speedup = rr.speedup_estimation()
                if speedup and speedup.get("speedup", 0) > 0:
                    msg = rr.rule_message()
                    all_recs.append({
                        "kernel": kernel_name,
                        "duration_us": dur,
                        "rule": rr.name(),
                        "speedup_pct": speedup["speedup"],
                        "scope": speedup["type"],    # LOCAL or GLOBAL
                        "message": msg.get("message", ""),
                        "focus": [fm["name"] for fm in rr.focus_metrics()],
                    })

    # Sort by speedup descending
    all_recs.sort(key=lambda x: x["speedup_pct"], reverse=True)

    for i, rec in enumerate(all_recs[:top_n]):
        print(f"\n#{i+1}: {rec['rule']} ({rec['scope']} speedup: {rec['speedup_pct']:.1f}%)")
        print(f"  Kernel: {rec['kernel']} ({rec['duration_us']:.1f} us)")
        print(f"  Focus metrics: {', '.join(rec['focus'])}")
        print(f"  {rec['message'][:200]}...")
```

### Example 6: Roofline data extraction

```python
def roofline_data(action):
    """Extract data needed for a roofline plot."""

    def safe_val(name):
        m = action.metric_by_name(name)
        return m.value() if m and m.has_value() else 0.0

    # Arithmetic intensity = FLOPs / bytes moved
    flops = safe_val("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") + \
            safe_val("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") + \
            safe_val("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") * 2  # FMA = 2 FLOPs

    dram_bytes = safe_val("dram__bytes.sum")

    ai = flops / dram_bytes if dram_bytes > 0 else 0

    # Achieved performance
    duration_s = safe_val("gpu__time_duration.sum") * 1e-9  # ns → s
    gflops = flops / duration_s / 1e9 if duration_s > 0 else 0

    # Peak theoretical (from GPU properties)
    peak_mem_bw = safe_val("dram__bytes.sum.peak_sustained")  # bytes/s
    peak_flops = safe_val("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")

    return {
        "arithmetic_intensity": ai,
        "achieved_gflops": gflops,
        "peak_mem_bw_gbs": peak_mem_bw / 1e9,
        "flops": flops,
        "dram_bytes": dram_bytes,
    }
```

### Example 7: Source-correlated hotspot ranking

```python
from tabulate import tabulate

def source_hotspots(action, metric_name="inst_executed", asm="ptx", top_n=20):
    """Rank instructions by a chosen per-PC metric, with source correlation.

    Args:
        action: IAction from an NCU report.
        metric_name: Any instanced metric or group metric to rank by.
        asm: "ptx" or "sass" — which assembly to show. Default "ptx".
        top_n: Number of top rows to return.

    Returns:
        list[dict]: Rows sorted by metric_name descending. Keys:
            "file", "line", "sass", "ptx", metric_name.
    """
    metric = action[metric_name]

    # Group metrics expand to comma-separated sub-metric names
    if "group" in metric_name:
        sub_names = metric.value().split(",")
        sub_metrics = [action[n] for n in sub_names]
    else:
        sub_metrics = [metric]

    # Build PC → aggregated value.
    # IMPORTANT: correlation_ids() returns an IMetric proxy, NOT a list.
    # Access individual PCs via correlation_ids().value(i).
    pc_to_value = {}
    for m in sub_metrics:
        if m.num_instances() == 0 or not m.has_correlation_ids():
            continue
        corr_ids = m.correlation_ids()
        for i in range(m.num_instances()):
            pc = corr_ids.value(i)      # IMetric.value(i), not list[i]
            val = m.value(i)
            if val and val > 0:
                pc_to_value[pc] = pc_to_value.get(pc, 0) + val

    if not pc_to_value:
        return []

    # Build rows with source correlation
    rows = []
    for pc, val in pc_to_value.items():
        src_info = action.source_info(pc)
        rows.append({
            "file": src_info.file_name().split("/")[-1] if src_info else "",
            "line": src_info.line() if src_info else 0,
            "sass": action.sass_by_pc(pc) or "",
            "ptx": action.ptx_by_pc(pc) or "",
            metric_name: val,
        })

    rows.sort(key=lambda r: r[metric_name], reverse=True)
    top = rows[:top_n]

    # Format with tabulate
    asm_header = "PTX" if asm == "ptx" else "SASS"
    headers = ["File", "Line", asm_header, metric_name]
    table_data = [[r["file"], r["line"], r[asm], r[metric_name]] for r in top]
    print(tabulate(table_data, headers=headers, tablefmt="simple",
                   colalign=("left", "right", "left", "right")))
    return top

# Usage:
# source_hotspots(action)                                              # default
# source_hotspots(action, "group:smsp__pcsamp_warp_stall_reasons")     # stalls
# source_hotspots(action, "inst_executed", asm="sass")                 # SASS view
```

### Example 8: Full source analysis (NCU-UI Source tab equivalent)

This function recreates NCU-UI's interleaved Source ↔ ASM view: each source line is
followed by its correlated SASS/PTX instructions, with all per-PC metrics attached.
The returned list of dicts is sortable by any column.

```python
from collections import defaultdict, OrderedDict
from tabulate import tabulate

def source_analysis(action, metric_names=None, asm="sass"):
    """Build a full source ↔ ASM table with per-line/per-PC metrics.

    Args:
        action: IAction from an NCU report.
        metric_names: List of instanced metric names. If None, uses all available.
        asm: "sass", "ptx", or "both".

    Returns:
        list[dict]: Rows in source order. Each dict has:
            "type"   — "source" or "asm"
            "file"   — source file basename
            "line"   — source line number
            "source" — source text (source rows only)
            "sass"   — SASS text (asm rows only)
            "ptx"    — PTX text (asm rows only)
            + one key per metric → value (aggregated for source rows, raw for asm)
    """
    # Discover available instanced metrics if not specified
    if metric_names is None:
        metric_names = []
        for name in action.metric_names():
            try:
                m = action.metric_by_name(name)
                if m and m.num_instances() > 0 and m.has_correlation_ids():
                    metric_names.append(name)
            except Exception:
                continue

    # Extract per-PC data for each metric
    # IMPORTANT: correlation_ids() returns an IMetric proxy.
    # Access PCs via corr_ids.value(i), not corr_ids[i].
    metric_data = OrderedDict()
    all_pcs = set()
    for mname in metric_names:
        m = action.metric_by_name(mname)
        if m is None or m.num_instances() == 0 or not m.has_correlation_ids():
            continue
        corr_ids = m.correlation_ids()
        pc_to_val = {}
        for i in range(m.num_instances()):
            pc = corr_ids.value(i)
            val = m.value(i)
            if pc is not None and val is not None:
                pc_to_val[pc] = pc_to_val.get(pc, 0) + val
        if pc_to_val:
            metric_data[mname] = pc_to_val
            all_pcs.update(pc_to_val.keys())

    metric_names = list(metric_data.keys())
    if not all_pcs:
        return []

    # Map PCs to source + ASM
    pc_info = {}
    for pc in all_pcs:
        src = action.source_info(pc)
        pc_info[pc] = {
            "file": src.file_name() if src else "",
            "line": src.line() if src else 0,
            "sass": action.sass_by_pc(pc) or "",
            "ptx": action.ptx_by_pc(pc) or "",
        }

    # Get source file contents
    try:
        source_files = dict(action.source_files())
    except Exception:
        source_files = {}

    # Group PCs by (file, line)
    line_pcs = defaultdict(list)
    for pc in sorted(all_pcs):
        key = (pc_info[pc]["file"], pc_info[pc]["line"])
        line_pcs[key].append(pc)

    # Aggregate metrics to source-line level
    line_agg = defaultdict(dict)
    for pc in all_pcs:
        key = (pc_info[pc]["file"], pc_info[pc]["line"])
        for mname, per_pc in metric_data.items():
            if pc in per_pc:
                prev = line_agg[key].get(mname, 0)
                line_agg[key][mname] = prev + per_pc[pc]

    # Build interleaved rows: source line → asm sub-rows
    rows = []
    for src_file in sorted({f for (f, _) in line_pcs if f}):
        file_short = src_file.split("/")[-1] if "/" in src_file else src_file
        try:
            content = source_files[src_file] if src_file in source_files else ""
        except Exception:
            content = ""
        src_lines = content.split("\n") if content else []

        for ln in sorted({l for (f, l) in line_pcs if f == src_file and l > 0}):
            key = (src_file, ln)
            src_text = src_lines[ln - 1].rstrip() if 0 < ln <= len(src_lines) else ""

            # Source row with aggregated metrics
            src_row = {"type": "source", "file": file_short, "line": ln,
                       "source": src_text, "sass": "", "ptx": ""}
            for mname in metric_names:
                src_row[mname] = line_agg[key].get(mname)
            rows.append(src_row)

            # ASM sub-rows with per-PC metrics
            for pc in line_pcs[key]:
                info = pc_info[pc]
                asm_row = {"type": "asm", "file": file_short, "line": ln,
                           "source": "", "sass": info["sass"], "ptx": info["ptx"]}
                for mname in metric_names:
                    asm_row[mname] = metric_data[mname].get(pc)
                rows.append(asm_row)

    return rows


def print_source_analysis(rows, metric_names, asm="sass"):
    """Print source_analysis() rows using tabulate."""
    headers = ["Line", "Source"]
    if asm in ("sass", "both"):
        headers.append("SASS")
    if asm in ("ptx", "both"):
        headers.append("PTX")
    headers.extend(metric_names)

    def fmt(v):
        if v is None or v == 0 or v == 0.0:
            return ""
        return f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 1 else str(v)

    table = []
    for r in rows:
        cells = [r["line"] if r["type"] == "source" else "",
                 r["source"][:60]]
        if asm in ("sass", "both"):
            cells.append(r.get("sass", "")[:50])
        if asm in ("ptx", "both"):
            cells.append(r.get("ptx", "")[:50])
        cells.extend(fmt(r.get(m)) for m in metric_names)
        table.append(cells)

    print(tabulate(table, headers=headers, tablefmt="simple",
                   colalign=["right", "left"] +
                            (["left"] if asm in ("sass", "both") else []) +
                            (["left"] if asm in ("ptx", "both") else []) +
                            ["right"] * len(metric_names)))

# Usage:
# rows = source_analysis(action, ["inst_executed", "thread_inst_executed_true"])
# print_source_analysis(rows, ["inst_executed", "thread_inst_executed_true"])
#
# # Sort by any metric:
# rows.sort(key=lambda r: r.get("inst_executed", 0), reverse=True)
#
# # Filter to asm rows only:
# asm_rows = [r for r in rows if r["type"] == "asm"]
```

See `experiments/ncu_source_view.py` for the full implementation with CLI, metric presets
(default/memory/stalls/instructions/all), string-typed metrics (Addr Space, Access Op),
and `--hotspots` / `--asm-only` / `--lines` filtering options.

---

## Part 4: The ncu_occupancy module

**File**: `ncu_occupancy.py` + `_ncu_occupancy.so`

SWIG-generated bindings for NVIDIA's occupancy calculator. Computes theoretical occupancy based on kernel launch parameters and GPU architecture properties.

```python
import ncu_occupancy

# Typically used in conjunction with metrics from IAction:
# - launch__registers_per_thread
# - launch__block_size
# - sm__sass_data_bytes_mem_shared_op_ld.sum (shared memory)
# - GPU compute capability
```

The occupancy calculator takes register count, shared memory, block size, and compute capability, then returns:
- **Theoretical occupancy**: Maximum possible warps per SM given resource usage
- **Active warps per SM**: Based on register and shared memory limitations
- **Limiting factor**: Whether registers, shared memory, or block size is the bottleneck

This is the same calculation that `cuda-calculator` (the Excel-based occupancy calculator) performs, exposed programmatically.

---

## Key implementation patterns

### NaN guard pattern

Used throughout Wafer's parser and recommended for any custom analysis:

```python
def safe_metric(action, name, default=None):
    try:
        m = action.metric_by_name(name)
        if m is None or not m.has_value():
            return default
        v = m.value()
        if isinstance(v, float) and (v != v or abs(v) == float('inf')):
            return default
        return v
    except Exception:
        return default
```

GPU counters produce NaN when: a counter is disabled, not applicable to the architecture, not collected in the profiling section set, or when computing a ratio where the denominator is zero.

### Metric discovery

When you don't know the exact metric name:

```python
# Search metrics by substring
def find_metrics(action, pattern):
    return [n for n in action.metric_names() if pattern in n]

# Example: find all DRAM-related metrics
dram_metrics = find_metrics(action, "dram__")
# → ['dram__bytes.sum', 'dram__bytes_read.sum', 'dram__throughput.avg.pct_of_peak_sustained_elapsed', ...]
```

### Report loading gotcha

`ncu_report.load_report()` auto-discovers `libnvperf_host.so` by looking relative to its own file location. If you've copied `ncu_report.py` somewhere else without the companion `.so` files, it will fail. Always use the module in-place from the Nsight Compute installation:

```python
sys.path.insert(0, "/path/to/nsight-compute/extras/python")
import ncu_report   # finds _ncu_report.so and libnvperf_host.so relative to itself
```

---

## Code references

| Component | Path |
|-----------|------|
| Wafer NCU parser | `wafer-ai/wafer-ext/extension/resources/ncu-tool/ncu_parse_report.py` |
| ncu_report Python bindings | `/home/jeromeku/nsight-compute-2026-01/extras/python/ncu_report.py` |
| ncu_occupancy bindings | `/home/jeromeku/nsight-compute-2026-01/extras/python/ncu_occupancy.py` |
| Compiled C++ bindings | `/home/jeromeku/nsight-compute-2026-01/extras/python/_ncu_report.so` |
| NCU documentation | [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/) |
