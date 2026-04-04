# Wafer VSCode Extension Architecture: NCU Report Viewing

How the Wafer extension opens, parses, and displays `.ncu-rep` (NVIDIA Nsight Compute) profiling reports inside VSCode / Cursor.

## High-level data flow

```
User opens .ncu-rep (via sidebar file picker)
        |
        v
  [Extension Host — Node.js]
  pages/ncu/extension/handlers.ts
    handleParseNcuReport()
        |
        v
  [Tool Handler — Node.js]
  services/tools/ncu/ncuHandler.ts
    parseReport(filePath)
        |
        v
  [Python subprocess]
  resources/ncu-tool/ncu_parse_report.py  <report.ncu-rep>
    uses NVIDIA's ncu_report Python API
        |
        v
  [stdout JSON]
  { success, summary: { gpu, kernels[], sections{} } }
        |
        v
  [Extension Host]
  AppState.ncuState.withReportData(parsed)
  webview.postMessage({ type: 'NCU_REPORT_PARSED', data })
        |
        v
  [Webview — React in browser sandbox]
  pages/ncu/ui/NcuPanel.tsx
    Summary table, detail sections, recommendations
```

## Extension architecture

The extension runs in two isolated environments connected by `postMessage`:

1. **Extension Host** (Node.js) — file system access, process spawning, VSCode APIs
2. **Webview** (browser) — React UI rendered in an `<iframe>`

All TypeScript source is compiled and shipped as:
- `dist/extension.js` — extension host bundle (esbuild)
- `dist/webview/panel.js` — React webview bundle (Vite)

Source is not included in the distribution. Only the Python parser is shipped as readable source.

### Layer structure

```
src/
├── pages/ncu/
│   ├── extension/           # Node.js: message handlers, state
│   │   ├── handlers.ts      # handleCheckNcuInstallation, handleParseNcuReport, ...
│   │   ├── state.ts         # NCUState (immutable, frozen)
│   │   └── panel.ts         # Panel setup
│   └── ui/                  # React: webview components
│       ├── NcuPanel.tsx     # Main report UI
│       ├── NcuTool.tsx      # Sidebar tool selector
│       └── tabs/            # Summary, Details, Metrics, Recommendations
│
├── services/tools/ncu/
│   ├── INCUHandler.ts       # Interface
│   └── ncuHandler.ts        # Stateless handler class (DI)
│
├── services/routing/
│   └── registry.ts          # Maps message commands → handler functions
│
├── providers/
│   ├── view/waferViewProvider.ts    # Main sidebar webview provider
│   └── customEditors/               # Custom editor providers (planned)
│
└── resources/ncu-tool/
    └── ncu_parse_report.py  # Python parser (source included)
```

### Dependency injection

All services receive dependencies via constructor injection. No singletons.

```
activate()
  → createExtensionDependencies(context)    // DI container
  → NCUHandler(extensionCtx, workspaceCtx, envCtx, configCtx, ...)
  → WaferViewProvider(extensionDeps, workspaceManager, ncuHandler, ...)
```

### Message routing

Webview sends `{ command: 'parseNcuReport', filePath: '...' }`.
Extension host looks up handler in a `Map<string, MessageHandler>`:

```typescript
// services/routing/registry.ts
messageHandlers.set('checkNcuInstallation', handleCheckNcuInstallation);
messageHandlers.set('parseNcuReport', handleParseNcuReport);
```

Handler signature:

```typescript
async function handleParseNcuReport(
  message: WebviewMessage,
  deps: RouterDependencies   // appState, ncuHandler, webview, ...
): Promise<void>
```

### State management

Immutable state classes with `with*` copy-on-write methods:

```typescript
class NCUState {
  readonly installed: boolean;
  readonly reportData: NCUReportData | null;
  // ...
  withReportData(data): NCUState { return new NCUState({...this, reportData: data}); }
}
```

State is held in `WaferViewProvider.appState` and threaded through `RouterDependencies`.

## Python report parser

**File**: `resources/ncu-tool/ncu_parse_report.py` (1,552 lines)

This is the only component with readable source in the distribution. It uses NVIDIA's `ncu_report` Python API (bundled with Nsight Compute, not pip-installable).

### How it finds the ncu_report module

Searches platform-specific paths:

| Platform | Path |
|----------|------|
| macOS | `/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python` |
| Linux | `/usr/local/cuda/extras/CUPTI/ncu_python` |
| Linux (alt) | `/opt/nvidia/nsight-compute/ncu_python` |

Adds the path to `sys.path`, then `import ncu_report`.

### What it extracts

For each kernel launch (range → action in NCU terminology):

| Data | NCU API / Metric |
|------|-----------------|
| Kernel name | `action.name()` with FUNCTION / DEMANGLED / MANGLED bases |
| Duration | `gpu__time_duration.sum` (ns → us) |
| Compute throughput % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput % | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput % | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| Occupancy % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers/thread | `launch__registers_per_thread` |
| Block/grid size | `launch__block_size`, `launch__grid_size` |
| Recommendations | `action.rule_results()` with speedup estimates |
| All metrics | `action.metric_names()` → categorized by prefix |
| Source files | `action.source_files()` |
| PTX by PC | `action.ptx_by_pc()` — source correlation |
| SASS by PC | `action.sass_by_pc()` — source correlation |

### Metric categorization

Metrics are categorized by prefix into NCU sections:

```
gpu__*           → gpu_speed_of_light
sm__*            → compute_workload
dram__*, lts__*,
  l1tex__*       → memory_workload
launch__*        → launch_statistics
smsp__*          → instruction_statistics
```

### Detail sections

`build_details_sections()` constructs 8+ sections matching NCU's own UI tabs:

1. **GPU Speed of Light** — memory/compute throughput, frequency, cycle counts
2. **PM Sampling** — pass groups, buffer size
3. **Compute Workload** — issue slots, SM busy, IPC
4. **Memory Workload** — L1/L2 hit rates, DRAM bandwidth, compression ratios
5. **Launch Statistics** — block/grid size, registers, shared memory config
6. **Occupancy** — active/theoretical warps per SM
7. **Scheduler Statistics** — eligible warps, issue activity
8. **Warp State Statistics** — cycles per instruction, thread utilization
9. **Instruction Statistics** — executed/issued instruction counts

Each section has fallback calculations when direct metrics are unavailable
(e.g., computing L2 compression ratio from input/output sector counts).

### NaN/Inf handling

GPU metrics frequently produce NaN/Infinity (unused counters, divide-by-zero in ratios).
`SafeJSONEncoder` recursively sanitizes the output, replacing all NaN/Inf with `null`.

### Output JSON structure

```json
{
  "success": true,
  "output_file": "/path/to/report.ncu-rep",
  "summary": {
    "gpu": "NVIDIA H100 80GB HBM3",
    "kernels": [
      {
        "name": "my_kernel",
        "names": { "function": "...", "demangled": "...", "mangled": "..." },
        "duration_us": 123.4,
        "memory_throughput_pct": 45.2,
        "compute_throughput_pct": 78.3,
        "achieved_occupancy_pct": 60.5,
        "registers_per_thread": 24,
        "block_size": 256,
        "grid_size": 128,
        "estimated_speedup_pct": 15.0,
        "recommendations": ["OPT Low occupancy...", "INF Memory bound..."],
        "rule_results": [{ "message": "...", "speedup": 15.0, "focus_metrics": [...] }],
        "metrics_by_section": { "gpu_speed_of_light": {...}, ... },
        "all_kernel_metrics": { "metric_name": { "value": ..., "unit": "..." } }
      }
    ],
    "recommendations": [...],
    "all_metrics": {...},
    "sections": {
      "gpu_speed_of_light": { "metrics": [...], "diagnostics": [...] },
      "memory_workload": { "metrics": [...] },
      ...
    }
  }
}
```

## Python environment management

The extension bundles `uv` (v0.9.27) for all platforms:

```
resources/bin/
├── uv-linux-x86_64/uv
├── uv-darwin-x86_64/uv
├── uv-darwin-arm64/uv
└── uv-windows-x86_64/uv.exe
```

`uv` manages isolated Python virtual environments for the parser and the bundled
`wafer-ai` Python package. This avoids polluting the user's system Python.

## NCU installation detection

The extension checks for `ncu` on PATH at several locations (extracted from compiled JS):

```
/usr/bin/ncu
/usr/local/bin/ncu
/usr/local/cuda/bin/ncu
/opt/nvidia/nsight-compute/ncu
/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/ncu    (macOS)
C:\Program Files\NVIDIA Corporation\Nsight Compute\ncu.exe    (Windows)
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\ncu.exe
```

## Current limitations

- **No custom editor provider** — `.ncu-rep` files cannot be opened by double-clicking.
  Users must select the file via the Wafer sidebar's file picker.
  The architecture docs reference `providers/customEditors/ncuEditorProvider.ts` as planned
  but `package.json` has no `customEditors` contribution point.

- **Source not included** — all TypeScript source is compiled. Only the Python parser,
  architecture docs, and bundled `wafer-ai` Python package are readable.

- **NCU Python API required** — the parser needs NVIDIA's `ncu_report` module,
  which is bundled with Nsight Compute (not pip-installable). On systems without
  Nsight Compute installed, parsing falls back to error.

## Wafer-ai bundled backend

The extension ships a full Python package at `bundled/wafer-ai/` that includes:

- **Kernel optimization agent** — AI agent that can run NCU, analyze results, write code
- **NCU profiling utilities** (`wafer/core/kernel/utils.py`) — builds `ncu` CLI commands,
  runs profiling, supports both binary `.ncu-rep` and CSV output
- **Trace comparison** (`wafer/core/profiling/trace_compare/`) — compare before/after profiles
- **CLI skills** — b200-guide, trace analysis templates
- **Perfetto integration** — trace processor for system-level analysis

The extension host invokes these via `callWaferCore()` which spawns
`uv run wafer <subcommand>` in the bundled venv.

## Code references

| Component | Path |
|-----------|------|
| Python parser | `resources/ncu-tool/ncu_parse_report.py` |
| Architecture doc | `ARCHITECTURE.md` |
| Tooling architecture | `TOOLING_ARCHITECTURE.md` |
| Package manifest | `package.json` |
| Development guide | `DEVELOPMENT.md` |
| Bundled uv | `resources/bin/` |
| Bundled wafer-ai | `bundled/wafer-ai/` |
| Compiled extension | `dist/extension.js` |
| Compiled webview | `dist/webview/panel.js` |
