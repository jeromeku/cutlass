#!/usr/bin/env python3
"""
NCU source-correlated metric viewer — programmatic equivalent of NCU-UI's Source tab.

Provides two main functions:
    source_analysis()  — Build a full source ↔ ASM table with all per-PC metrics (sortable)
    source_hotspots()  — Rank source/ASM lines by a chosen metric

Usage:
    python ncu_source_view.py <report.ncu-rep>                          # interleaved source + SASS view
    python ncu_source_view.py <report.ncu-rep> --asm ptx                # use PTX instead of SASS
    python ncu_source_view.py <report.ncu-rep> --hotspots               # top instructions by inst_executed
    python ncu_source_view.py <report.ncu-rep> --hotspots --sort-by smsp__pcsamp_warps_issue_stalled_barrier
    python ncu_source_view.py <report.ncu-rep> --preset stalls          # stall-focused metrics
    python ncu_source_view.py <report.ncu-rep> --preset memory          # memory-focused metrics
    python ncu_source_view.py <report.ncu-rep> --catalog                # list all per-PC metrics
    python ncu_source_view.py <report.ncu-rep> --metrics inst_executed,thread_inst_executed_true
    python ncu_source_view.py <report.ncu-rep> --lines 77,120,158       # filter to specific lines
    python ncu_source_view.py <report.ncu-rep> --asm-only               # skip source, show ASM only
"""

import sys
import os
import argparse
import math
from collections import defaultdict, OrderedDict

from tabulate import tabulate

# ─── NCU Python path discovery ───
NCU_PYTHON_PATHS = [
    "/home/jeromeku/nsight-compute-2026-01/extras/python",
    "/usr/local/cuda/extras/CUPTI/ncu_python",
    "/opt/nvidia/nsight-compute/ncu_python",
]

for p in NCU_PYTHON_PATHS:
    if os.path.exists(os.path.join(p, "ncu_report.py")):
        sys.path.insert(0, p)
        break

import ncu_report


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "default": [
        "inst_executed",
        "thread_inst_executed_true",
        "derived__avg_thread_executed_true",
        "memory_l2_theoretical_sectors_global",
        "derived__memory_l2_theoretical_sectors_global_excessive",
    ],
    "memory": [
        "inst_executed",
        "thread_inst_executed_true",
        "memory_type",
        "memory_access_type",
        "memory_access_size_type",
        "memory_l1_wavefronts_shared",
        "memory_l1_wavefronts_shared_ideal",
        "derived__memory_l1_wavefronts_shared_excessive",
        "derived__memory_l1_conflicts_shared_nway",
        "memory_l1_tag_requests_global",
        "memory_l2_theoretical_sectors_global",
        "memory_l2_theoretical_sectors_global_ideal",
        "derived__memory_l2_theoretical_sectors_global_excessive",
    ],
    "instructions": [
        "inst_executed",
        "thread_inst_executed",
        "thread_inst_executed_true",
        "derived__avg_thread_executed",
        "derived__avg_thread_executed_true",
    ],
    "stalls": [
        "inst_executed",
        "smsp__pcsamp_sample_count",
        "smsp__pcsamp_warps_issue_stalled_barrier",
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
        "smsp__pcsamp_warps_issue_stalled_wait",
        "smsp__pcsamp_warps_issue_stalled_lg_throttle",
        "smsp__pcsamp_warps_issue_stalled_tex_throttle",
        "smsp__pcsamp_warps_issue_stalled_mio_throttle",
        "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
        "smsp__pcsamp_warps_issue_stalled_no_instructions",
        "smsp__pcsamp_warps_issue_stalled_not_selected",
        "smsp__pcsamp_warps_issue_stalled_selected",
        "smsp__pcsamp_warps_issue_stalled_sleeping",
        "smsp__pcsamp_warps_issue_stalled_membar",
        "smsp__pcsamp_warps_issue_stalled_drain",
    ],
}

# Human-readable labels for metrics (matching NCU-UI column headers)
METRIC_LABELS = {
    "inst_executed": "Inst Executed",
    "thread_inst_executed": "Thread Inst",
    "thread_inst_executed_true": "Thread Inst (pred-on)",
    "derived__avg_thread_executed": "Avg Threads",
    "derived__avg_thread_executed_true": "Avg Threads (pred-on)",
    "memory_type": "Addr Space",
    "memory_access_type": "Access Op",
    "memory_access_size_type": "Access Size [bit]",
    "memory_l1_wavefronts_shared": "L1 Wavefronts Shared",
    "memory_l1_wavefronts_shared_ideal": "L1 Wavefronts Ideal",
    "derived__memory_l1_wavefronts_shared_excessive": "L1 Wavefronts Excessive",
    "derived__memory_l1_conflicts_shared_nway": "L1 Conflicts N-way",
    "memory_l1_tag_requests_global": "L1 Tag Requests Global",
    "memory_l2_theoretical_sectors_global": "L2 Sectors Global",
    "memory_l2_theoretical_sectors_global_ideal": "L2 Sectors Ideal",
    "derived__memory_l2_theoretical_sectors_global_excessive": "L2 Sectors Excessive",
    "smsp__pcsamp_sample_count": "Samples",
    "smsp__pcsamp_warps_issue_stalled_barrier": "Stall: Barrier",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard": "Stall: Long SB",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard": "Stall: Short SB",
    "smsp__pcsamp_warps_issue_stalled_wait": "Stall: Wait",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle": "Stall: LG Throttle",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle": "Stall: TEX Throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle": "Stall: MIO Throttle",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle": "Stall: Math Throttle",
    "smsp__pcsamp_warps_issue_stalled_no_instructions": "Stall: No Inst",
    "smsp__pcsamp_warps_issue_stalled_not_selected": "Stall: Not Selected",
    "smsp__pcsamp_warps_issue_stalled_selected": "Stall: Selected",
    "smsp__pcsamp_warps_issue_stalled_sleeping": "Stall: Sleeping",
    "smsp__pcsamp_warps_issue_stalled_membar": "Stall: Membar",
    "smsp__pcsamp_warps_issue_stalled_drain": "Stall: Drain",
}

# Enum-valued metrics that need string conversion
MEMORY_TYPE_MAP = {0: "", 2: "", 4: "Global", 8: "Local", 16: "Shared",
                   32: "Texture", 64: "Tmem", 128: "TmemShared"}
MEMORY_ACCESS_TYPE_MAP = {0: "", 2: "Load", 4: "Store", 8: "Atomic", 16: "Shift"}

# Metrics whose per-PC values are categorical strings, not numeric sums
STRING_METRICS = {"memory_type", "memory_access_type", "memory_access_size_type"}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_val(v):
    """Return None for NaN/Inf."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _convert_metric_value(metric_name, value):
    """Convert enum-typed metric values to human-readable strings."""
    if metric_name == "memory_type":
        return MEMORY_TYPE_MAP.get(value, value)
    if metric_name == "memory_access_type":
        return MEMORY_ACCESS_TYPE_MAP.get(value, value)
    return value


def _aggregate(base, increment):
    """Aggregate a metric value: sum numerics, collect strings."""
    if base is None:
        return increment
    if isinstance(base, str):
        return base if increment in base else f"{base}, {increment}"
    return base + increment


def metric_label(name):
    """Human-readable column header for a metric."""
    return METRIC_LABELS.get(name, name)


def discover_instanced_metrics(action, pattern=None):
    """Find all metrics that have per-PC instance data with correlation IDs.

    Returns:
        list of dicts: [{"name", "num_instances", "label", "unit"}, ...]
    """
    results = []
    for name in action.metric_names():
        try:
            m = action.metric_by_name(name)
            if m is None:
                continue
            n = m.num_instances()
            if n > 0 and m.has_correlation_ids():
                if pattern is None or pattern in name:
                    results.append({
                        "name": name,
                        "num_instances": n,
                        "label": m.label() or name,
                        "unit": m.unit() or "",
                    })
        except Exception:
            continue
    return results


def _extract_per_pc(action, metric_name):
    """Extract {pc: value} for an instanced metric. Handles group metrics."""
    metric = action.metric_by_name(metric_name)
    if metric is None:
        return {}

    # Group metrics expand to comma-separated sub-metric names
    if "group" in metric_name:
        try:
            sub_names = metric.value().split(",")
            sub_metrics = [action[n] for n in sub_names]
        except Exception:
            return {}
    else:
        sub_metrics = [metric]

    pc_to_value = {}
    for m in sub_metrics:
        if m.num_instances() == 0 or not m.has_correlation_ids():
            continue
        corr_ids = m.correlation_ids()
        is_string = metric_name in STRING_METRICS
        for i in range(m.num_instances()):
            try:
                pc = corr_ids.value(i)
                raw_val = _safe_val(m.value(i))
                if pc is None or raw_val is None:
                    continue
                val = _convert_metric_value(metric_name, raw_val)
                if is_string:
                    # For categorical metrics, keep first non-empty value per PC
                    if val and pc not in pc_to_value:
                        pc_to_value[pc] = val
                else:
                    pc_to_value[pc] = pc_to_value.get(pc, 0) + val
            except Exception:
                continue
    return pc_to_value


def _fmt_val(val):
    """Format a single metric value for tabulate cells."""
    if val is None or val == "" or val == 0 or val == 0.0:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        return f"{val:,.2f}" if abs(val) < 100 else f"{val:,.0f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: source_analysis
# ═══════════════════════════════════════════════════════════════════════════════

def source_analysis(
    action,
    metric_names=None,
    asm="sass",
    line_filter=None,
    asm_only=False,
):
    """Build a full source ↔ ASM table with per-line and per-PC metrics.

    Produces the same interleaved view as NCU-UI's Source tab: each source line
    is followed by its correlated SASS or PTX instructions, with all requested
    metrics shown on both levels (aggregated on source lines, raw on ASM lines).

    Args:
        action: ncu_report.IAction from a loaded report.
        metric_names: List of instanced metric names to include. If None, uses
            all available instanced metrics with PC correlation IDs.
        asm: "sass", "ptx", or "both". Which assembly representation to show.
            Defaults to "sass".
        line_filter: Optional set of source line numbers to restrict output to.
        asm_only: If True, emit only ASM rows (no source rows).

    Returns:
        list[dict]: Rows in source order. Each dict has keys:
            "type"      — "source" or "asm"
            "file"      — source file basename
            "line"      — source line number (0 if unknown)
            "source"    — source code text (source rows) or "" (asm rows)
            "sass"      — SASS instruction text (asm rows) or "" (source rows)
            "ptx"       — PTX instruction text (asm rows) or "" (source rows)
            + one key per metric name → metric value (numeric or string)

        The returned list can be sorted/filtered with standard Python, e.g.:
            sorted(rows, key=lambda r: r.get("inst_executed", 0), reverse=True)
    """
    # ── Resolve metric list ──
    if metric_names is None:
        discovered = discover_instanced_metrics(action)
        metric_names = [m["name"] for m in discovered]

    # ── Extract per-PC data for each metric ──
    metric_data = OrderedDict()  # metric_name → {pc: value}
    all_pcs = set()
    for mname in metric_names:
        per_pc = _extract_per_pc(action, mname)
        if per_pc:
            metric_data[mname] = per_pc
            all_pcs.update(per_pc.keys())

    # Prune metric_names to those that actually had data
    metric_names = list(metric_data.keys())

    if not all_pcs:
        return []

    # ── Map each PC → source info + SASS + PTX ──
    pc_info = {}
    for pc in all_pcs:
        src = action.source_info(pc)
        pc_info[pc] = {
            "file": src.file_name() if src else "",
            "line": src.line() if src else 0,
            "sass": action.sass_by_pc(pc) or "",
            "ptx": action.ptx_by_pc(pc) or "",
        }

    # ── Get source file contents ──
    try:
        source_files = dict(action.source_files())
    except Exception:
        source_files = {}

    # ── Group PCs by (file, line), preserving PC order ──
    line_pcs = defaultdict(list)
    for pc in sorted(all_pcs):
        info = pc_info[pc]
        key = (info["file"], info["line"])
        line_pcs[key].append(pc)

    # ── Aggregate metrics to source-line level ──
    line_metrics = defaultdict(dict)  # (file, line) → {metric: agg_value}
    for pc in all_pcs:
        key = (pc_info[pc]["file"], pc_info[pc]["line"])
        for mname, per_pc in metric_data.items():
            if pc in per_pc:
                val = per_pc[pc]
                if mname in STRING_METRICS:
                    # Collect unique string values
                    existing = line_metrics[key].get(mname, "")
                    s = str(val)
                    if not existing:
                        line_metrics[key][mname] = s
                    elif s not in existing:
                        line_metrics[key][mname] = f"{existing}, {s}"
                else:
                    line_metrics[key][mname] = _aggregate(
                        line_metrics[key].get(mname), val
                    )

    # ── Build rows in source order ──
    rows = []
    files = sorted({f for (f, _) in line_pcs if f})

    for src_file in files:
        file_short = os.path.basename(src_file) if src_file else ""

        # Get source text
        try:
            content = source_files[src_file] if src_file in source_files else ""
        except (KeyError, TypeError):
            content = ""
        src_lines = content.split("\n") if content else []

        # Lines with data in this file
        data_lines = sorted({l for (f, l) in line_pcs if f == src_file and l > 0})
        if line_filter:
            data_lines = [l for l in data_lines if l in line_filter]
        if not data_lines:
            continue

        for ln in data_lines:
            key = (src_file, ln)
            src_text = ""
            if src_lines and 0 < ln <= len(src_lines):
                src_text = src_lines[ln - 1].rstrip()

            # Source row (aggregated metrics)
            if not asm_only:
                src_row = {
                    "type": "source",
                    "file": file_short,
                    "line": ln,
                    "source": src_text,
                    "sass": "",
                    "ptx": "",
                }
                for mname in metric_names:
                    src_row[mname] = line_metrics[key].get(mname)
                rows.append(src_row)

            # ASM rows (per-PC metrics)
            for pc in line_pcs.get(key, []):
                info = pc_info[pc]
                asm_row = {
                    "type": "asm",
                    "file": file_short,
                    "line": ln,
                    "source": "",
                    "sass": info["sass"],
                    "ptx": info["ptx"],
                }
                for mname in metric_names:
                    asm_row[mname] = metric_data[mname].get(pc)
                rows.append(asm_row)

    return rows


def format_source_analysis(
    rows,
    metric_names,
    asm="sass",
    max_source_width=60,
    max_asm_width=50,
    tablefmt="simple",
):
    """Format source_analysis() output as a tabulate string.

    Args:
        rows: Output from source_analysis().
        metric_names: Metric names to display as columns.
        asm: "sass", "ptx", or "both".
        max_source_width: Max width for source code column.
        max_asm_width: Max width for ASM column.
        tablefmt: Tabulate format (e.g. "simple", "grid", "pretty", "pipe").

    Returns:
        str: Formatted table.
    """
    if not rows:
        return "(no data)"

    # Build headers
    headers = ["Line", "Source"]
    if asm in ("sass", "both"):
        headers.append("SASS")
    if asm in ("ptx", "both"):
        headers.append("PTX")
    for mname in metric_names:
        headers.append(metric_label(mname))

    # Build table data
    table_data = []
    for row in rows:
        is_source = row["type"] == "source"

        cells = []

        # Line number: show for source rows, blank for asm
        cells.append(row["line"] if is_source else "")

        # Source text
        src = row["source"]
        if is_source:
            cells.append(src[:max_source_width] if len(src) > max_source_width else src)
        else:
            cells.append("")

        # ASM columns
        if asm in ("sass", "both"):
            s = row.get("sass", "")
            cells.append(s[:max_asm_width] if len(s) > max_asm_width else s)
        if asm in ("ptx", "both"):
            p = row.get("ptx", "")
            cells.append(p[:max_asm_width] if len(p) > max_asm_width else p)

        # Metric columns
        for mname in metric_names:
            cells.append(_fmt_val(row.get(mname)))

        table_data.append(cells)

        # Separator after the last asm row of a source line group
        # (detected when the next row is a source row or end of list)

    # Determine alignment: right-align metric columns
    n_fixed = len(headers) - len(metric_names)
    colalign = ["right", "left"]
    if asm in ("sass", "both"):
        colalign.append("left")
    if asm in ("ptx", "both"):
        colalign.append("left")
    colalign.extend(["right"] * len(metric_names))

    return tabulate(table_data, headers=headers, tablefmt=tablefmt,
                    colalign=colalign, missingval="")


# ═══════════════════════════════════════════════════════════════════════════════
# Core: source_hotspots
# ═══════════════════════════════════════════════════════════════════════════════

def source_hotspots(
    action,
    metric_name="inst_executed",
    asm="ptx",
    top_n=20,
    metric_names=None,
    tablefmt="simple",
):
    """Find hotspot instructions ranked by a chosen metric.

    Args:
        action: ncu_report.IAction from a loaded report.
        metric_name: Metric to sort/rank by. Can be any instanced metric
            or a group metric (e.g. "group:smsp__pcsamp_warp_stall_reasons").
        asm: "ptx", "sass", or "both". Assembly to display. Default "ptx".
        top_n: Number of top rows to return.
        metric_names: Additional metrics to show alongside the sort metric.
            If None, shows only the sort metric.
        tablefmt: Tabulate format string.

    Returns:
        list[dict]: Top rows sorted by metric_name descending. Each dict has:
            "file", "line", "source", "sass", "ptx", + metric columns.
    """
    # Build the full metric list (sort metric first, then extras)
    all_metrics = [metric_name]
    if metric_names:
        all_metrics.extend(m for m in metric_names if m != metric_name)

    rows = source_analysis(action, metric_names=all_metrics, asm=asm, asm_only=True)
    if not rows:
        return []

    # Sort by the ranking metric, descending
    rows.sort(key=lambda r: r.get(metric_name) or 0, reverse=True)
    top_rows = rows[:top_n]

    return top_rows


def format_hotspots(rows, metric_name, metric_names=None, asm="ptx",
                    max_asm_width=50, tablefmt="simple"):
    """Format source_hotspots() output as a tabulate string.

    Args:
        rows: Output from source_hotspots().
        metric_name: The primary sort metric.
        metric_names: Additional metrics shown.
        asm: "ptx", "sass", or "both".
        max_asm_width: Max width for ASM column.
        tablefmt: Tabulate format.

    Returns:
        str: Formatted table.
    """
    if not rows:
        return "(no hotspot data)"

    all_metrics = [metric_name]
    if metric_names:
        all_metrics.extend(m for m in metric_names if m != metric_name)

    headers = ["File", "Line"]
    if asm in ("sass", "both"):
        headers.append("SASS")
    if asm in ("ptx", "both"):
        headers.append("PTX")
    for mname in all_metrics:
        headers.append(metric_label(mname))

    table_data = []
    for row in rows:
        cells = [row.get("file", ""), row.get("line", "")]
        if asm in ("sass", "both"):
            s = row.get("sass", "")
            cells.append(s[:max_asm_width] if len(s) > max_asm_width else s)
        if asm in ("ptx", "both"):
            p = row.get("ptx", "")
            cells.append(p[:max_asm_width] if len(p) > max_asm_width else p)
        for mname in all_metrics:
            cells.append(_fmt_val(row.get(mname)))
        table_data.append(cells)

    n_fixed = len(headers) - len(all_metrics)
    colalign = ["left", "right"]
    if asm in ("sass", "both"):
        colalign.append("left")
    if asm in ("ptx", "both"):
        colalign.append("left")
    colalign.extend(["right"] * len(all_metrics))

    return tabulate(table_data, headers=headers, tablefmt=tablefmt,
                    colalign=colalign, missingval="")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_report_summary(action):
    """Print kernel summary."""
    name = action.name(ncu_report.IAction.NameBase_FUNCTION)

    def get(n, default="N/A"):
        m = action.metric_by_name(n)
        if m and m.has_value():
            v = _safe_val(m.value())
            return v if v is not None else default
        return default

    dur = get("gpu__time_duration.sum")
    dur_str = f"{dur/1000:.2f} us" if isinstance(dur, (int, float)) else dur
    pct = lambda v: f"{v:.1f}%" if isinstance(v, (int, float)) else v

    print(f"Kernel:    {name}")
    print(f"Duration:  {dur_str}")
    print(f"Compute:   {pct(get('sm__throughput.avg.pct_of_peak_sustained_elapsed'))}  "
          f"Memory: {pct(get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'))}  "
          f"Occupancy: {pct(get('sm__warps_active.avg.pct_of_peak_sustained_active'))}")
    print(f"Regs:      {get('launch__registers_per_thread')}   "
          f"Block: {get('launch__block_size')}   Grid: {get('launch__grid_size')}")


def print_catalog(instanced_metrics):
    """Print catalog of all per-PC instanced metrics."""
    headers = ["Metric Name", "Label", "Unit", "Instances"]
    table_data = []
    for m in sorted(instanced_metrics, key=lambda x: x["name"]):
        table_data.append([m["name"], metric_label(m["name"]), m["unit"],
                           m["num_instances"]])
    print(tabulate(table_data, headers=headers, tablefmt="simple"))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NCU source-correlated metric viewer (programmatic NCU-UI Source tab)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available presets: {', '.join(PRESETS.keys())}",
    )
    parser.add_argument("report", help="Path to .ncu-rep file")
    parser.add_argument("--action", type=int, default=0, help="Action (kernel) index")
    parser.add_argument("--asm", choices=["sass", "ptx", "both"], default="sass",
                        help="Assembly type (default: sass)")
    parser.add_argument("--preset", type=str, default="default",
                        choices=list(PRESETS.keys()) + ["all"],
                        help="Metric preset (default: %(default)s)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated metric names (overrides --preset)")
    parser.add_argument("--catalog", action="store_true",
                        help="Print catalog of all instanced per-PC metrics")
    parser.add_argument("--lines", type=str, default=None,
                        help="Comma-separated source line numbers to show")
    parser.add_argument("--asm-only", action="store_true",
                        help="Skip source rows, show only ASM rows")
    parser.add_argument("--hotspots", action="store_true",
                        help="Show hotspot ranking instead of full source view")
    parser.add_argument("--sort-by", type=str, default="inst_executed",
                        help="Metric to sort hotspots by (default: inst_executed)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of hotspot rows (default: 20)")
    parser.add_argument("--fmt", type=str, default="simple",
                        help="Tabulate format (simple, grid, pretty, pipe, etc.)")

    args = parser.parse_args()

    print(f"Loading: {args.report}")
    ctx = ncu_report.load_report(args.report)

    num_ranges = ctx.num_ranges()
    total_actions = sum(ctx.range_by_idx(ri).num_actions() for ri in range(num_ranges))
    print(f"Ranges: {num_ranges}, Actions: {total_actions}")

    action = ctx.range_by_idx(0).action_by_idx(args.action)
    print()
    print_report_summary(action)

    # Discover available instanced metrics
    instanced = discover_instanced_metrics(action)
    available = {m["name"] for m in instanced}
    print(f"\nPer-PC instanced metrics: {len(instanced)}")

    if args.catalog:
        print()
        print_catalog(instanced)
        return

    # Resolve metric list
    if args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(",")]
    elif args.preset == "all":
        metric_names = sorted(available)
    else:
        metric_names = [m for m in PRESETS[args.preset] if m in available]

    # Warn and filter missing
    for m in metric_names:
        if m not in available:
            print(f"  Warning: '{m}' not available as instanced metric")
    metric_names = [m for m in metric_names if m in available]

    if not metric_names:
        print("No valid metrics selected. Use --catalog to see available metrics.")
        return

    print(f"\nMetrics ({len(metric_names)}):")
    for m in metric_names:
        print(f"  {metric_label(m):<30} {m}")

    line_filter = None
    if args.lines:
        line_filter = set(int(x) for x in args.lines.split(","))

    # ── Hotspot mode ──
    if args.hotspots:
        sort_metric = args.sort_by
        if sort_metric not in available:
            print(f"Sort metric '{sort_metric}' not available. Using inst_executed.")
            sort_metric = "inst_executed"

        extra = [m for m in metric_names if m != sort_metric]
        rows = source_hotspots(
            action, metric_name=sort_metric, asm=args.asm,
            top_n=args.top, metric_names=extra, tablefmt=args.fmt,
        )
        print(f"\nTop {len(rows)} hotspots by '{metric_label(sort_metric)}':\n")
        print(format_hotspots(
            rows, sort_metric, metric_names=extra,
            asm=args.asm, tablefmt=args.fmt,
        ))
        return

    # ── Full source analysis mode ──
    rows = source_analysis(
        action, metric_names=metric_names, asm=args.asm,
        line_filter=line_filter, asm_only=args.asm_only,
    )

    if not rows:
        print("\nNo source-correlated data found.")
        return

    # Group output by file
    files = []
    seen = set()
    for r in rows:
        f = r["file"]
        if f and f not in seen:
            files.append(f)
            seen.add(f)

    for src_file in files:
        file_rows = [r for r in rows if r["file"] == src_file]
        print(f"\n{'=' * 30} {src_file} {'=' * 30}\n")
        print(format_source_analysis(
            file_rows, metric_names, asm=args.asm, tablefmt=args.fmt,
        ))

    # Summary: top source lines
    source_rows = [r for r in rows if r["type"] == "source"]
    if source_rows and "inst_executed" in metric_names:
        source_rows.sort(key=lambda r: r.get("inst_executed") or 0, reverse=True)
        total = sum(r.get("inst_executed") or 0 for r in source_rows)

        print(f"\n{'=' * 30} Top Source Lines by Instructions {'=' * 30}\n")
        summary_headers = ["Line", "Source", "Inst Executed", "% Total"]
        summary_data = []
        for r in source_rows[:15]:
            inst = r.get("inst_executed") or 0
            pct = 100.0 * inst / total if total else 0
            src = r.get("source", "")[:50]
            summary_data.append([r["line"], src, _fmt_val(inst), f"{pct:.1f}%"])
        print(tabulate(summary_data, headers=summary_headers, tablefmt=args.fmt,
                        colalign=("right", "left", "right", "right")))


if __name__ == "__main__":
    main()
