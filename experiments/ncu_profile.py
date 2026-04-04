"""
ncu_profile — Lightweight NCU profiling module for AI-assisted optimization.

Wraps ncu_source_view.py with dual output modes:
    format="human"  — Rich tabulate tables for terminal display
    format="llm"    — Compact markdown for LLM context windows

Usage (programmatic):
    from ncu_profile import load, summary, source_view, hotspots, catalog

    action = load("kernel.ncu-rep")
    print(summary(action))
    print(hotspots(action, sort_by="inst_executed", preset="stalls"))
    print(source_view(action, preset="memory", lines={77, 120}))

Usage (CLI):
    python ncu_profile.py kernel.ncu-rep                       # summary
    python ncu_profile.py kernel.ncu-rep --hotspots            # top instructions
    python ncu_profile.py kernel.ncu-rep --source              # full source view
    python ncu_profile.py kernel.ncu-rep --catalog             # list metrics
    python ncu_profile.py kernel.ncu-rep --fmt llm             # LLM-friendly output
"""

import sys
import os
import math
import argparse

# ─── NCU Python path discovery ───
NCU_PYTHON_PATHS = [
    "/home/jeromeku/nsight-compute-2026-01/extras/python",
    "/usr/local/cuda/extras/CUPTI/ncu_python",
    "/opt/nvidia/nsight-compute/ncu_python",
]

for _p in NCU_PYTHON_PATHS:
    if os.path.exists(os.path.join(_p, "ncu_report.py")):
        sys.path.insert(0, _p)
        break

# Add experiments/ to path for ncu_source_view imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import ncu_report
import ncu_source_view as sv


# ═══════════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════════

def load(report_path, action_idx=0):
    """Load an .ncu-rep file and return an IAction.

    Args:
        report_path: Path to .ncu-rep file.
        action_idx: Kernel index within the first range (default 0).

    Returns:
        ncu_report.IAction
    """
    ctx = ncu_report.load_report(report_path)
    return ctx.range_by_idx(0).action_by_idx(action_idx)


def summary(action, format="human"):
    """Kernel summary: name, duration, throughput, launch params.

    Args:
        action: IAction from load().
        format: "human" for rich display, "llm" for compact markdown.

    Returns:
        str
    """
    name = action.name(ncu_report.IAction.NameBase_FUNCTION)

    def get(n, default="N/A"):
        m = action.metric_by_name(n)
        if m and m.has_value():
            v = m.value()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return default
            return v
        return default

    dur = get("gpu__time_duration.sum")
    dur_str = f"{dur/1000:.2f} us" if isinstance(dur, (int, float)) else dur
    compute = get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    memory = get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    occupancy = get("sm__warps_active.avg.pct_of_peak_sustained_active")
    regs = get("launch__registers_per_thread")
    block = get("launch__block_size")
    grid = get("launch__grid_size")

    pct = lambda v: f"{v:.1f}%" if isinstance(v, (int, float)) else str(v)

    if format == "llm":
        lines = [
            f"**Kernel**: `{name}`",
            f"**Duration**: {dur_str}",
            f"**Compute**: {pct(compute)} | **Memory**: {pct(memory)} | **Occupancy**: {pct(occupancy)}",
            f"**Regs**: {regs} | **Block**: {block} | **Grid**: {grid}",
        ]
        return "\n".join(lines)

    lines = [
        f"Kernel:    {name}",
        f"Duration:  {dur_str}",
        f"Compute:   {pct(compute)}  Memory: {pct(memory)}  Occupancy: {pct(occupancy)}",
        f"Regs:      {regs}   Block: {block}   Grid: {grid}",
    ]
    return "\n".join(lines)


def source_view(action, preset="default", metrics=None, asm="sass",
                lines=None, asm_only=False, format="human", tablefmt=None):
    """Full source-correlated view matching NCU-UI's Source tab.

    Args:
        action: IAction from load().
        preset: "default", "memory", "stalls", "instructions", or "all".
        metrics: Explicit metric list (overrides preset).
        asm: "sass", "ptx", or "both".
        lines: Optional set of source line numbers to filter to.
        asm_only: If True, skip source rows.
        format: "human" or "llm".
        tablefmt: Override tabulate format (default: "simple" for human, "pipe" for llm).

    Returns:
        str: Formatted table.
    """
    metric_names = _resolve_metrics(action, preset, metrics)
    if not metric_names:
        return "(no valid metrics)"

    rows = sv.source_analysis(
        action, metric_names=metric_names, asm=asm,
        line_filter=lines, asm_only=asm_only,
    )
    if not rows:
        return "(no source-correlated data)"

    if format == "llm":
        return _format_llm_source(rows, metric_names, asm)

    fmt = tablefmt or "simple"
    # Group by file
    parts = []
    files = _ordered_unique(r["file"] for r in rows if r["file"])
    for f in files:
        file_rows = [r for r in rows if r["file"] == f]
        parts.append(f"{'=' * 20} {f} {'=' * 20}")
        parts.append(sv.format_source_analysis(file_rows, metric_names, asm=asm, tablefmt=fmt))
    return "\n\n".join(parts)


def hotspots(action, sort_by="inst_executed", preset="default", metrics=None,
             asm="sass", top_n=20, format="human", tablefmt=None):
    """Top instructions ranked by a metric.

    Args:
        action: IAction from load().
        sort_by: Metric to rank by.
        preset: Additional metrics to show alongside sort_by.
        metrics: Explicit additional metric list (overrides preset).
        asm: "sass", "ptx", or "both".
        top_n: Number of rows.
        format: "human" or "llm".
        tablefmt: Override tabulate format.

    Returns:
        str: Formatted table.
    """
    extra_metrics = _resolve_metrics(action, preset, metrics)
    extra = [m for m in extra_metrics if m != sort_by]

    rows = sv.source_hotspots(
        action, metric_name=sort_by, asm=asm,
        top_n=top_n, metric_names=extra,
    )
    if not rows:
        return "(no hotspot data)"

    if format == "llm":
        return _format_llm_hotspots(rows, sort_by, extra, asm)

    fmt = tablefmt or "simple"
    return sv.format_hotspots(rows, sort_by, metric_names=extra, asm=asm, tablefmt=fmt)


def catalog(action, pattern=None, format="human"):
    """List all per-PC instanced metrics available in this report.

    Args:
        action: IAction from load().
        pattern: Optional substring filter.
        format: "human" or "llm".

    Returns:
        str: Formatted metric catalog.
    """
    instanced = sv.discover_instanced_metrics(action, pattern=pattern)
    if not instanced:
        return "(no instanced metrics found)"

    if format == "llm":
        lines = [f"**{len(instanced)} instanced metrics**" + (f" matching `{pattern}`" if pattern else "")]
        for m in sorted(instanced, key=lambda x: x["name"]):
            lines.append(f"- `{m['name']}` ({m['num_instances']} instances)")
        return "\n".join(lines)

    from tabulate import tabulate
    headers = ["Metric Name", "Label", "Unit", "Instances"]
    data = [[m["name"], sv.metric_label(m["name"]), m["unit"], m["num_instances"]]
            for m in sorted(instanced, key=lambda x: x["name"])]
    return tabulate(data, headers=headers, tablefmt="simple")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_metrics(action, preset, explicit_metrics):
    """Resolve metric list from preset or explicit list, filtered to available."""
    instanced = sv.discover_instanced_metrics(action)
    available = {m["name"] for m in instanced}

    if explicit_metrics:
        names = explicit_metrics
    elif preset == "all":
        return sorted(available)
    else:
        names = sv.PRESETS.get(preset, sv.PRESETS["default"])

    return [m for m in names if m in available]


def _ordered_unique(items):
    """Preserve insertion order while deduplicating."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _format_llm_source(rows, metric_names, asm):
    """Compact markdown source view for LLM consumption."""
    parts = []
    current_file = None

    for row in rows:
        f = row["file"]
        if f != current_file:
            if current_file is not None:
                parts.append("")
            parts.append(f"### {f}")
            current_file = f

        if row["type"] == "source":
            # Source line with aggregated metrics
            metric_vals = []
            for mn in metric_names:
                v = row.get(mn)
                if v and v != 0:
                    label = sv.metric_label(mn).split(":")[-1].strip()  # short label
                    metric_vals.append(f"{label}={sv._fmt_val(v)}")
            metric_str = f"  [{', '.join(metric_vals)}]" if metric_vals else ""
            parts.append(f"L{row['line']:>4}: {row['source']}{metric_str}")
        else:
            # ASM line with per-PC metrics
            asm_text = ""
            if asm in ("sass", "both") and row.get("sass"):
                asm_text = row["sass"]
            elif asm in ("ptx", "both") and row.get("ptx"):
                asm_text = row["ptx"]
            if not asm_text:
                asm_text = row.get("sass") or row.get("ptx") or ""

            metric_vals = []
            for mn in metric_names:
                v = row.get(mn)
                if v and v != 0:
                    label = sv.metric_label(mn).split(":")[-1].strip()
                    metric_vals.append(f"{label}={sv._fmt_val(v)}")
            metric_str = f"  [{', '.join(metric_vals)}]" if metric_vals else ""
            if asm_text:
                parts.append(f"       {asm_text[:60]}{metric_str}")

    return "\n".join(parts)


def _format_llm_hotspots(rows, sort_by, extra_metrics, asm):
    """Compact markdown hotspot list for LLM consumption."""
    all_metrics = [sort_by] + extra_metrics
    parts = [f"**Top {len(rows)} by {sv.metric_label(sort_by)}**\n"]

    for i, row in enumerate(rows, 1):
        asm_text = ""
        if asm in ("sass", "both") and row.get("sass"):
            asm_text = row["sass"][:50]
        elif asm in ("ptx", "both") and row.get("ptx"):
            asm_text = row["ptx"][:50]
        if not asm_text:
            asm_text = row.get("sass", "")[:50] or row.get("ptx", "")[:50]

        metric_vals = []
        for mn in all_metrics:
            v = row.get(mn)
            if v and v != 0:
                label = sv.metric_label(mn).split(":")[-1].strip()
                metric_vals.append(f"{label}={sv._fmt_val(v)}")
        metric_str = ", ".join(metric_vals)

        loc = f"{row.get('file', '')}:{row.get('line', '')}"
        parts.append(f"{i:>2}. `{asm_text}` @ {loc}  [{metric_str}]")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NCU profiling module")
    parser.add_argument("report", help="Path to .ncu-rep file")
    parser.add_argument("--action", type=int, default=0, help="Kernel index")
    parser.add_argument("--fmt", choices=["human", "llm"], default="human",
                        help="Output format")
    parser.add_argument("--asm", choices=["sass", "ptx", "both"], default="sass")
    parser.add_argument("--preset", default="default",
                        choices=list(sv.PRESETS.keys()) + ["all"])
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated metric names")
    parser.add_argument("--source", action="store_true", help="Full source view")
    parser.add_argument("--hotspots", action="store_true", help="Hotspot ranking")
    parser.add_argument("--sort-by", default="inst_executed")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--catalog", action="store_true", help="List metrics")
    parser.add_argument("--lines", type=str, default=None,
                        help="Comma-separated source line numbers")

    args = parser.parse_args()

    action = load(args.report, args.action)
    metric_list = [m.strip() for m in args.metrics.split(",")] if args.metrics else None
    line_set = set(int(x) for x in args.lines.split(",")) if args.lines else None

    # Always print summary
    print(summary(action, format=args.fmt))
    print()

    if args.catalog:
        print(catalog(action, format=args.fmt))
        return

    if args.hotspots:
        print(hotspots(
            action, sort_by=args.sort_by, preset=args.preset,
            metrics=metric_list, asm=args.asm, top_n=args.top,
            format=args.fmt,
        ))
        return

    if args.source:
        print(source_view(
            action, preset=args.preset, metrics=metric_list,
            asm=args.asm, lines=line_set, format=args.fmt,
        ))
        return

    # Default: summary only (already printed)


if __name__ == "__main__":
    main()
