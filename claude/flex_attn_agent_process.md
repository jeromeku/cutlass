# Agent Process: FlexAttention Trace

**Date:** 2026-02-13

---

## Overview

The task was to trace the FlexAttention kernel across 3 backends (reference, Triton, CuteDSL) for both forward and backward passes, explaining how `score_mod` and `block_mask` are stitched in. The output was to go in `claude/flex_attention_trace.md`.

---

## Phase 1: Initial Assessment

**Before touching any tool**, I read `FLEX_ATTN_IMPL.md` (already shown in the conversation) to understand the scope:

- 3 backends × 2 directions = 6 call paths
- Key source roots: `thirdparty/pytorch`, `thirdparty/flash-attention/flash_attn/cute`, `thirdparty/attention-gym`
- The document pointed me directly to the key files

My mental model going in:
1. The public API (`torch.nn.attention.flex_attention`) wraps a Higher-Order Operator (HOP)
2. The HOP dispatches differently in eager vs. compiled mode
3. In compiled mode, Inductor's lowering picks the backend
4. The CuteDSL backend routes through `flash_attn.cute.interface`

The critical question I needed to answer before starting: *which files actually matter?* There are thousands of files in the pytorch and flash-attn repos. I did not want to read blindly.

---

## Phase 2: Parallelizing the Exploration

I launched **two background agents simultaneously** rather than reading files sequentially myself. The reason: these were large repos and I needed to survey a lot of ground quickly without burning main-thread context on speculative reads.

### Agent 1: `aa008dd` — Broad exploration (attention-gym + flash-attn + HOP dir listing)

**Prompt given:**
```
Explore the following paths and collect detailed information:
1. Read thirdparty/attention-gym/examples/flex_flash_attention.py (full file)
2. List files under thirdparty/flash-attention/flash_attn/cute/
3. List files under thirdparty/pytorch/torch/_higher_order_ops/ (flex_attention files)
4. List files under thirdparty/pytorch/torch/_inductor/ (flex_attention files — just top level)
5. Search for "flex_attention" in the higher_order_ops dir
6. Read the main flex_attention higher order ops file if found
7. Search for "flex_triton" and "flex_flash" definitions in attention-gym/
```

**Why:** I needed the example file (entry point), the flash-attn cute directory listing (so I knew what files existed), and confirmation that `_higher_order_ops/flex_attention.py` was the right HOP file. The listing was cheap and would tell me what to read next.

### Agent 2: `ad4a9dc` — Inductor-specific exploration

**Prompt given:**
```
In thirdparty/pytorch, find and read all files related to flex_attention. Specifically:
1. Search for files named "*flex*" anywhere under torch/
2. Search for "flex_attention" in torch/_inductor/ — list all files that match
3. Read these files if they exist:
   - torch/_higher_order_ops/flex_attention.py
   - Any file in torch/_inductor/ with "flex" in the name
4. Search for kernel generation code in torch/_inductor/
5. Read torch/nn/attention/flex_attention.py if it exists
```

**Why:** I needed the inductor file tree confirmed. I knew from experience that inductor kernels live in `kernel/flex/` but I wanted the full listing, not to guess. This agent's job was to map the inductor side and return file sizes / line counts so I could prioritize what to read directly.

### Why two agents in parallel?

- They covered **different source trees** with no overlap: agent 1 covered attention-gym + flash-attn + HOP listing; agent 2 covered pytorch/inductor
- Running them simultaneously saved significant wall-clock time (both took ~30–100s)
- Each produced a structured summary I could extract line numbers from

---

## Phase 3: Reading Critical Files Directly (Main Thread)

While the agents ran, I did **targeted direct reads** of files I was confident I needed in full:

### Direct reads (main thread):
1. **`flex_flash_attention.py` (example)** — already had from agent 1's work, but read it directly to confirm the exact lines (79–129) referenced in the task
2. **`flash_attn/cute/interface.py`** — 1,778 lines. I read this in full because it is the deepest layer: `_flash_attn_fwd`, `_flash_attn_bwd`, `FlashAttnFunc`, `cute.compile`. This was the "bottom of the stack" for the CuteDSL backend
3. **`torch/_inductor/kernel/flex/flex_flash_attention.py`** — 606 lines. Read in full — this is the bridge between Inductor IR and the flash-attn library, containing `HierarchicalIndex`, `patch_fixed_layout_indexer_for_cutedsl`, `create_flex_flash_attention_kernel`
4. **`torch/_higher_order_ops/flex_attention.py`** — 1,462 lines. Read in 200-line chunks: lines 1–200 (class defs), 200–500 (dispatch impls), 500–800 (backward)
5. **`torch/_inductor/kernel/flex/flex_attention.py`** — 1,068 lines. Read lines 1–400 (the lowering entry and backend dispatch)
6. **`torch/nn/attention/flex_attention.py`** — 1,743 lines. Read lines 1–650 (public API, BlockMask, `_vmap_for_bhqkv`)

### Why direct reads rather than agent 3?

For these specific files I was confident about *which* files and *roughly which line ranges* mattered. Using an agent for targeted reads of known files adds latency without benefit. The Explore/Bash agents shine for **search** (unknown locations) or **survey** (large directory trees); they are slower for deterministic targeted reads.

---

## Phase 4: Agent 3 — Parallel deep-read while I wrote

I launched a third agent **while I was writing the document** to get the full verbatim text of 4 key files:

**Agent `a1d78af`:**
```
Read the following files in full and return their complete contents with line numbers.
Do NOT summarize — return actual code.

1. torch/_higher_order_ops/flex_attention.py (all 1462 lines)
2. torch/_inductor/kernel/flex/flex_attention.py (all 1068 lines)
3. torch/_inductor/kernel/flex/flex_flash_attention.py (full)
4. torch/nn/attention/flex_attention.py (full)
```

**Why:** I was writing the document and occasionally needed to double-check exact line numbers or function signatures. Rather than interrupting my write loop to read files, I set this agent to work in the background. I didn't end up needing most of its output since I already had the key content, but it served as a verification pass.

---

## Phase 5: Writing the Document

With the source material gathered, I wrote `claude/flex_attention_trace.md` in a single `Write` tool call. The structure I chose:

1. **Architecture overview first** — ASCII diagram of the dispatch stack. This gives the reader orientation before any code.
2. **Frame-by-frame trace** per backend, numbered so the reader can follow sequentially
3. **Shared mechanism section** (§6) for score_mod / block_mask stitching — because all 3 backends go through the same Dynamo tracing path, I didn't want to repeat it three times
4. **Key Functions Index** as a quick-reference table
5. **Code Map** with clickable VSCode/GitHub links

### Decisions made while writing:

- **Reference backward**: I initially wrote "autograd through the forward" which is wrong — `sdpa_dense_backward` is a hand-written backward. Agent `aa008dd`'s detailed output confirmed this and I corrected it in a post-write patch.
- **log₂ domain**: The logsumexp being stored ÷ log(2) is a non-obvious detail that is critical for understanding the backward. I noted this in both the forward and backward frames.
- **`HierarchicalIndex`**: This is the most non-obvious CuteDSL-specific mechanism. I gave it a dedicated subsection (§6.3) because it explains why CuteDSL needs a patched indexer at all.
- **`create_fw_bw_graph`**: I traced this in §6.4 because it explains how `score_mod_bwd` (needed by the CuteDSL backward) is derived automatically from `score_mod` — a key question the task asked about.

---

## Phase 6: Post-write Patches

After the main document was written, agent `aa008dd` completed and returned additional details. I applied two patches:

1. **Reference backward** (`sdpa_dense_backward`) — replaced the vague "autograd through forward" description with the actual algorithm (recompute P, compute delta, apply joint_graph, GQA reduction)
2. **Jinja2 templates** — added the 5 template file paths to the Code Map section, including the CuteDSL-specific ones (`flash_attention.py.jinja`, `flash_attention_backward.py.jinja`)

---

## What Worked Well

- **Parallel agents for survey** — launching two agents simultaneously for directory listing and file discovery was efficient; I got the file map in ~30s while reading the example file myself
- **Direct reads for known critical files** — `interface.py` and `flex_flash_attention.py` are the deepest layers and the most important to read in full; doing this on the main thread let me ask follow-up questions as I read (e.g., "why does `_flash_attn_bwd` have three separate compile caches?")
- **Writing the document while agents ran** — the third agent (`a1d78af`) ran concurrently with my write, acting as insurance against missed line numbers

## What I Would Do Differently

- **Read `common.py`** — `build_subgraph_buffer` in `torch/_inductor/kernel/flex/common.py` is the exact bridge between the FX subgraph and the inlinable Triton IR, and I described it only at a high level. A targeted read would have let me trace that frame more precisely.
- **Read one Jinja2 template** — the actual `{{ modification(score, ...) }}` hook in `flex_attention.py.jinja` is the moment where score_mod IR gets inlined. Reading it would have let me show the exact template substitution point rather than describing it abstractly.
- **Read `flash_fwd.py` (SM90 kernel body)** — the actual CuteDSL kernel spec for `FlashAttentionForwardSm90` was described from general knowledge of the FlashAttention-2 algorithm rather than from the actual source. The real CuteDSL code uses TMA pipelines, warp specialization, and WGMMA instructions that are architecture-specific.

---

## Agent Delegation Heuristics Used

| Situation | Tool used | Reason |
|---|---|---|
| Unknown directory structure | Explore agent | Agent has Glob + Grep; can survey multi-level trees |
| Known file, known line range | Read directly | No agent overhead needed |
| Large file (>1000 lines), needed fully | Explore agent | Protect main context from large verbatim dumps |
| Writing in progress, verification needed | Explore agent (background) | Don't block write loop |
| Exact function signature needed NOW | Read directly with offset | Faster than waiting for agent |
| Two independent source trees | Two agents in parallel | Parallel execution saves wall-clock time |
