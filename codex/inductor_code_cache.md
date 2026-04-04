# Inductor CuTeDSL Cache Artifact Provenance

This note documents the provenance of artifacts under:

- `/home/jeromeku/cutlass/thirdparty/attention-gym/inductor_logs/torchinductor_cache/tmpn8xhoyoi`

Observed layout:

```text
tmpn8xhoyoi/
  ez/
    cezxxjqogse5logg5ymgku2xsdby3cibvwmd2rzaej7x43p6h47u.py
    cezxxjqogse5logg5ymgku2xsdby3cibvwmd2rzaej7x43p6h47u.debug/
    cezxxjqogse5logg5ymgku2xsdby3cibvwmd2rzaej7x43p6h47u.debug.lock
  tf/
    ctfqfcnbqd7k2sw2gaovzrn3ippbtvqptaaenix5glsuai2guznl.py
  il/
    cilur3yntbcvgmz4nuwbqmlaq2kujbmv57e4tlumkbei2utkkk3c.py
```

## 1. Why `tmpn8xhoyoi` exists at all

`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` triggers a fresh temporary cache root for compile:

- `with_fresh_cache_if_config()` wraps compilation when `force_disable_caches` is enabled:
  - `thirdparty/pytorch/torch/_inductor/compile_fx.py:753`
- `fresh_cache()` creates `tempfile.mkdtemp(dir=cache_dir())`, producing `tmp*`:
  - `thirdparty/pytorch/torch/_inductor/utils.py:1351`
  - `thirdparty/pytorch/torch/_inductor/utils.py:1366`

So `tmpn8xhoyoi` is an ephemeral per-compile cache shard, not a semantic phase name.

## 2. Why `ez`, `tf`, `il` subdirs exist

They are hash-prefix buckets from codecache pathing:

- `get_path()` uses `basename[1:3]` as subdir:
  - `thirdparty/pytorch/torch/_inductor/codecache.py:342`
  - `thirdparty/pytorch/torch/_inductor/codecache.py:351`

File placement is hash-derived, not backend-derived. `ez/tf/il` do not mean fixed roles globally.

## 3. Provenance of each file in this run

### A) `ez/cez...py` (wrapper module)

Producer:

- GraphLowering writes wrapper python via `PyCodeCache.write(wrapper_code.value)`:
  - `thirdparty/pytorch/torch/_inductor/graph.py:2559` (code path around `PyCodeCache.write`)

Role:

- Top-level compiled graph module (`call`, guards, runner, benchmark harness).
- Contains embedded CuTeDSL compile invocation:
  - `async_compile.cutedsl('cutedsl_fused_flex_attention_df816cdf', r'''...''')`
  - in your file: `.../ez/cez...py:53`

### B) `ez/cez....debug/` and `.debug.lock` (compile-debug copy)

Producer:

- Debug copy of compile artifacts (`output_code.py`, fx graph dumps, IR dumps):
  - `thirdparty/pytorch/torch/_inductor/graph.py:2565`
  - `thirdparty/pytorch/torch/_inductor/debug.py:628`
- Lock file from `DebugContext.copy()` file lock:
  - `thirdparty/pytorch/torch/_inductor/debug.py:415`

Role:

- Human-readable compile-debug snapshot.
- Not the runtime kernel source actually imported by `AsyncCompile.cutedsl`.

### C) `tf/ctf...py` (benchmark candidate CuTeDSL module)

Producer:

- `CuteDSLTemplate.generate()` always constructs a `CuteDSLBenchmarkRequest`:
  - `thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_template.py:66`
  - `thirdparty/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_template.py:93`
- `CuteDSLBenchmarkRequest.__init__` materializes python source with `PyCodeCache.write(finalized_code)`:
  - `thirdparty/pytorch/torch/_inductor/autotune_process.py:1013`
  - `thirdparty/pytorch/torch/_inductor/autotune_process.py:1027`

Role:

- Candidate kernel module used by benchmark/selection path.
- In your file, function signature is template-stage name:
  - `cutedsl_flash_attention_cutedsl_1_main`
  - `.../tf/ctf...py:11`

### D) `il/cil...py` (final scheduled runtime CuTeDSL module)

Producer:

- Runtime wrapper executes `async_compile.cutedsl(...)` from `ez/cez...py`.
- `AsyncCompile.cutedsl.task()` writes source via `PyCodeCache.write(source_code)`:
  - `thirdparty/pytorch/torch/_inductor/async_compile.py:571`
  - `thirdparty/pytorch/torch/_inductor/async_compile.py:591`

Role:

- Final callable module loaded and wrapped as `CuteDSLKernelWrapper`.
- In your file, function signature is graph-scheduled fused name:
  - `cutedsl_fused_flex_attention_df816cdf_main`
  - `.../il/cil...py:11`

## 4. Why there are two CuTeDSL kernel files (`tf/...` and `il/...`)

They represent two different stages:

1. Candidate/benchmark stage source (template naming, benchmark request object).
2. Final scheduled runtime source (graph/fused naming, wrapper `async_compile.cutedsl` path).

They can be near-identical in body, but differ in symbol names and hash-derived filenames.

Your diff confirms this: main differences are kernel function name and related hash labels.

## 5. Why multiple files overall

Multiple files are expected because Inductor separately emits:

- compiled graph wrapper (`ez/cez...py`),
- compile-debug snapshot (`ez/...debug/*`),
- benchmark candidate kernel module (`tf/ctf...py`),
- finalized runtime kernel module (`il/cil...py`).

These files correspond to distinct lifecycle steps (codegen, debug copy, benchmarking, final load).
