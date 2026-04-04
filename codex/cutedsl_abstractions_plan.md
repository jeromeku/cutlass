# Plan: CuTe DSL Pipeline Abstractions

## Goal

Map the CuTe DSL SM90 pipeline abstractions in:

- `python/CuTeDSL/cutlass/pipeline/sm90.py`

to their closest CUTLASS/CuTe C++ equivalents, with emphasis on:

- `PipelineProducer`
- `PipelineConsumer`
- `ImmutableResourceHandle`

and document the result in:

- `codex/cutedsl_abstractions.md`

## Scope Notes

- The closest C++ counterparts are in `include/cutlass/pipeline/sm90_pipeline.hpp`.
- `include/cute` provides lower-level layout, swizzle, and cluster helpers used by the C++ pipeline code, but the pipeline classes themselves live under `include/cutlass/pipeline`.
- Early inspection suggests `PipelineProducer` and `PipelineConsumer` are DSL-only ergonomic wrappers over a pipeline object plus a mutable `PipelineState`, not direct 1:1 C++ classes.

## Chunks

1. Base abstraction mapping
   - Map `PipelineState`, `PipelineUserType`, `CooperativeGroup`, `PipelineOp`, `SyncObject`, `MbarrierArray`, and `TmaStoreFence`.
   - Output: a reference table for the shared building blocks.

2. Specialized pipeline mapping
   - Map `PipelineAsync`, `PipelineCpAsync`, `PipelineTmaAsync`, `PipelineTmaStore`, and `PipelineOrder`.
   - Output: class-by-class correspondence notes and important mismatches.

3. Participant-wrapper mapping
   - Explain what `PipelineProducer`, `PipelineConsumer`, and `ImmutableResourceHandle` add on top of the lower-level APIs.
   - Identify the closest C++ equivalents: explicit `PipelineState`, `ProducerToken` / `ConsumerToken`, `producer_get_barrier`, and the manual acquire/commit/release workflow.

4. Final document pass
   - Write `codex/cutedsl_abstractions.md` with code links, a mapping table, and a short execution-path explanation.
   - Verify claims against the repo sources.
