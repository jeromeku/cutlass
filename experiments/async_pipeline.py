# %%
import torch
from utils.logging import patch_cutlass_env
patch_cutlass_env(keep_ptx=True, dumpdir="async_pp_dump")

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.base_dsl.compiler import CompileCallable, PtxasOptions, GenerateLineInfo, OptLevel
from cutlass.cutlass_dsl import CutlassBaseDSL
from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
# %% [markdown]
# <style>
# div.mermaid > svg {
#   width: 50% !important;
#   height: auto !important;
# }
# </style>
# 
# # Tutorial: Warp Specialization with Async Pipeline in CuTe DSL
# 
# This tutorial explores advanced CUDA programming techniques for implementing efficient producer-consumer 
# patterns using asynchronous communication primitives in the CuTe Domain Specific Language (DSL).
# 
# ## Foundation: Inter-Warp Communication Basics
# 
# ### Understanding CUDA Warps and Shared Memory
# 
# A **warp** is the fundamental execution unit in CUDA, consisting of 32 threads that execute instructions in Single Instruction, 
# Multiple Thread (SIMT) fashion on a Streaming Multiprocessor (SM). Understanding warp-level programming is crucial for 
# achieving optimal GPU performance.
# 
# **Key Concepts:**
# - Warps execute in lockstep, making them ideal for SIMD operations
# - Multiple warps within a thread block (CTA) can cooperate through shared memory
# - Shared memory provides low-latency, high-bandwidth communication between threads
# 
# ### Shared Memory Architecture
# 
# **Shared memory** serves as a programmer-managed cache with several important characteristics:
# 
# - **Speed**: ~100x faster than global memory access
# - **Scope**: Accessible by all threads within the same thread block
# - **Organization**: Divided into banks (typically 32) to enable parallel access
# - **Conflicts**: Bank conflicts occur when multiple threads access the same bank simultaneously
# 
# ### Traditional Synchronous Communication
# 
# The conventional approach for inter-warp communication relies on explicit synchronization barriers. The following sequence diagram 
# illustrates the typical producer-consumer pattern:
# 
# ```mermaid
# sequenceDiagram
#   participant W0 as Producer Warp
#   participant SMEM as Shared Memory
#   participant W1 as Consumer Warp
#   
#   W0->>SMEM: Write data
#   critical Synchronization Barrier
#     W0-->W1: __syncthreads()
#     SMEM->>W1: Read data
#     W0-->W1: __syncthreads()
#   end
# ```
# 
# **Limitations of Synchronous Communication:**
# - All warps must wait at synchronization points
# - No opportunity for overlapped computation
# - Reduced overall throughput due to forced serialization

# %%
@cute.kernel
def synced_producer_consumer(SharedStorage: cutlass.Constexpr, res: cute.Tensor):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)

    staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(1))
    staging_smem.fill(0)
    cute.arch.sync_threads()
    print(f"{res}")
    for i in cutlass.range(cute.size(res)):
        if warp_idx == 0:
            staging_smem[0] = i * 1.0
        # mark enter of critical region
        cute.arch.sync_threads()
        if warp_idx == 1:
            res[i] = staging_smem[0]
        # mark exit of critical region
        cute.arch.sync_threads()


@cute.jit
def run_synced_producer_consumer(res: cute.Tensor):
    @cute.struct
    class SharedStorage:
        staging_buffer: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, 1], 1024
        ]

    synced_producer_consumer(SharedStorage, res).launch(
        grid=(1, 1, 1), block=(64, 1, 1), smem=SharedStorage.size_in_bytes()
    )


res = torch.zeros((8,), device="cuda")
compiler: CompileCallable = cute.compile
options = (PtxasOptions("-O0 --verbose"), OptLevel(0), GenerateLineInfo())
#dsl = CutlassBaseDSL._get_dsl()
# options_str = "--opt-level 0 --generate-line-info --ptxas-options '--opt-level=0'"
options_str=""
compiled: CudaDialectJitCompiledFunction = cute.compile(run_synced_producer_consumer, from_dlpack(res), options=options_str)
compiled(from_dlpack(res))

#print(dsl.compile_options.to_str())
if 0:
    # %%
    res

    # %% [markdown]
    # <style>
    # div.mermaid > svg {
    #   width: 50% !important;
    #   height: auto !important;
    # }
    # </style>
    # 
    # ## Asynchronous Communication: Breaking the Synchronization Bottleneck
    # 
    # ### The Problem with Synchronous Patterns
    # 
    # The previous example demonstrates traditional synchronized communication between warps. While functional, this approach 
    # has significant performance limitations:
    # 
    # **Critical Section Analysis:**
    # - **First `__syncthreads()`**: Ensures data is written and ready for consumption
    # - **Second `__syncthreads()`**: Guarantees data has been consumed and memory can be safely overwritten
    # 
    # **Performance Impact:**
    # - All warps are forced into lockstep execution
    # - No computational overlap between producer and consumer operations
    # - Wasted cycles as warps wait at synchronization barriers
    # 
    # ### Hopper Architecture: Enabling Asynchronous Primitives
    # 
    # Starting with the Hopper architecture, CUDA introduced sophisticated asynchronous communication primitives that enable 
    # **warp specialization**—allowing different warps to perform distinct, specialized roles while maintaining loose coupling.
    # 
    # **Key Benefits:**
    # - **Overlapped Execution**: Producer and consumer warps can perform computations concurrently
    # - **Reduced Latency**: Eliminates unnecessary synchronization stalls
    # - **Better Resource Utilization**: Maximizes SM occupancy and throughput
    # 
    # ### Async Pipeline Communication Pattern
    # 
    # The async pipeline abstraction provides a elegant solution for producer-consumer communication without rigid synchronization constraints:
    # 
    # ```mermaid
    # sequenceDiagram
    #   participant W0 as Producer Warp
    #   participant Pipeline as Async Pipeline
    #   participant SMEM as Shared Memory  
    #   participant W1 as Consumer Warp
    #   
    #   W0->>Pipeline: Acquire (request write slot)
    #   activate W1
    #   Pipeline-->>W0: Grant access
    #   deactivate W1
    #   
    #   W1->>Pipeline: Wait (for data availability)
    #   activate Pipeline
    #   
    #   W0->>SMEM: Write data
    #   W0->>Pipeline: Commit (signal data ready)
    #   
    #   Pipeline-->>W1: Data available
    #   deactivate Pipeline
    #   
    #   activate W0
    #   SMEM->>W1: Read data
    #   deactivate W0
    #   W1->>Pipeline: Release (mark slot available)
    # ```
    # 
    # **Async Pipeline Advantages:**
    # - **Non-blocking Operations**: Warps can perform other work while waiting
    # - **Fine-grained Control**: Explicit control over data readiness and consumption
    # - **Scalable**: Supports multiple producer-consumer pairs efficiently

    # %% [markdown]
    # ### Async Pipeline API Reference
    # 
    # The `PipelineAsync` abstraction in CuTe DSL provides a comprehensive set of primitives for implementing efficient producer-consumer patterns:
    # 
    # #### Producer Operations
    # - **`PipelineProducer.acquire()`**: Blocks until a write slot becomes available (released by consumer)
    #   - Returns with a handle pointing to a available slot immediately if there is
    #   - Enables backpressure control to prevent buffer overflow
    #   - **`PipelineProducer.acquire_and_advance()`** additionally moves the producer's write index to the next buffer slot
    # 
    # - **`PipelineProducer.commit(PipelineProducer.ImmutableProducerHandle)`** / **`PipelineProducer.ImmutableProducerHandle.commit()`**: Signals that data has been written to the handle-pointed slot and is ready for consumption
    #   - Triggers waiting consumers
    #   - Maintains data consistency guarantees
    #   - If no assigned handle, **`PipelineConsumerHandle.release()`** tracks its internal maintained handle (pointed to the last one it acquires)
    # 
    # #### Consumer Operations  
    # - **`PipelineConsumer.wait()`**: Blocks until data becomes available for reading
    #   - Returns with a handle pointing to a committed slot when producer commits new data
    #   - Supports timeout and polling variants
    #   - **`PipelineConsumer.wait_and_advance()`** additionally moves the consumer's read index to the next buffer slot
    # 
    # - **`PipelineConsumerHandle.release(PipelineConsumer.ImmutableConsumerHandle)`** / **`PipelineConsumer.ImmutableConsumerHandle.release()`**: Marks data as consumed and the handle-pointed slot as consumed and available for reuse
    #   - Enables producers to acquire released slots
    #   - Critical for preventing deadlock in circular buffers
    #   - If no assigned handle, **`PipelineConsumerHandle.release()`** tracks its internal maintained handle (pointed to the last one it waits for)
    # 
    # #### Disclaimer
    # 
    # The `pipeline` APIs provided abstractions for developers to manage synchornization between warps, thread-blocks, etc. It doesn't provide deadlock-free guarantee. It's still developer's responsibility to write correct code to avoid deadlock.
    # 
    # #### Performance Characteristics
    # 
    # **Computational Overlap**: This asynchronous communication pattern enables limited but significant computational overlap:
    # - **Producer**: Can perform preprocessing, data transformation, or prefetching while consumer processes previous data
    # - **Consumer**: Can execute post-processing, result computation, or output operations while producer prepares next data
    # 
    # **Memory Efficiency**: Explicit slot management ensures optimal memory utilization without unnecessary copying or buffering.

    # %%
    @cute.kernel
    def async_pipeline_kernel(res: cute.Tensor):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            staging_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 1], 1024
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 64)

        # Warp 0
        producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 32
        )
        # Warp 1
        consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 32
        )

        pipeline = cutlass.pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            barrier_storage=storage.tma_mbar_ptr.data_ptr(),
        )

        staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(1))
        staging_smem.fill(0)
        cute.arch.sync_threads()

        producer, consumer = pipeline.make_participants()

        # Producer warp
        if warp_idx == 0:
            for i in cutlass.range(cute.size(res)):
                # Producer: Wait for data buffer is available
                handle = producer.acquire_and_advance()
                # Producer: Write data to shared memory
                staging_smem[handle.index] = 1.0 * i
                # Producer: Signal data is ready for consumption
                handle.commit()
            producer.tail()

        # Consumer warp
        if warp_idx == 1:
            for i in cutlass.range(cute.size(res)):
                # Consumer: Wait for producer to signal when data is available for use
                handle = consumer.wait_and_advance()
                # Conumer: consumes data
                res[i] = staging_smem[handle.index]
                # Conumer: Signal data buffer is ready for write
                handle.release()


    @cute.jit
    def async_pipeline(res: cute.Tensor):
        # Launch kernel with two warps: producer and consumer
        async_pipeline_kernel(res).launch(grid=(1, 1, 1), block=(64, 1, 1))


    res = torch.zeros((8,), device="cuda")
    async_pipeline(from_dlpack(res))

    # %%
    res

    # %% [markdown]
    # <style>
    # div.mermaid > svg {
    #   width: 50% !important;
    #   height: auto !important;
    # }
    # </style>
    # 
    # ## Advanced Pattern: Staged Async Pipeline with Circular Buffering
    # 
    # ### Limitations of Single-Stage Pipelines
    # 
    # While async communication provides significant improvements over synchronous patterns, single-stage pipelines 
    # still exhibit serialization bottlenecks:
    # 
    # **Dependency Chain Analysis:**
    # ```mermaid
    # sequenceDiagram
    #   participant W0 as Producer
    #   participant Pipeline as Pipeline
    #   participant W1 as Consumer
    #   
    #   W0->>Pipeline: Acquire
    #   Note over W0,W1: Producer waits here
    #   W1->>Pipeline: Release
    #   Pipeline-->>W0: Granted
    # ```
    # 
    # **Performance Bottleneck**: The producer must wait for the consumer to complete processing and release the buffer 
    # before acquiring the next write slot. This creates a serialization point that limits overall throughput.
    # 
    # ### Multi-Stage Pipeline Architecture
    # 
    # The **staged async pipeline** implements a circular buffer managed by an array of synchronization barriers, 
    # enabling much higher degrees of parallelism:
    # 
    # #### Core Concepts
    # 
    # **Circular Buffer Management:**
    # - **Multiple Stages**: Support for N concurrent buffer slots (typically 2-8 stages)
    # - **Independent Indexing**: Producer and consumer maintain separate advancement indices
    # - **Barrier Array**: Each stage has an associated memory barrier for fine-grained synchronization
    # 
    # #### Enhanced API Operations
    # 
    # - **`PipelineProducer.advance()`**: Moves the producer's write index to the next buffer slot
    #   - Enables round-robin buffer allocation
    #   - Allows producer to continue without waiting for all previous data to be consumed
    #   - Can be conducted implicitly when calling **`PipelineProducer.require_and_advance()`**
    # 
    # - **`PipelineConsumer.advance()`**: Moves the consumer's read index to the next buffer slot
    #   - Maintains proper ordering of data consumption
    #   - Signals availability of processed slots
    #   - Can be conducted implicitly when calling **`PipelineConsumer.wait_and_advance()`**
    # 
    # - **`PipelineProducer.ImmutableResourceHandle.index`** / **`PipelineConsumer.ImmutableResourceHandle.index`**: Returns pointed buffer slot index
    #   - Used for addressing specific staging buffer locations
    #   - Enables direct slot-based data access
    # 
    # ### Circular Buffer State Visualization
    # 
    # ```
    # Legend:
    #     W: Currently being written (producer active)
    #     D: Data ready for consumption  
    #     R: Currently being read (consumer active)
    #     X: Empty slot available for writing
    #   
    #           Advance Direction
    #         <-------------------
    # 
    #          Producer   Consumer
    #              |         ^
    #              V         |
    #         +-----------------+
    #       --|X|X|W|D|D|D|D|R|X|<-.
    #      /  +-----------------+   \
    #      |                        |
    #      `------------------------' 
    # ```
    # 
    # **Key Advantages:**
    # - **Increased Throughput**: Producer can stay ahead of consumer by multiple stages
    # - **Latency Hiding**: Consumer processing latency is hidden by buffered data
    # - **Better Resource Utilization**: Both warps can maintain high activity levels
    # - **Scalable Design**: Buffer depth can be tuned based on workload characteristics
    # 
    # The following implementation demonstrates efficient multi-stage pipeline communication with proper circular buffer management:

    # %%
    @cute.kernel
    def async_pipeline_staged_kernel(
        SharedStorage: cutlass.Constexpr, res: cute.Tensor, staging: cute.Tensor
    ):
        stages = cute.size(staging)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 64)

        # Warp 0
        producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 32
        )
        # Warp 1
        consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 32
        )

        pipeline = cutlass.pipeline.PipelineAsync.create(
            num_stages=stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            barrier_storage=storage.tma_mbar_ptr.data_ptr(),
        )

        staging_smem = storage.staging_buffer.get_tensor(staging.layout)
        staging_smem.fill(0)
        cute.arch.sync_threads()

        producer, consumer = pipeline.make_participants()

        # Producer warp
        if warp_idx == 0:
            for i in cutlass.range(cute.size(res)):
                handle = producer.acquire_and_advance()
                staging_smem[handle.index] = 1.0 * i
                handle.commit()  # or producer.commit(handle)

            # prevents CTA0 from retiring until it receives all expected arrives.
            producer.tail()

        # Consumer warp
        if warp_idx == 1:
            for i in cutlass.range(cute.size(res)):
                handle = consumer.wait_and_advance()
                res[i] = staging_smem[handle.index]
                handle.release()  # or consumer.release(handle)

        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            staging.store(staging_smem.load())


    @cute.jit
    def async_pipeline_staged(res: cute.Tensor, staging: cute.Tensor):
        stages = cute.size(staging)

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
            staging_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, stages], 1024
            ]

        async_pipeline_staged_kernel(SharedStorage, res, staging).launch(
            grid=(1, 1, 1), block=(64, 1, 1), smem=SharedStorage.size_in_bytes()
        )


    res = torch.zeros((8,), device="cuda")
    staging = torch.zeros((5,), device="cuda")
    async_pipeline_staged(from_dlpack(res), from_dlpack(staging))
    torch.cuda.synchronize()

    # %%
    res, staging

    # %% [markdown]
    # ### Try Acquire/Wait
    # 
    # In some circumstances, developers may want to just check status of pipeline state without blocking. This could benefit some cases that we have independent instructions to hide latency of checking pipeline state. We provided `try_aquire` or `try_wait` which are non-blocking APIs.  




