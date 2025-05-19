# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 06: CUDA Graphs
#
# In this example we demonstrate how to use CUDA graphs through PyTorch with CuTe DSL.
# The process of interacting with PyTorch's CUDA graph implementation requires exposing PyTorch's CUDA streams to CUTLASS.
#
# To use CUDA graphs with Blackwell requires a version of PyTorch that supports Blackwell.
# This can be obtained through:
# - The [PyTorch NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
# - [PyTorch 2.7 with CUDA 12.8 or later](https://pytorch.org/) (e.g., `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`)
# - Building PyTorch directly with your version of CUDA.

# %%
# import torch for CUDA graphs
import torch
import cutlass
import cutlass.cute as cute
# import CUstream type from the cuda driver bindings
from cuda.bindings.driver import CUstream
# import the current_stream function from torch
from torch.cuda import current_stream


# %% [markdown]
# ## Kernel Creation
#
# We create a kernel which prints "Hello world" as well as a host function to launch the kernel.
# We then compile the kernel for use in our graph, by passing in a default stream.
#
# Kernel compilation before graph capture is required since CUDA graphs cannot JIT compile kernels during graph execution.

# %%
@cute.kernel
def hello_world_kernel():
    """
    A kernel that prints hello world
    """
    cute.printf("Hello world")

@cute.jit
def hello_world(stream : CUstream):
    """
    Host function that launches our (1,1,1), (1,1,1) grid in stream
    """
    hello_world_kernel().launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

# Grab a stream from PyTorch, this will also initialize our context
# so we can omit cutlass.cuda.initialize_cuda_context()
stream = current_stream()
hello_world_compiled = cute.compile(hello_world, CUstream(stream.cuda_stream))

# %% [markdown]
# ## Creating and replaying a CUDA Graph
#
# We create a stream through torch as well as a graph.
# When we create the graph we can pass the stream we want to capture to torch. We similarly run the compiled kernel with the stream passed as a CUstream.
#
# Finally we can replay our graph and synchronize.

# %%
# Create a CUDA Graph
g = torch.cuda.CUDAGraph()
# Capture our graph
with torch.cuda.graph(g):
    # Turn our torch Stream into a cuStream stream.
    # This is done by getting the underlying CUstream with .cuda_stream
    graph_stream = CUstream(current_stream().cuda_stream)
    # Run 2 iterations of our compiled kernel
    for _ in range(2):
        # Run our kernel in the stream
        hello_world_compiled(graph_stream)

# Replay our graph
g.replay()
# Synchronize all streams (equivalent to cudaDeviceSynchronize() in C++)
torch.cuda.synchronize()

# %% [markdown]
# Our run results in the following execution when viewed in NSight Systems:
#
# ![Image of two hello world kernels run back to back in a CUDA graph](images/cuda_graphs_image.png)
#
# We can observe the launch of the two kernels followed by a `cudaDeviceSynchronize()`.
#
# Now we can confirm that this minimizes some launch overhead:

# %%
# Get our CUDA stream from PyTorch
stream = CUstream(current_stream().cuda_stream)

# Create a larger CUDA Graph of 100 iterations
g = torch.cuda.CUDAGraph()
# Capture our graph
with torch.cuda.graph(g):
    # Turn our torch Stream into a cuStream stream.
    # This is done by getting the underlying CUstream with .cuda_stream
    graph_stream = CUstream(current_stream().cuda_stream)
    # Run 2 iterations of our compiled kernel
    for _ in range(100):
        # Run our kernel in the stream
        hello_world_compiled(graph_stream)

# Create CUDA events for measuring performance
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Run our kernel to warm up the GPU
for _ in range(100):
    hello_world_compiled(stream)

# Record our start time
start.record()
# Run 100 kernels
for _ in range(100):
    hello_world_compiled(stream)
# Record our end time
end.record()
# Synchronize (cudaDeviceSynchronize())
torch.cuda.synchronize()

# Calculate the time spent when launching kernels in a stream
# Results are in ms
stream_time = start.elapsed_time(end) 

# Warmup our GPU again
g.replay()
# Record our start time
start.record()
# Run our graph
g.replay()
# Record our end time
end.record()
# Synchronize (cudaDeviceSynchronize())
torch.cuda.synchronize()

# Calculate the time spent when launching kernels in a graph
# units are ms
graph_time = start.elapsed_time(end)

# %%
# Print out speedup when using CUDA graphs
percent_speedup = (stream_time - graph_time) / graph_time
print(f"{percent_speedup * 100.0:.2f}% speedup when using CUDA graphs for this kernel!")
