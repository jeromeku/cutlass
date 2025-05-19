# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Your First Program with CuTe DSL
#
# ## Introduction
#
# Welcome! In this tutorial, we'll write a simple "Hello World" program that runs on your GPU using CuTe DSL. This will help you understand the basics of GPU programming with our framework.
#
# ### What You'll Learn
#
# - How to write code that runs on both CPU (host) and GPU (device),
# - How to launch a GPU kernel (a function that runs on the GPU),
# - Basic CUDA concepts like threads and thread blocks,
#
# ### Step 1: Import Required Libraries
#
# First, let's import the libraries we need:
#

# %%
import os

os.environ["CUTE_DSL_LOG_TO_FILE"] = "cutlass_DSL.log"
os.environ["CUTE_DSL_LOG_LEVEL"] = "10"


# %%
import cuda
cuda

# %%

import cutlass               
import cutlass.cute as cute
from cutlass.base_dsl.utils.logger import setup_log, logger
import logging

# %%
logger.setLevel(logging.DEBUG)

# %%
logger


# %% [markdown]
# ### Step 2: Write Our GPU Kernel
#
# A GPU kernel is a function that runs on the GPU. Here's a simple kernel that prints "Hello World".
# Key concepts:
#
# - `@cute.kernel`: This decorator tells CUTLASS that this function should run on the GPU
# - `cute.arch.thread_idx()`: Gets the ID of the current GPU thread (like a worker's ID number)
# - We only want one thread to print the message (thread 0) to avoid multiple prints
#

# %%
@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if tidx == 0:
        cute.printf("Hello world")


# %% [markdown]
# ### Step 3: Write Our Host Function
#
# Now we need a function that sets up the GPU and launches our kernel.
# Key concepts:
#
# - `@cute.jit`: This decorator is for functions that run on the CPU but can launch GPU code
# - We need to initialize CUDA before using the GPU
# - `.launch()` tells CUDA how many blocks, threads, shared memory, etc. to use
#

# %%
@cute.jit
def hello_world():

    # Print hello world from host code
    cute.printf("hello world")
    
    # Initialize CUDA context for launching a kernel with error checking
    # We make context initialization explicit to allow users to control the context creation 
    # and avoid potential issues with multiple contexts
    cutlass.cuda.initialize_cuda_context()

    # Launch kernel
    kernel().launch(
        grid=(1, 1, 1),   # Single thread block
        block=(32, 1, 1)  # One warp (32 threads) per thread block
    )


# %% [markdown]
# ### Step 4: Run Our Program
#
# There are 2 ways we can run our program:
#
# 1. compile and run immediately
# 2. separate compilation which allows us to compile the code once and run multiple times
#
# Please note the `Compiling...` for Method 2 prints before the "Hello world" of the first kernel. This shows the asynchronous behavior between CPU and GPU prints.
#

# %%
# Method 1: Just-In-Time (JIT) compilation - compiles and runs the code immediately
print("Running hello_world()...")
hello_world()

# Method 2: Compile first (useful if you want to run the same code multiple times)
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)
# Run the pre-compiled version
print("Running compiled version...")
hello_world_compiled()

# %%
