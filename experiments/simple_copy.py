from utils import logging

logging.patch_cutlass_env(log_to_console=True)

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from cutlass.base_dsl.compiler import CompileCallable, CompileOptions

NUM_BLOCKS = 1
THREADS_PER_BLOCK = 1


class SimpleKernel:
    def __init__(
        self,
        num_blocks: int = NUM_BLOCKS,
        threads_per_block: int = THREADS_PER_BLOCK,
        compile_opts: list[any] = None,
    ):
        self.num_blocks = num_blocks
        self.threads_per_block = threads_per_block
        self.compile_opts = compile_opts or CompileOptions()
        print(f"Initializing with compile opts: {compile_opts}")

    @cute.kernel
    def kernel(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        global_idx = bidx * bdim + tidx

        if global_idx < cute.size(input):
            val = input[global_idx]
            output[global_idx] = val + 1

    @cute.jit()
    def _launch(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
    ):
        self.kernel(input, output).launch(
            grid=(self.num_blocks, 1, 1),
            block=(self.threads_per_block, 1, 1),
        )

    def run(self, input: torch.Tensor, output: torch.Tensor):
        self._launch(from_dlpack(self.input), from_dlpack(self.outputs))

    def compile(self, input: torch.Tensor, output: torch.Tensor = None, opts: list[str] = None) -> CompileCallable:
        opts = opts or self.compile_opts
        compiler: CompileCallable = cute.compile
        args = from_dlpack(input), from_dlpack(output)
        return compiler[opts](self._launch, *args)


if __name__ == "__main__":
    from cutlass.base_dsl.compiler import (
        PtxasOptions,
        EnableAssertions,
        GenerateLineInfo,
        KeepCUBIN,
        KeepPTX,
        DumpDir,
        GPUArch,
    )

    compile_opts = CompileOptions()
    """
    CompileOptions
        OptLevel: OptLevel(3),
        PtxasOptions: PtxasOptions(""),
        # Debugging options
        EnableAssertions: EnableAssertions(False),
        GenerateLineInfo: GenerateLineInfo(False),
        KeepCUBIN: KeepCUBIN(False),
        KeepPTX: KeepPTX(False),
        GPUArch: GPUArch(""),
        LinkLibraries: LinkLibraries(""),
        EnableTVMFFI: EnableTVMFFI(False),
        DumpDir: DumpDir(""),
    """
    opts = (PtxasOptions("-v"), EnableAssertions(), GenerateLineInfo(), KeepPTX(), DumpDir("copy_kernel_dump"))
    input = torch.ones(10, dtype=torch.uint32, device="cuda:0")
    output = torch.empty_like(input)
    kernel = SimpleKernel()
    compiled = kernel.compile(input, output, opts=opts)
