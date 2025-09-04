from cutlass_package import cute


@cute.kernel
def _empty():
    # Minimal device kernel (no body). IR still emits a gpu.func and a launch site.
    return


@cute.jit
def run():
    # Builds MLIR, compiles to cubin, loads via CUDA driver, and launches the kernel.
    _empty().launch(grid=[1, 1, 1], block=[1, 1, 1])


if __name__ == "__main__":
    run()

