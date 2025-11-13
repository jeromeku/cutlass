
import argparse
import operator
import time
from typing import Type, List

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    result: cute.Tensor,
    stream: cuda.CUstream,
):
    """CUDA kernel applying binary operator on each element of two n-D input tensors in
    CuTe Python and store to result tensor.

    :param op: Binary operator or lambda function to apply element-wise
    :type op: cutlass.Constexpr
    :param a: First input tensor
    :type a: cute.Tensor
    :param b: Second input tensor
    :type b: cute.Tensor
    :param result: Output tensor to store the results of op(a, b)
    :type result: cute.Tensor
    :return: None
    :rtype: None

    .. code-block:: python

        # Example 1: Adding two tensors
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cuda")
        y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device="cuda")
        result = torch.empty_like(x)
        elementwise_apply(operator.add, from_dlpack(x), from_dlpack(y), from_dlpack(result))
        # result:
        # tensor([[6.0, 8.0],
        #         [10.0, 12.0]], device='cuda:0')

        # Example 2: Using a lambda function
        elementwise_apply(lambda a, b: a * a + b * b, from_dlpack(x), from_dlpack(y), from_dlpack(result))
        # result:
        # tensor([[  2.,   8.],
        #         [ 54., 512.]], device='cuda:0')
    """

    # Baseline: naive TV layout
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (512, 4) tile
    #   * tidx maps to mode-0 but input layout is contiguous on mode-1, performance will be bad
    # tv_layout = cute.make_layout((128, (4, 4)), stride=(4, (512, 1)))
    # cta_tiler = (512, 4)

    # Opt-1: better TV layout with better 1D thread layout (SOL with 1D thread layout)
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (4, 512) tile
    #   * tidx maps to mode-1 which is leading mode of input tensor for coalesced load
    # tv_layout = cute.make_layout((128, (4, 4)), stride=(16, (4, 1)))
    # cta_tiler = (4, 512)

    # Opt-2: 2D tile but worse
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (128, 16) logical tile
    #   * V layout is bad as contiguous mode is not on right-most
    #     * `cute.copy` only supports vectorize when stride-1 of v-layout on right-most )
    # tv_layout = cute.make_layout(((32, 4), (4, 4)), stride=((4, 512), (1, 128)))
    # cta_tiler = (128, 16)

    # Opt-3: SOL with 2D thread tile
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (16, 128) logical tile
    #   * tidx maps to mode-1 and input layout is contiguous on mode-1 for coalesced load-store
    ACCESS_SIZE_BITS = 128
    VECTOR_WIDTH = ACCESS_SIZE_BITS // a.element_type.width

    thr_layout = cute.make_ordered_layout((4,32), order=(1,0)) #cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_ordered_layout((4, VECTOR_WIDTH), order=(1,0)) #cute.make_layout((4, VECTOR_WIDTH), stride=(4, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   a = {a.type}")
    print(f"[DSL INFO]   b = {b.type}")
    print(f"[DSL INFO]   result = {result.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   thr_layout = {thr_layout}")
    print(f"[DSL INFO]   val_layout = {val_layout}")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")
    thread0_vals = cute.slice_(tv_layout, (0, None))
    print(f"[DSL_INFO]: thread 0 val layout: {thread0_vals}")
    gA = cute.zipped_divide(a, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(b, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(result, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    idC = cute.make_identity_tensor(result.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    print(f"[DSL INFO]   coord tensor = {cC.type}")
    print(f"[DSL INFO]   coord tensor = {cC}")


def run_elementwise_apply_and_verify(
    op,
    M,
    N,
    dtype: Type[cutlass.Numeric],
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=100,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    # Create non default CUDA stream from PyTorch
    torch_stream = torch.cuda.Stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    print(f"\nRunning Elementwise Apply test with:")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    # Allocate tensors with random values.
    a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
    b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
    c = torch.zeros_like(a)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    epsilon = 1.2
    if op in (operator.truediv, operator.floordiv):
        b = torch.where(b == 0, torch.tensor(epsilon), b)

    print("Executing elementwise apply kernel...")

    elementwise_apply(
        op,
        from_dlpack(a),
        from_dlpack(b),
        from_dlpack(c).mark_layout_dynamic(),
        current_stream,
    )

    # if not benchmark:
    #     return

    # compiled_func = cute.compile(
    #     elementwise_apply,
    #     op,
    #     from_dlpack(a),
    #     from_dlpack(b),
    #     from_dlpack(c).mark_layout_dynamic(),
    #     current_stream,
    # )
    # compiled_func(a, b, c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise apply to demonstrate building elementwise kernels"
    )
    parser.add_argument("--M", default=128, type=int)
    parser.add_argument("--N", default=128, type=int)
    parser.add_argument("--op", default="add", type=str)
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run_elementwise_apply_and_verify(
        getattr(operator, args.op),
        args.M,
        args.N,
        dtype=cutlass.Float16,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
    )
    print("\nPASS")
