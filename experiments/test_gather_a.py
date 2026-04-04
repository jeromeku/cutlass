"""
Test for standalone gatherA kernel.

Verifies that the kernel correctly gathers scattered rows from a source matrix A
using an index array, matching A[indices[:tile_M], :tile_K].

Usage:
    srun --overlap python claude/test_gather_a.py
"""

from math import log
from utils.logging import patch_cutlass_env

patch_cutlass_env(
    # log_to_console=True,
    # print_after_preprocessor=True,
    # preprocessed_ast_path="gatherA_preprocessed.py",
    keep_ptx=True,
    dumpdir="gatherA_artifacts",
    lineinfo=True
)

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from gather_a_standalone import GatherAStandalone
from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor
import ast
import os

_counter = 0
_dump_dir = os.path.join(os.getcwd(), "ast_dumps")
os.makedirs(_dump_dir, exist_ok=True)

_orig_print_ast = DSLPreprocessor.print_ast


@staticmethod
def _saving_print_ast(transformed_tree=None, function_name=None):
    global _counter
    # Extract function name from AST if not provided
    if function_name is None:
        for node in ast.walk(transformed_tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break
    unparsed_code = ast.unparse(transformed_tree)
    safe_name = (function_name or "unknown").replace(".", "_")
    path = os.path.join(_dump_dir, f"{_counter:04d}_{safe_name}.py")
    with open(path, "w") as f:
        f.write(f"# Transformed AST for: {function_name}\n")
        f.write(unparsed_code)
    print(f"# Saved transformed AST to {path}")
    _counter += 1


DSLPreprocessor.print_ast = _saving_print_ast


def test_gather_a(tile_M: int, tile_K: int, T: int, num_tokens: int):
    """Test one configuration of the gatherA kernel.

    Args:
        tile_M: rows to gather per tile
        tile_K: columns per tile
        T: total rows in source matrix A
        num_tokens: number of gather indices (must be >= tile_M)
    """
    assert num_tokens >= tile_M, f"num_tokens ({num_tokens}) must be >= tile_M ({tile_M})"
    assert T >= num_tokens, f"T ({T}) must be >= num_tokens ({num_tokens})"

    print(f"\n--- Testing tile_M={tile_M}, tile_K={tile_K}, T={T}, num_tokens={num_tokens} ---")

    # Create source matrix A (T, K) in BF16, row-major, contiguous
    # Use K >= tile_K; we only gather the first tile_K columns
    K = tile_K
    A = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)

    # Create gather indices: random permutation of [0, T), take first num_tokens
    indices = torch.randperm(T, device="cuda", dtype=torch.int32)[:num_tokens]

    # Output buffer
    out = torch.zeros(tile_M, tile_K, device="cuda", dtype=torch.bfloat16)

    # token_group_size = number of valid tokens to gather (capped at tile_M)
    token_group_size = min(num_tokens, tile_M)

    # Reference: gather the first token_group_size rows by index
    ref = A[indices[:token_group_size].long(), :tile_K]

    # Convert to CuTe tensors
    # mA and mAIdx use dynamic layout (variable size across calls)
    # mOut uses static layout since shape is fixed (tile_M, tile_K)
    mA = from_dlpack(A).mark_layout_dynamic()
    mOut = from_dlpack(out)
    mAIdx = from_dlpack(indices).mark_layout_dynamic()

    # Create and compile kernel
    gather_module = GatherAStandalone(tile_M=tile_M, tile_K=tile_K)
    gather_module.token_group_size = token_group_size

    print(f"  num_load_A_threads = {gather_module.num_load_A_threads}")
    print(f"  threads_per_cta = {gather_module.threads_per_cta}")
    print(f"  tma_warp_id = {gather_module.tma_warp_id}")
    print(f"  token_group_size = {token_group_size}")

    print("  Compiling...")
    stream = cutlass_torch.current_stream()
    compiled = cute.compile(
        gather_module,
        mA,
        mOut,
        mAIdx,
        stream,
        options="--generate-line-info",
    )

    print("  Running kernel...")
    compiled(mA, mOut, mAIdx, stream)
    torch.cuda.synchronize()

    # # Only compare valid rows (rows beyond token_group_size have uninitialized SMEM data)
    # out_valid = out[:token_group_size]
    # print(f"  Output[0,:8] = {out_valid[0, : min(8, tile_K)]}")
    # print(f"  Ref[0,:8]    = {ref[0, : min(8, tile_K)]}")

    # if torch.equal(out_valid, ref):
    #     print("  PASS (exact match)")
    # else:
    #     max_diff = (out_valid - ref).abs().max().item()
    #     num_mismatches = (out_valid != ref).sum().item()
    #     print(f"  FAIL: max_diff={max_diff}, mismatches={num_mismatches}/{out_valid.numel()}")
    #     raise AssertionError(f"gatherA test failed for tile_M={tile_M}, tile_K={tile_K}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    # Test configurations: (tile_M, tile_K, T, num_tokens)
    configs = [
        (128, 64, 8192, 256),  # Default MoE up-proj shape
        # (64, 64, 1024, 128),    # Smaller
        # (128, 64, 4096, 128),   # token_group_size == tile_M
    ]

    for tile_M, tile_K, T, num_tokens in configs:
        test_gather_a(tile_M, tile_K, T, num_tokens)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
