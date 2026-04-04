from __future__ import annotations

import pytest
import torch

from codex.standalone_gather_a import gather_a, gather_a_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the standalone CuTe kernel")
@pytest.mark.parametrize(
    "case",
    [
        {"tile_M": 128, "tile_K": 64, "K_start": 0, "K_extent": 256, "token_group_size": 192},
        {"tile_M": 128, "tile_K": 64, "K_start": 64, "K_extent": 128, "token_group_size": 113},
        {"tile_M": 64, "tile_K": 64, "K_start": 0, "K_extent": 128, "token_group_size": 257},
    ],
)
def test_standalone_gather_a_matches_reference(case):
    """Validate the tiled standalone kernel against a direct PyTorch gather reference."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    source_rows = 512
    source_cols = 256
    A = torch.randn(source_rows, source_cols, device=device, dtype=dtype)

    # Include repeated indices to exercise the real MoE gather case.
    base_idx = torch.randint(0, source_rows, (384,), device=device, dtype=torch.int32)
    base_idx[8:16] = base_idx[0]
    A_idx = base_idx

    out = gather_a(
        A,
        A_idx,
        tile_M=case["tile_M"],
        tile_K=case["tile_K"],
        K_start=case["K_start"],
        K_extent=case["K_extent"],
        token_group_size=case["token_group_size"],
    )
    ref = gather_a_reference(
        A,
        A_idx,
        K_start=case["K_start"],
        K_extent=case["K_extent"],
        token_group_size=case["token_group_size"],
    )

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
