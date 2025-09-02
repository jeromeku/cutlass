import torch
from groupedmm_ext.jit_build import build_groupedmm_ext

def grouped_mm_helper(alist, blist, outlist):
    breakpoint()
    for a, b, out in zip(
        alist, blist, outlist
    ):
        a = a.clone().detach().requires_grad_()
        b = b.clone().detach().requires_grad_()
        out_ref = torch.mm(a, b.t())
        torch.testing.assert_close(out, out_ref)


def test_grouped_gemm_2d_3d_forward(
    m=16, n=32, k=64, n_groups=4, a_row_major=True, b_row_major=True, strided=False
):
    device = "cuda"
    dtype = torch.bfloat16
    s_int = int(strided)

    if a_row_major:
        a = torch.randn(m * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
    else:
        a = torch.randn(k, (m + 2 * s_int) * n_groups, device=device, dtype=dtype).t()[
            : m * n_groups, :
        ]

    if b_row_major:
        b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[
            :: (1 + s_int), :, :k
        ]
    else:
        b = torch.randn(
            n_groups * (1 + s_int), k * (1 + s_int), n, device=device, dtype=dtype
        ).transpose(-2, -1)[:: (1 + s_int), :, :k]

    a_contig = a if a_row_major else a.t()
    assert a_contig.is_contiguous() is not strided
    b_contig = b if b_row_major else b.transpose(-2, -1)
    assert b_contig.is_contiguous() is not strided

    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    f = torch.ops.groupedmm_ext._grouped_mm
    out = f(a, b.transpose(-2, -1), offs=offs)
    print(f"{a.shape=}, {b.shape=}, {out.shape=}")

    offs_cpu = offs.cpu()
    alist, outlist = [], []
    start = 0
    for i in range(n_groups):
        alist.append(a[start : offs_cpu[i]])
        outlist.append(out[start : offs_cpu[i]])
        start = offs_cpu[i]
    breakpoint()
    grouped_mm_helper(alist, b, outlist)


if __name__ == "__main__":
    CUTLASS_PATH = "/home/jeromeku/cutlass"

    ext = build_groupedmm_ext(
        cutlass_path=CUTLASS_PATH,  # or set CUTLASS_PATH env var
        verbose=True,
    )

    test_grouped_gemm_2d_3d_forward()
