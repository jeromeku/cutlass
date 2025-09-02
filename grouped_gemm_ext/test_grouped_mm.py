import torch
from groupedmm_ext.jit_build import build_groupedmm_ext, grouped_mm
CUTLASS_PATH = "/home/jeromeku/cutlass"

ext = build_groupedmm_ext(
    cutlass_path=CUTLASS_PATH,  # or set CUTLASS_PATH env var
    verbose=True
)

# Call allocating variant
# G, M, K, N = 8, 64, 512, 2048
# A = torch.randn(G, M, K, device="cuda", dtype=torch.bfloat16).contiguous()
# B = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16).contiguous()
# Y = torch.ops.groupedmm_ext._grouped_mm(A, B)  # or grouped_mm(A, B)

# # Preallocated variant (exactly the same underlying kernel)
# out = torch.empty(G*M, N, device="cuda", dtype=torch.bfloat16)
# torch.ops.groupedmm_ext._grouped_mm_out(A, B, None, None, out)
def grouped_mm_helper(alist, blist, gOlist, agradlist, bgradlist, outlist):
    for a, b, out in zip(alist, blist, outlist): #, gO, agrad, bgrad, out in zip(alist, blist, gOlist, agradlist, bgradlist, outlist):
        a = a.clone().detach().requires_grad_()
        b = b.clone().detach().requires_grad_()
        out_ref = torch.mm(a, b.t())
        # out_ref.backward(gO)
        torch.testing.assert_close(out, out_ref)
        # if agrad is not None:
        #     torch.testing.assert_close(agrad, a.grad)
        #     torch.testing.assert_close(bgrad, b.grad)

def test_grouped_gemm_2d_3d(a_row_major=True, b_row_major=True, strided=False):
    device = "cuda"
    dtype = torch.bfloat16
    s_int = int(strided)
    m, n, k, n_groups = 16, 32, 64, 4
    if a_row_major:
        a = torch.randn(m * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
    else:
        a = torch.randn(k, (m + 2 * s_int) * n_groups, device=device, dtype=dtype).t()[:m * n_groups, :]

    if b_row_major:
        b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
    else:
        b = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                        dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]

    a.requires_grad_(True)
    b.requires_grad_(True)

    a_contig = a if a_row_major else a.t()
    assert a_contig.is_contiguous() is not strided
    b_contig = b if b_row_major else b.transpose(-2, -1)
    assert b_contig.is_contiguous() is not strided
    for check_zero_size in (False, True):
        if check_zero_size and n_groups <= 1:
            continue

        a.grad = None
        b.grad = None
        offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
        if check_zero_size:
            offs[0] = offs[1]

        f = torch.ops.groupedmm_ext._grouped_mm
        out = f(a, b.transpose(-2, -1), offs=offs)

        # gO = torch.rand_like(out)
        # # if not check_zero_size:
        # #     out.backward(gO)
        # offs_cpu = offs.cpu()
        # alist, agradlist, gOlist, outlist = [], [], [], []
        # bgradlist = [None] * n_groups if check_zero_size else b.grad
        # start = 0
        # for i in range(n_groups):
        #     alist.append(a[start:offs_cpu[i]])
        #     agradlist.append(None)# if check_zero_size else a.grad[start:offs_cpu[i]])
        #     outlist.append(out[start:offs_cpu[i]])
        #     gOlist.append(gO[start:offs_cpu[i]])
        #     start = offs_cpu[i]
        # breakpoint()
        # grouped_mm_helper(alist, b, gOlist, agradlist, bgradlist, outlist)
test_grouped_gemm_2d_3d()