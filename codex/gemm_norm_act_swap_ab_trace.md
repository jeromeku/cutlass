# `gemm_norm_act_tuned`: `swap_ab` and `.mT` at the call site

Relevant wrapper site:

- [gemm_interface.py](../thirdparty/quack/quack/gemm_interface.py#L1697)
- [gemm_interface.py](../thirdparty/quack/quack/gemm_interface.py#L1719)

Kernel contract being targeted:

- [gemm_norm_act.py](../thirdparty/quack/quack/gemm_norm_act.py#L246)
- [gemm_norm_act.py](../thirdparty/quack/quack/gemm_norm_act.py#L286)

Layout helpers:

- [gemm_tvm_ffi_utils.py](../thirdparty/quack/quack/gemm_tvm_ffi_utils.py#L22)
- [gemm_tvm_ffi_utils.py](../thirdparty/quack/quack/gemm_tvm_ffi_utils.py#L41)

## Big picture

The public API for `gemm_norm_act` is:

- `A`: `(M, K)` or `(L, M, K)`
- `B`: `(K, N)` or `(L, K, N)`
- `preact_out`, `C`, `postact_out`: `(M, N)` or `(L, M, N)`
- `rstd`: `(M,)` or `(L, M)`

But the lower-level kernel wrapper `gemm_norm_act_fn` wants:

- `A`: `(l, m, k)`
- `B`: `(l, n, k)`
- `D`, `C`, `PostAct`: `(l, m, n)`

That is why the wrapper always does `B = B.mT` first: it converts logical `B` from `(K, N)` into the kernel-facing `(N, K)` view.

## What `.mT` means here

`Tensor.mT` transposes the last two dimensions only.

- `B: (K, N) -> (N, K)`
- `B: (L, K, N) -> (L, N, K)`
- `D/C/PostAct: (L, M, N) -> (L, N, M)`

This is a view, not a data copy.

## Line-by-line at the swap site

At [gemm_interface.py](../thirdparty/quack/quack/gemm_interface.py#L1720), the wrapper chooses between two equivalent formulations:

### `swap_ab == False`

The kernel sees the normal orientation:

```python
A
B.mT
D
C
PostAct
colvec=rstd
rowvec=None
```

Shapes after wrapper normalization:

- `A`: `(L, M, K)`
- `B`: `(L, N, K)` because of the earlier `B = B.mT`
- `D`, `C`, `PostAct`: `(L, M, N)`
- `rstd`: `(L, M)`

So the kernel computes the usual logical result:

- GEMM output shape: `(M, N)`
- normalization vector is a column-vector over `M`
- `D` and `PostAct` stay logically `(M, N)`

Layout consequence:

- after `perm3d_single`, `D/C/PostAct` become `(M, N, L)`
- for a normal contiguous `(L, M, N)` tensor, `stride(1) == 1`, so `get_major(..., "m", "n")` returns `"n"` at [gemm_norm_act.py](../thirdparty/quack/quack/gemm_norm_act.py#L294)
- `"n"` here means N is the contiguous dimension, i.e. standard row-major storage for an `(M, N)` tensor

### `swap_ab == True`

The wrapper deliberately reformulates the same math as a transposed GEMM:

```python
A = B.mT
B = A
D = D.mT
C = C.mT
PostAct = PostAct.mT
colvec=None
rowvec=rstd
```

Using the original logical names, the kernel now sees:

- first operand: original `B.mT`, shape `(L, N, K)`
- second operand: original `A`, shape `(L, M, K)`
- output buffers: `(L, N, M)` views via `.mT`
- normalization vector moved from `colvec` to `rowvec`

Why this is still correct:

- normal path computes `A @ B`, shape `(M, N)`
- swapped path computes `(B.mT @ A.mT)`, shape `(N, M)`
- that is exactly `(A @ B).T`

Because the output buffers are also passed as transposed views, the kernel writes `(N, M)` into a view backed by the same original `(M, N)` storage. The wrapper therefore preserves the user-visible result while letting the autotuner choose the faster orientation.

## Why `rstd` changes from `colvec` to `rowvec`

From the public API and reference implementation, `rstd` is logically an `M`-length vector multiplied along rows:

- [gemm_interface.py](../thirdparty/quack/quack/gemm_interface.py#L1871)
- [gemm_interface.py](../thirdparty/quack/quack/gemm_interface.py#L1948)

In the normal `(M, N)` orientation, that is a column-vector broadcast, so the wrapper passes:

- `colvec=rstd`

After swapping, the kernel operates on the transposed `(N, M)` view. The same logical `M` axis is now the trailing axis, so the identical data must be broadcast as a row-vector instead:

- `rowvec=rstd`

This is why lines [1734-1735](../thirdparty/quack/quack/gemm_interface.py#L1734) switch from `colvec` to `rowvec`.

## Layout summary

| Mode | Kernel A | Kernel B | Kernel D/C/PostAct view | Kernel result orientation | Major order for D/C/PostAct |
|---|---|---|---|---|---|
| `swap_ab=False` | original `A` `(L,M,K)` | original `B.mT` `(L,N,K)` | original `(L,M,N)` | `(M,N)` | `"n"` |
| `swap_ab=True` | original `B.mT` `(L,N,K)` | original `A` `(L,M,K)` | `.mT` view `(L,N,M)` | `(N,M)` | `"m"` |

The `"n"` vs `"m"` distinction comes from [get_major](../thirdparty/quack/quack/gemm_tvm_ffi_utils.py#L41): if `stride(1) == 1`, the second logical dimension is contiguous; otherwise the first logical dimension is contiguous.

## Lowest-level confirmation

The lower-level SM90 helper does the same conceptual transform when `swap_AB` is enabled:

- [sm90_utils.py](../thirdparty/quack/quack/sm90_utils.py#L102)
- [sm90_utils.py](../thirdparty/quack/quack/sm90_utils.py#L125)
- [sm90_utils.py](../thirdparty/quack/quack/sm90_utils.py#L141)

In particular, [partition_fragment_ABC](../thirdparty/quack/quack/sm90_utils.py#L133) allocates the accumulator with shape `(N, M)` instead of `(M, N)` when swapped, which matches the wrapper’s `.mT` output views.

## Short answer

- `B = B.mT` is always there because the kernel interface wants `B` as `(N, K)`, not `(K, N)`.
- `swap_ab=False` keeps the usual `(M, N)` output orientation and uses `rstd` as `colvec`.
- `swap_ab=True` computes the transposed GEMM orientation `(N, M)`, so the wrapper also passes transposed output views (`.mT`) and reclassifies `rstd` as `rowvec`.
- No math changes for the caller; only the internal orientation and physical layout presented to the kernel change.
