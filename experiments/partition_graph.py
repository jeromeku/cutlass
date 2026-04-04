"""
partition_graph.py
==================
Automatically split a joint forward+backward FX graph into separate
fw_module and bw_module using AOT Autograd's partitioner.

The partitioner decides:
  - which nodes belong to the forward
  - which intermediate tensors must be saved (the "saved tensors")
  - which nodes belong to the backward

Two partitioners are available:
  default_partition              -- greedy, saves everything needed
  min_cut_rematerialization_partition -- smarter: may recompute cheap ops
                                         rather than save them, reducing memory

Run:
    .venv/bin/python experiments/partition_graph.py
"""
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import (
    default_partition,
    min_cut_rematerialization_partition,
)
from typing import Literal

# ── joint graph helper (same as vjp_graph.py) ────────────────────────────────

def get_joint_graph(fn, *example_inputs, mode: Literal["real", "fake", "symbolic"] = "real"):
    primals = [
        x.detach().requires_grad_(True)
        if isinstance(x, torch.Tensor) and x.is_floating_point() else x
        for x in example_inputs
    ]
    with torch.no_grad():
        ex = fn(*[p.detach() if isinstance(p, torch.Tensor) else p for p in primals])
    outs = ex if isinstance(ex, tuple) else (ex,)
    tangents = [torch.ones_like(o) for o in outs]
    n = len(primals)

    def joint(*args):
        primals, tangents = list(args[:n]), list(args[n:])
        outputs = fn(*primals)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        inputs = [x for x in primals if isinstance(x, torch.Tensor) and x.requires_grad]
        grads = torch.autograd.grad(outputs, inputs, grad_outputs=tangents,
                                    allow_unused=True, create_graph=False)
        return (*outputs, *grads)

    gm = make_fx(joint, tracing_mode=mode)(*primals, *tangents)
    return gm, primals, tangents


# ── partition ─────────────────────────────────────────────────────────────────

def partition(fn, *example_inputs, use_min_cut=False, mode: Literal["real", "fake", "symbolic"] = "real"):
    """
    Returns (fw_module, bw_module).

    fw_module(*primals) -> (*fwd_outputs, *saved_tensors)
    bw_module(*saved_tensors, *tangents) -> (*grad_primals)
    """
    joint_gm, primals, tangents = get_joint_graph(fn, *example_inputs, mode=mode)

    joint_inputs = primals + tangents          # all graph placeholders
    num_fwd_outputs = 1                        # how many outputs are "forward" outputs
                                               # (loss is the only fwd output here)

    part_fn = (min_cut_rematerialization_partition if use_min_cut
               else default_partition)
    fw_module, bw_module = part_fn(
        joint_gm,
        joint_inputs,
        num_fwd_outputs=num_fwd_outputs,
    )
    return fw_module, bw_module


# ── the same two-layer network from vjp_graph.py ─────────────────────────────

def two_layer(x, W1, b1, W2):
    h = x @ W1 + b1
    y = h @ W2
    return y


def print_module(label, gm):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {label}".center(72))
    print(sep)
    gm.graph.print_tabular()
    print(f"\n--- generated Python ---\n{gm.code}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H, O = 2, 4, 8, 3
    mode = "symbolic"

    x      = torch.randn(B, D)
    W1     = torch.randn(D, H)
    b1     = torch.randn(H)
    W2     = torch.randn(H, O)

    for label, use_min_cut in [("default_partition", False),
                                ("min_cut_rematerialization_partition", True)]:
        fw, bw = partition(two_layer, x, W1, b1, W2,
                           use_min_cut=use_min_cut, mode=mode)
        print_module(f"FORWARD  [{label}]", fw)
        print_module(f"BACKWARD [{label}]", bw)


    if mode == "real":
        # ── correctness check ─────────────────────────────────────────────────────
        # Use min_cut: fw receives (primals + tangents), bw receives only saved tensors.
        print("\n" + "=" * 72)
        print("  Correctness check  [min_cut]".center(72))
        print("=" * 72)

        fw, bw = partition(two_layer, x, W1, b1, W2, use_min_cut=True)
        out = two_layer(x, W1, b1, W2)
        tangent = torch.ones_like(out)

        # fw takes ALL joint inputs (primals + tangents) and returns (loss, *saved)
        fw_outs   = fw(x, W1, b1, W2, tangent)
        loss_sym  = fw_outs[0]
        saved     = fw_outs[1:]

        # bw takes only saved tensors (tangent already folded in by min_cut fw)
        grad_outs = bw(*saved)

        # Reference via autograd
        xr  = x.detach().requires_grad_(True)
        W1r = W1.detach().requires_grad_(True)
        b1r = b1.detach().requires_grad_(True)
        W2r = W2.detach().requires_grad_(True)
        loss_ref = two_layer(xr, W1r, b1r, W2r)
        torch.autograd.backward(loss_ref, tangent)

        print(f"\nloss  matches : {torch.allclose(loss_ref.detach(), loss_sym)}")
        # grad order matches diff_primals order: x, W1, b1, W2 (target not diff)
        for name, ref, sym in zip(
            ["grad_x", "grad_W1", "grad_b1", "grad_W2"],
            [xr.grad, W1r.grad, b1r.grad, W2r.grad],
            grad_outs,
        ):
            print(f"{name:8s} matches : {torch.allclose(ref, sym, atol=1e-5)}")
