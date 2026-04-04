"""
vjp_graph.py
============
Minimal demo: use make_fx + torch.autograd.grad to print the symbolic
backward (VJP) of an arbitrary function as an FX graph.

No flex_attention, no inductor, no private APIs.

Run:
    python experiments/vjp_graph.py
"""
import torch
from torch.fx.experimental.proxy_tensor import make_fx


def get_vjp_graph(fn, *example_inputs):
    """
    Trace fn and its VJP into a single FX GraphModule.

    The returned graph has placeholders:
        (*primals, *tangents)
    and returns:
        (*forward_outputs, *grad_primals)
    """
    primals = [
        x.detach().requires_grad_(True)
        if isinstance(x, torch.Tensor) and x.is_floating_point() else x
        for x in example_inputs
    ]

    with torch.no_grad():
        example_out = fn(*[p.detach() if isinstance(p, torch.Tensor) else p
                           for p in primals])

    outs = example_out if isinstance(example_out, tuple) else (example_out,)
    tangents = [torch.ones_like(o) for o in outs]
    n = len(primals)

    def joint(*args):
        p, t = list(args[:n]), list(args[n:])
        fwd = fn(*p)
        fwd_tuple = fwd if isinstance(fwd, tuple) else (fwd,)
        diff_p = [x for x in p if isinstance(x, torch.Tensor) and x.requires_grad]
        grads = torch.autograd.grad(fwd_tuple, diff_p, grad_outputs=t,
                                    allow_unused=True, create_graph=False)
        return (*fwd_tuple, *grads)

    return make_fx(joint)(*primals, *tangents)


# ── the function ──────────────────────────────────────────────────────────────
# A small two-layer network step with a non-trivial mix of ops:
#   h = gelu(x @ W1 + b1)
#   y = h @ W2
#   loss = (y - target).pow(2).mean()
#
# Primals:  x, W1, b1, W2, target
# Tangents: ones (scalar loss, so one tangent)
# Outputs:  loss, grad_x, grad_W1, grad_b1, grad_W2   (target assumed const)

def two_layer(x, W1, b1, W2, target):
    h = torch.nn.functional.gelu(x @ W1 + b1)
    y = h @ W2
    return (y - target).pow(2).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H, O = 2, 4, 8, 3

    x      = torch.randn(B, D)
    W1     = torch.randn(D, H)
    b1     = torch.randn(H)
    W2     = torch.randn(H, O)
    target = torch.randn(B, O)

    gm = get_vjp_graph(two_layer, x, W1, b1, W2, target)

    print("=== FX graph (forward + backward) ===\n")
    gm.graph.print_tabular()

    print("\n=== generated Python ===\n")
    print(gm.code)

    # Verify: symbolic grads match torch.autograd
    x.requires_grad_(True); W1.requires_grad_(True)
    b1.requires_grad_(True); W2.requires_grad_(True)
    loss = two_layer(x, W1, b1, W2, target)
    loss.backward()

    # graph returns: (loss, grad_x, grad_W1, grad_b1, grad_W2, grad_target)
    # target has requires_grad=False so its grad slot is neg(dL/dy) — ignored
    sym_loss, sym_gx, sym_gW1, sym_gb1, sym_gW2, *_ = gm(
        x.detach(), W1.detach(), b1.detach(), W2.detach(), target,
        torch.ones(())
    )

    for name, ref, sym in [("x",  x.grad,  sym_gx),
                            ("W1", W1.grad, sym_gW1),
                            ("b1", b1.grad, sym_gb1),
                            ("W2", W2.grad, sym_gW2)]:
        ok = torch.allclose(ref, sym, atol=1e-5)
        print(f"grad_{name} matches: {ok}")
