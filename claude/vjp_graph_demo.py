"""
vjp_graph_demo.py
=================
Demonstrate how to re-use the same machinery PyTorch uses internally for
flex_attention's score_mod_bwd to print the symbolic VJP (backward) FX graph
for any arbitrary differentiable function.

The two key primitives are:
  1. create_joint(fn)    -- wraps fn into a joint (fwd+bwd) callable
  2. make_fx(joint_f)()  -- traces the joint under fake-tensor mode → FX GraphModule

Usage:
    python vjp_graph_demo.py
"""
import operator
import torch
from torch.fx.experimental.proxy_tensor import make_fx


# ---------------------------------------------------------------------------
# Core helper: create_joint
# ---------------------------------------------------------------------------

def create_joint(fn):
    """
    Given fn(*primals) -> Tensor (or tuple of Tensors), return a joint
    callable with signature:

        joint(primals: list[Tensor], tangents: list[Tensor])
            -> tuple[tuple[outputs], tuple[grads]]

    When traced with make_fx this captures the full forward + VJP as an
    FX GraphModule — exactly what flex_attention uses for score_mod_bwd.
    """
    def joint(primals, tangents):
        # --- forward ---
        outs = fn(*primals)
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)

        # Only differentiate primals that carry gradients
        diff_primals = [p for p in primals
                        if isinstance(p, torch.Tensor) and p.requires_grad]

        # --- backward (VJP) ---
        grads = torch.autograd.grad(
            outs,
            diff_primals,
            grad_outputs=tangents,
            allow_unused=True,
            create_graph=False,
        )
        return outs, grads

    return joint


# ---------------------------------------------------------------------------
# get_vjp_graph: trace the joint into an FX GraphModule
# ---------------------------------------------------------------------------

def get_vjp_graph(fn, *example_inputs):
    """
    Return an fx.GraphModule whose graph encodes:
      - the forward computation of fn
      - the VJP (reverse-mode gradient) of fn

    Parameters
    ----------
    fn            : callable(*Tensor) -> Tensor
    example_inputs: concrete Tensors (shapes / dtypes used for tracing)

    Returns
    -------
    fx.GraphModule
    """
    # Detach inputs; mark floating-point tensors as requiring grad so the
    # tracer can propagate gradients through them.
    primals = [
        x.detach().requires_grad_(True)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x
        for x in example_inputs
    ]

    # Compute an example output so we can build a matching tangent.
    with torch.no_grad():
        example_out = fn(*[p.detach() if isinstance(p, torch.Tensor) else p
                           for p in primals])
    if isinstance(example_out, (list, tuple)):
        tangents = [torch.ones_like(o) for o in example_out]
    else:
        tangents = [torch.ones_like(example_out)]

    joint = create_joint(fn)

    # Flatten into a single argument list for make_fx:
    #   (*primals, *tangents)
    n_primals = len(primals)

    def joint_flat(*args):
        p = list(args[:n_primals])
        t = list(args[n_primals:])
        outs, grads = joint(p, t)
        # Return everything as a flat tuple so the graph has visible outputs.
        return (*outs, *grads)

    graph_module = make_fx(joint_flat)(*primals, *tangents)
    return graph_module


# ---------------------------------------------------------------------------
# pretty printer
# ---------------------------------------------------------------------------

def print_vjp(fn, *example_inputs, label=None):
    gm = get_vjp_graph(fn, *example_inputs)
    sep = "=" * 70
    title = f"  VJP graph: {label or fn.__name__}  "
    print(f"\n{sep}")
    print(f"{title:^70}")
    print(sep)
    gm.graph.print_tabular()
    print()
    # Also print the raw Python source that make_fx generated
    print("--- generated Python (make_fx) ---")
    print(gm.code)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def softcap(s, cap=50.0):
    """score_mod used in Gemma: tanh softcapping."""
    return cap * torch.tanh(s / cap)


def alibi_bias(s, b, h, q_idx, kv_idx, slopes):
    """ALiBi positional bias (flex_attention score_mod style)."""
    return s - slopes[h] * (q_idx - kv_idx).abs().float()


def two_output(x):
    """Multiple outputs — tangents must match."""
    return x.sin(), x.cos()


def residual_block(x, w1, w2):
    """Small MLP-like block with two weight matrices."""
    h = torch.relu(x @ w1)
    return h @ w2 + x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    # 1. Simple element-wise score_mod (softcap)
    s = torch.randn(4, device=device)
    print_vjp(softcap, s, label="softcap (tanh)")

    # # 2. ALiBi — score_mod with a captured buffer (slopes)
    # #    Pass slopes as an extra primal so it appears in the grad output.
    # slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], device=device)
    # s2  = torch.randn((), device=device)
    # b   = torch.zeros((), dtype=torch.long, device=device)
    # h   = torch.zeros((), dtype=torch.long, device=device)
    # q_i = torch.tensor(3, dtype=torch.long, device=device)
    # k_i = torch.tensor(1, dtype=torch.long, device=device)

    # # Wrap so alibi_bias matches the signature (scalar inputs)
    # def alibi_wrapped(s, slopes):
    #     # fixed b/h/q_idx/kv_idx for tracing; in real use these are loop vars
    #     return s - slopes[0] * torch.abs((q_i - k_i).float())

    # print_vjp(alibi_wrapped, s2, slopes, label="alibi_bias (captured slopes)")

    # # 3. Multiple outputs
    # x3 = torch.randn(8, device=device)
    # print_vjp(two_output, x3, label="two_output (sin + cos)")

    # # 4. Small residual block — multiple primals, grad w.r.t. all of them
    # x4  = torch.randn(4, 8,  device=device)
    # w1  = torch.randn(8, 16, device=device)
    # w2  = torch.randn(16, 4, device=device)
    # print_vjp(residual_block, x4, w1, w2, label="residual_block (x, w1, w2)")

    # # 5. Demonstrate using the private AOT-autograd version directly
    # #    (same result, more faithful to what flex_attention actually calls)
    # print("\n" + "=" * 70)
    # print("  Using torch internal create_joint directly (AOT-autograd path)  ".center(70))
    # print("=" * 70)
    # try:
    #     from torch._functorch._aot_autograd.graph_capture_wrappers import (
    #         create_joint as _create_joint_internal,
    #     )
    #     from torch._functorch._aot_autograd.schemas import AOTConfig

    #     dummy_aot_config = AOTConfig(
    #         fw_compiler=None,
    #         bw_compiler=None,
    #         partition_fn=None,
    #         decompositions={},
    #         num_params_buffers=0,
    #         aot_id=0,
    #         keep_inference_input_mutations=False,
    #     )

    #     def fw_with_masks(s):
    #         out = softcap(s)
    #         return ((out,), (out.requires_grad,))

    #     joint_internal = _create_joint_internal(
    #         fw_with_masks, aot_config=dummy_aot_config
    #     )

    #     s_ex = torch.randn(4, requires_grad=True)
    #     grad_ex = torch.ones(4)

    #     def joint_flat_internal(s, grad):
    #         outs, grads = joint_internal([s], [grad])
    #         return (*outs[0], *grads)

    #     gm2 = make_fx(joint_flat_internal)(s_ex, grad_ex)
    #     print("\n--- internal AOT create_joint graph (softcap) ---")
    #     gm2.graph.print_tabular()
    #     print(gm2.code)

    # except Exception as e:
    #     print(f"[skipped — internal API unavailable: {e}]")
