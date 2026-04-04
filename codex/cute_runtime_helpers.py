from __future__ import annotations

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class _TensorWithStream:
    """Pass a CUDA stream through DLPack conversion without package-level imports."""

    def __init__(self, tensor, stream: int):
        self._tensor = tensor
        self._stream = -1 if stream == 0 else stream

    def __dlpack__(self, stream=None):  # noqa: ARG002
        return self._tensor.__dlpack__(stream=self._stream)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def convert_torch_tensor_to_cute_tensor(
    x,
    stride_order,
    leading_dim: int,
    alignment: int,
    divisibility: int,
    stream: int | None = None,
):
    """Convert a torch tensor into a CuTe runtime tensor with dynamic layout annotations."""
    tensor_input = _TensorWithStream(x, stream) if stream is not None else x
    return (
        from_dlpack(tensor_input, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, stride_order=stride_order, divisibility=divisibility)
    )
