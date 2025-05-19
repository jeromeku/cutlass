import functools
import math
from contextlib import contextmanager

import cutlass.cute as cute
import hunter
import torch
from cutlass.cute.runtime import from_dlpack


@contextmanager
def hunter_trace(output_path, force_colors=False, filename_alignment=100, enable=True, debug=False, **filters):
    if enable:
        tracer = hunter.Tracer()

        with open(output_path, 'w') as f:
            call_printer = hunter.CallPrinter(force_colors=force_colors, stream=f, filename_alignment=filename_alignment)
            actions = [call_printer]
            if debug:
                actions.append(hunter.Debugger())
            q = hunter.Q(actions=actions, **filters)
            print(q)
            
            tracer.trace(q)
            yield tracer
            tracer.stop()
    else:
        yield   

@cute.jit
def print_tensor_torch(t: cute.Tensor):
    cute.print_tensor(t)
    tiled_tensor = cute.tiled_divide(t, (2, 2))
    cute.printf(tiled_tensor)
    assert len(tiled_tensor.shape) == 3
    num_rows = tiled_tensor.shape[1]
    num_cols = tiled_tensor.shape[2]

    for i in range(num_rows):
        for j in range(num_cols):
            coord = (None, i, j)
            t_ = tiled_tensor[coord]
            cute.printf(f"({i},{j}):")
            cute.print_tensor(t_)
        cute.printf("\n")
shape = (4, 4)
numels = math.prod(shape)

a = torch.arange(numels, dtype=torch.int).view(shape)
print(a)
print_tensor_torch(from_dlpack(a))
# with hunter_trace("cutedsl.trace.txt", stdlib=False):
#     print_tensor_torch(a)
