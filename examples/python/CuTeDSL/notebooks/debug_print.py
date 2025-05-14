from contextlib import contextmanager

import cutlass.cute as cute
import hunter
import torch


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
def print_tensor_torch(src):
    print(src)
    cute.print_tensor(src)

a = torch.randn(8, 5, dtype=torch.float)

with hunter_trace("cutedsl.trace.txt", stdlib=False):
    print_tensor_torch(a)
