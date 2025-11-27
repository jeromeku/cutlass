from torch._dynamo.eval_frame import _debug_get_cache_entry_list, innermost_fn
import torch


def f(x, y):
    return (x * y).relu()

opt_f = torch.compile(f)

x = torch.randn(4, 4)
y = torch.randn(4, 4)
opt_f(x, y)  # triggers compilation

entries = _debug_get_cache_entry_list(innermost_fn(opt_f))
print(entries)  # list of _CacheEntry objects from C land
