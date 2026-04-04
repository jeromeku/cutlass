import torch

copy_stream = torch.cuda.Stream()
default_stream = torch.cuda.current_stream()

pin_memory = False
non_blocking = False
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    host_tensor = torch.randn(4096, 5120, device="cpu", pin_memory=pin_memory)
    t = torch.randn(4096, 5120, device="cuda", dtype=torch.float32)
    w = torch.randn_like(t)

    with default_stream:
        out = t @ w.T

    with copy_stream:
        device_tensor = host_tensor.to("cuda", non_blocking=non_blocking)

prof.export_chrome_trace(f"overlap-{pin_memory}-{non_blocking}.trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total"))