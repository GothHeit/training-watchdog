"""Host<->GPU and on-GPU copy bandwidth, measured with PyTorch."""

from training_watchdog.checks import PASS, SKIP, WARN, Check

# Realistic pinned H2D: ~12 GB/s on PCIe 3 x16, ~25 on PCIe 4 x16, ~6 on x8.
MIN_HOST_GBPS = 6.0


def _time_copy(torch, dst, src, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dst.copy_(src, non_blocking=True)  # warm up
    start.record()
    for _ in range(repeats):
        dst.copy_(src, non_blocking=True)
    end.record()
    end.synchronize()
    return src.numel() * src.element_size() * repeats / (start.elapsed_time(end) / 1e3) / 1e9


def run(size_mb=256, repeats=10):
    try:
        import torch
    except ImportError:
        return [Check("bandwidth", SKIP, "install PyTorch to run the bandwidth test")]
    if not torch.cuda.is_available():
        return [Check("bandwidth", SKIP, "PyTorch can't use the GPU")]

    n = size_mb * 1024 * 1024
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    out = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            a = torch.empty(n, dtype=torch.uint8, device="cuda")
            b = torch.empty_like(a)
            h2d = _time_copy(torch, a, host, repeats)
            d2h = _time_copy(torch, host, a, repeats)
            d2d = _time_copy(torch, b, a, repeats)
            del a, b
        detail = f"host->gpu {h2d:.1f} GB/s, gpu->host {d2h:.1f} GB/s, on-gpu {d2d:.0f} GB/s"
        if min(h2d, d2h) < MIN_HOST_GBPS:
            out.append(Check("bandwidth", WARN, detail + " (slow PCIe, data loading may bottleneck)", f"cuda:{i}"))
        else:
            out.append(Check("bandwidth", PASS, detail, f"cuda:{i}"))
    return out
