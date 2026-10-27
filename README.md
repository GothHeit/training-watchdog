# Training Watchdog

Stop paying for GPUs that aren't training.

Rented a GPU on RunPod, Vast.ai or Lambda? Run `preflight` before you start the meter on a long job.
It catches broken memory, degraded PCIe links, throttling and GPUs that compute wrong answers, and
tells you plainly: **OK** or **DON'T TRAIN ON THIS MACHINE**.

No W&B account, no provider lock-in, no runtime dependencies.

## Install

```bash
pip install git+https://github.com/GothHeit/training-watchdog
```

## Use

```bash
training-watchdog preflight              # ~1 min: nvidia-smi checks, 30s burn-in, bandwidth
training-watchdog preflight --quick      # nvidia-smi checks only, a few seconds
training-watchdog preflight --json       # for scripts

# Only start training if the machine is healthy (exit code 1 = don't train)
training-watchdog preflight && python train.py
```

`twd` is a short alias for `training-watchdog`.

## What it checks

| Check | Fails the machine when | Warns when |
|---|---|---|
| ECC | uncorrectable errors since boot | lifetime uncorrectable errors, ECC off |
| Retired pages / row remap | retirement or remap pending, remap failed, near page limit | pages retired after double-bit errors |
| PCIe | | link narrower than the card supports, or below x8 |
| Throttle | | clocks throttled at idle or under load, very hot |
| Busy | | another process already holds memory or compute |
| Burn-in | wrong matmul results, gpu-burn/DCGM report a fault | one GPU much slower than its siblings |
| Bandwidth | | host↔GPU copies under 6 GB/s |

Burn-in uses [gpu-burn](https://github.com/wilicc/gpu-burn) if `gpu_burn` is on your PATH, then
[DCGM](https://github.com/NVIDIA/DCGM) (`dcgmi diag -r 2`), then a PyTorch matmul loop.
The bandwidth test needs PyTorch. Without them those checks are skipped, not failed.
