"""Short burn-in: gpu-burn if installed, else DCGM, else a PyTorch matmul loop."""

import os
import re
import shutil
import statistics
import subprocess
import threading
import time

from training_watchdog import smi
from training_watchdog.checks import FAIL, PASS, SKIP, WARN, Check, _yes

THROTTLE_FIELDS = [
    "index", "temperature.gpu",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
]
HOT_LIMIT = 87
SLOW_SIBLING_RATIO = 0.7


class Sampler(threading.Thread):
    """Polls nvidia-smi during the burn-in to catch heat and throttling under load."""

    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.max_temp = {}
        self.throttled = set()
        self._done = threading.Event()

    def run(self):
        while not self._done.wait(self.interval):
            try:
                rows = smi.query(THROTTLE_FIELDS)
            except smi.SmiError:
                return
            for r in rows:
                i = r["index"]
                try:
                    self.max_temp[i] = max(self.max_temp.get(i, 0), int(r["temperature.gpu"]))
                except (TypeError, ValueError):
                    pass
                if any(_yes(r.get(f)) for f in THROTTLE_FIELDS[2:]):
                    self.throttled.add(i)

    def stop(self):
        self._done.set()
        self.join(timeout=self.interval + 30)

    def checks(self):
        out = []
        for i in sorted(set(self.max_temp) | self.throttled):
            temp = self.max_temp.get(i)
            if i in self.throttled:
                out.append(Check("load-throttle", WARN, f"clocks throttled under load (max {temp}C)", i))
            elif temp is not None and temp >= HOT_LIMIT:
                out.append(Check("load-throttle", WARN, f"reached {temp}C under load", i))
            elif temp is not None:
                out.append(Check("load-throttle", PASS, f"max {temp}C under load", i))
        return out


def run(seconds):
    sampler = Sampler()
    sampler.start()
    try:
        if shutil.which("gpu_burn"):
            checks = gpu_burn(seconds)
        else:
            checks = dcgm() if shutil.which("dcgmi") else []
            if not checks or checks[0].status == SKIP:
                checks = torch_burn(seconds)
    finally:
        sampler.stop()
    return checks + sampler.checks()


def gpu_burn(seconds):
    binary = os.path.realpath(shutil.which("gpu_burn"))
    try:
        # gpu_burn loads compare.ptx from its working directory.
        r = subprocess.run([binary, str(seconds)], capture_output=True, text=True,
                           cwd=os.path.dirname(binary), timeout=seconds + 120)
    except subprocess.TimeoutExpired:
        return [Check("burn-in", FAIL, f"gpu_burn hung for {seconds + 120}s")]
    results = re.findall(r"GPU (\d+): (OK|FAULTY)", r.stdout)
    if not results:
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
        return [Check("burn-in", WARN, f"gpu_burn exited {r.returncode} without a result: {tail[0]}")]
    return [
        Check("burn-in", PASS if state == "OK" else FAIL,
              f"gpu_burn {seconds}s: " + ("no compute errors" if state == "OK" else "wrong results (faulty GPU)"),
              gpu)
        for gpu, state in results
    ]


def dcgm():
    # Level 2 runs the short memory, PCIe and stress plugins, about 2 minutes.
    try:
        r = subprocess.run(["dcgmi", "diag", "-r", "2"], capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return [Check("burn-in", FAIL, "dcgmi diag hung for 15 min")]
    text = r.stdout + r.stderr
    if "Unable to" in text or "Error connecting" in text:
        return [Check("burn-in", SKIP, "dcgmi found but nv-hostengine is not running")]
    failed = [line.strip(" |") for line in text.splitlines() if "Fail" in line]
    if failed:
        return [Check("burn-in", FAIL, "dcgmi diag: " + "; ".join(failed[:3]))]
    if r.returncode != 0:
        return [Check("burn-in", WARN, f"dcgmi diag exited {r.returncode}")]
    return [Check("burn-in", PASS, "dcgmi diag -r 2 passed")]


def torch_burn(seconds, n=4096):
    try:
        import torch
    except ImportError:
        return [Check("burn-in", SKIP, "install gpu-burn, DCGM or PyTorch to run a burn-in")]
    if not torch.cuda.is_available():
        return [Check("burn-in", FAIL, f"PyTorch {torch.__version__} can't use the GPU "
                                       f"(built for CUDA {torch.version.cuda}), training would fail")]

    torch.backends.cuda.matmul.allow_tf32 = False
    devices = []
    for i in range(torch.cuda.device_count()):
        dev = torch.device("cuda", i)
        g = torch.Generator(device=dev).manual_seed(i)
        a = torch.randn(n, n, device=dev, generator=g)
        b = torch.randn(n, n, device=dev, generator=g)
        devices.append({"dev": dev, "a": a, "b": b, "ref": a @ b,
                        "bad": torch.zeros((), dtype=torch.long, device=dev), "iters": 0})
    for d in devices:
        torch.cuda.synchronize(d["dev"])

    # Same inputs must give the same product every time; anything else is a hardware fault.
    start = time.time()
    while time.time() - start < seconds:
        for d in devices:
            c = d["a"] @ d["b"]
            d["bad"] += (~torch.isclose(c, d["ref"], rtol=1e-3, atol=1e-3)).sum()
            d["iters"] += 1
    for d in devices:
        torch.cuda.synchronize(d["dev"])
    elapsed = time.time() - start

    tflops = [2 * n**3 * d["iters"] / elapsed / 1e12 for d in devices]
    median = statistics.median(tflops)
    out = []
    for i, (d, tf) in enumerate(zip(devices, tflops)):
        bad = int(d["bad"])
        if bad:
            out.append(Check("burn-in", FAIL, f"{bad} wrong values in {d['iters']} matmuls (faulty GPU)", f"cuda:{i}"))
        elif len(devices) > 1 and tf < SLOW_SIBLING_RATIO * median:
            out.append(Check("burn-in", WARN, f"{tf:.1f} TFLOPS fp32, others do {median:.1f}", f"cuda:{i}"))
        else:
            out.append(Check("burn-in", PASS, f"{seconds}s matmul, no errors, {tf:.1f} TFLOPS fp32", f"cuda:{i}"))
    return out
