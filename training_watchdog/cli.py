"""`training-watchdog preflight`: is this rented GPU safe to train on?"""

import argparse
import json
import sys
from dataclasses import asdict

from training_watchdog import __version__, bandwidth, burnin, smi
from training_watchdog.checks import FAIL, GPU_FIELDS, REMAP_FIELDS, WARN, Check, check_gpu, verdict


def gpu_checks():
    if not smi.available():
        return [], [Check("nvidia-smi", FAIL, "nvidia-smi not found, no NVIDIA driver on this machine")]
    try:
        gpus = smi.query(GPU_FIELDS)
    except smi.SmiError as e:
        return [], [Check("nvidia-smi", FAIL, str(e))]
    if not gpus:
        return [], [Check("nvidia-smi", FAIL, "no GPUs visible")]
    try:
        remaps = {r["gpu_uuid"]: r for r in smi.query(REMAP_FIELDS, kind="remapped-rows")}
    except smi.SmiError:
        remaps = {}  # pre-Ampere: no row remapping
    checks = []
    for g in gpus:
        checks += check_gpu(g, remaps.get(g.get("uuid")))
    return gpus, checks


def preflight(seconds, quick):
    gpus, checks = gpu_checks()
    if gpus and not quick:
        checks += burnin.run(seconds)
        checks += bandwidth.run()
    return gpus, checks


def print_report(gpus, checks):
    p = print

    p(f"Training Watchdog preflight {__version__}")
    for g in gpus:
        mem = int(float(g["memory.total"] or 0)) // 1024
        p(f"  GPU {g['index']}: {g['name']}, {mem} GB, driver {g['driver_version']}")
    p()
    for c in checks:
        where = f"[{c.gpu}]" if c.gpu is not None else ""
        p(f"  {c.status:<4}  {c.name + where:<20} {c.detail}")
    p()
    failed = [c for c in checks if c.status == FAIL]
    warned = sum(c.status == WARN for c in checks)
    if failed:
        p("VERDICT: DON'T TRAIN ON THIS MACHINE")
        for c in failed:
            p(f"  - {c.name}" + (f" (GPU {c.gpu})" if c.gpu is not None else "") + f": {c.detail}")
    else:
        p("VERDICT: OK" + (f" ({warned} warning{'s' * (warned != 1)})" if warned else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="training-watchdog")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    pf = sub.add_parser("preflight", help="check the GPUs before you start paying for a run")
    pf.add_argument("--seconds", type=int, default=30, help="burn-in length (default 30)")
    pf.add_argument("--quick", action="store_true", help="nvidia-smi checks only, no burn-in or bandwidth")
    pf.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    gpus, checks = preflight(args.seconds, args.quick)
    if args.json:
        json.dump({"verdict": verdict(checks), "gpus": gpus, "checks": [asdict(c) for c in checks]},
                  sys.stdout, indent=2)
        print()
    else:
        print_report(gpus, checks)
    return 1 if verdict(checks) == FAIL else 0
