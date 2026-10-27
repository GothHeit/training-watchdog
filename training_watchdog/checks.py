"""Health checks on nvidia-smi data. Pure functions so they are easy to test."""

from dataclasses import dataclass

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

GPU_FIELDS = [
    "index", "name", "uuid", "driver_version", "memory.total", "memory.used",
    "utilization.gpu", "temperature.gpu",
    "pcie.link.gen.max", "pcie.link.gen.current",
    "pcie.link.width.max", "pcie.link.width.current",
    "ecc.mode.current",
    "ecc.errors.uncorrected.volatile.total", "ecc.errors.uncorrected.aggregate.total",
    "retired_pages.single_bit_ecc.count", "retired_pages.double_bit.count",
    "retired_pages.pending",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
]

# Ampere and newer replace page retirement with row remapping.
REMAP_FIELDS = [
    "gpu_uuid", "remapped_rows.uncorrectable", "remapped_rows.pending", "remapped_rows.failure",
]

# NVIDIA retires at most 64 pages; close to that the card is on its way out.
RETIRED_PAGES_LIMIT = 48
IDLE_TEMP_LIMIT = 85
BUSY_MEMORY_FRACTION = 0.10
BUSY_UTILIZATION = 20


@dataclass
class Check:
    name: str
    status: str
    detail: str
    gpu: str = None


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _yes(value):
    return value is not None and value.strip().lower() in ("yes", "1", "active", "true")


def check_gpu(g, remap=None):
    """Checks for one GPU row from GPU_FIELDS (plus its REMAP_FIELDS row, if any)."""
    gpu = g.get("index")
    out = []

    def add(name, status, detail):
        out.append(Check(name, status, detail, gpu))

    # ECC
    volatile = _int(g.get("ecc.errors.uncorrected.volatile.total"))
    aggregate = _int(g.get("ecc.errors.uncorrected.aggregate.total"))
    if volatile:
        add("ecc", FAIL, f"{volatile} uncorrectable ECC errors since last reboot")
    elif aggregate:
        add("ecc", WARN, f"{aggregate} uncorrectable ECC errors over the card's lifetime")
    elif (g.get("ecc.mode.current") or "").lower() == "disabled":
        add("ecc", WARN, "ECC is disabled, memory errors would go unnoticed")
    elif volatile is None and aggregate is None:
        add("ecc", SKIP, "ECC not reported (normal on consumer GPUs)")
    else:
        add("ecc", PASS, "no uncorrectable ECC errors")

    # Page retirement (pre-Ampere)
    sbe = _int(g.get("retired_pages.single_bit_ecc.count"))
    dbe = _int(g.get("retired_pages.double_bit.count"))
    if _yes(g.get("retired_pages.pending")):
        add("retired-pages", FAIL, "page retirement pending, GPU needs a reset")
    elif sbe is not None or dbe is not None:
        total = (sbe or 0) + (dbe or 0)
        if total >= RETIRED_PAGES_LIMIT:
            add("retired-pages", FAIL, f"{total} retired pages, near the 64 page limit")
        elif dbe:
            add("retired-pages", WARN, f"{dbe} pages retired after double-bit errors")
        else:
            add("retired-pages", PASS, f"{total} retired pages")

    # Row remapping (Ampere+)
    if remap:
        if _yes(remap.get("remapped_rows.failure")):
            add("row-remap", FAIL, "row remapping failed, memory is faulty")
        elif _yes(remap.get("remapped_rows.pending")):
            add("row-remap", FAIL, "row remap pending, GPU needs a reset")
        elif _int(remap.get("remapped_rows.uncorrectable")):
            add("row-remap", WARN,
                f"{_int(remap['remapped_rows.uncorrectable'])} rows remapped after uncorrectable errors")
        else:
            add("row-remap", PASS, "no pending or failed row remaps")

    # PCIe link
    w_max, w_cur = _int(g.get("pcie.link.width.max")), _int(g.get("pcie.link.width.current"))
    if w_cur is not None:
        if w_max and w_cur < w_max:
            add("pcie", WARN, f"PCIe link running x{w_cur}, card supports x{w_max}")
        elif w_cur < 8:
            add("pcie", WARN, f"PCIe link is only x{w_cur} (riser?), data loading may bottleneck")
        else:
            gen = g.get("pcie.link.gen.max")
            add("pcie", PASS, f"PCIe x{w_cur}" + (f" gen{gen}" if gen else ""))

    # Throttling while idle means power brake, a bad PSU or bad cooling.
    reasons = [
        label for field, label in (
            ("clocks_throttle_reasons.hw_slowdown", "hardware slowdown"),
            ("clocks_throttle_reasons.hw_thermal_slowdown", "hardware thermal"),
            ("clocks_throttle_reasons.sw_thermal_slowdown", "software thermal"),
        ) if _yes(g.get(field))
    ]
    temp = _int(g.get("temperature.gpu"))
    if reasons:
        add("throttle", WARN, "clocks throttled at idle: " + ", ".join(reasons))
    elif temp is not None and temp >= IDLE_TEMP_LIMIT:
        add("throttle", WARN, f"{temp}C at idle, will throttle under load")
    elif temp is not None:
        add("throttle", PASS, f"{temp}C, not throttled")

    # Someone else's process on "your" GPU
    total, used = _int(g.get("memory.total")), _int(g.get("memory.used"))
    util = _int(g.get("utilization.gpu"))
    if total and used is not None and (used / total > BUSY_MEMORY_FRACTION or (util or 0) > BUSY_UTILIZATION):
        add("busy", WARN, f"GPU already in use: {used} MiB used, {util}% utilization")
    elif total and used is not None:
        add("busy", PASS, f"idle ({used} MiB used)")

    return out


def verdict(checks):
    """'FAIL' if any check failed, else 'OK'."""
    return FAIL if any(c.status == FAIL for c in checks) else "OK"
