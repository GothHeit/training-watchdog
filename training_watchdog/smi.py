"""Thin wrapper around nvidia-smi. No NVML bindings needed."""

import shutil
import subprocess

_MISSING = {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]", "Unknown Error"}


class SmiError(RuntimeError):
    pass


def available():
    return shutil.which("nvidia-smi") is not None


def _run(args, timeout=30):
    try:
        return subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SmiError(f"nvidia-smi failed: {e}") from e


def _clean(value):
    value = value.strip()
    return None if value in _MISSING else value


def _lines(stdout):
    return [line for line in stdout.splitlines() if line.strip()]


def query(fields, kind="gpu"):
    """Return one dict per row for `nvidia-smi --query-<kind>=...`.

    Older or newer drivers reject fields they don't know, and one bad field
    fails the whole query, so on error we retry field by field and leave the
    unknown ones as None.
    """
    def ask(fs):
        return _run([f"--query-{kind}={','.join(fs)}", "--format=csv,noheader,nounits"])

    out = ask(fields)
    if out.returncode == 0:
        rows = []
        for line in _lines(out.stdout):
            values = [_clean(v) for v in line.split(",")]
            if len(values) == len(fields):
                rows.append(dict(zip(fields, values)))
        return rows

    columns = {}
    for f in fields:
        r = ask([f])
        columns[f] = [_clean(v) for v in _lines(r.stdout)] if r.returncode == 0 else None
    n = max((len(c) for c in columns.values() if c), default=0)
    if n == 0:
        raise SmiError((out.stderr or out.stdout).strip() or "nvidia-smi returned no GPUs")
    return [
        {f: (c[i] if c and i < len(c) else None) for f, c in columns.items()}
        for i in range(n)
    ]
