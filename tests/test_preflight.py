import io
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from training_watchdog import checks, cli, smi
from training_watchdog.checks import FAIL, PASS, WARN

HEALTHY = {
    "index": "0", "name": "NVIDIA A100-SXM4-80GB", "uuid": "GPU-abc", "driver_version": "550.54",
    "memory.total": "81920", "memory.used": "4", "utilization.gpu": "0", "temperature.gpu": "34",
    "pcie.link.gen.max": "4", "pcie.link.gen.current": "1",
    "pcie.link.width.max": "16", "pcie.link.width.current": "16",
    "ecc.mode.current": "Enabled",
    "ecc.errors.uncorrected.volatile.total": "0", "ecc.errors.uncorrected.aggregate.total": "0",
    "retired_pages.single_bit_ecc.count": "[N/A]", "retired_pages.double_bit.count": "[N/A]",
    "retired_pages.pending": "[N/A]",
    "clocks_throttle_reasons.hw_slowdown": "Not Active",
    "clocks_throttle_reasons.hw_thermal_slowdown": "Not Active",
    "clocks_throttle_reasons.sw_thermal_slowdown": "Not Active",
}
REMAP = {"gpu_uuid": "GPU-abc", "remapped_rows.uncorrectable": "0",
         "remapped_rows.pending": "No", "remapped_rows.failure": "No"}


def statuses(g, remap=None):
    clean = {k: smi._clean(v) for k, v in g.items()}
    return {c.name: c.status for c in checks.check_gpu(clean, remap)}


class CheckGpuTest(unittest.TestCase):
    def test_healthy(self):
        s = statuses(HEALTHY, REMAP)
        self.assertEqual(set(s.values()), {PASS})
        self.assertNotIn("retired-pages", s)

    def test_volatile_ecc_fails(self):
        s = statuses({**HEALTHY, "ecc.errors.uncorrected.volatile.total": "3"})
        self.assertEqual(s["ecc"], FAIL)

    def test_lifetime_ecc_warns(self):
        s = statuses({**HEALTHY, "ecc.errors.uncorrected.aggregate.total": "1"})
        self.assertEqual(s["ecc"], WARN)

    def test_pending_remap_fails(self):
        s = statuses(HEALTHY, {**REMAP, "remapped_rows.pending": "Yes"})
        self.assertEqual(s["row-remap"], FAIL)

    def test_retired_pages(self):
        base = {**HEALTHY, "retired_pages.single_bit_ecc.count": "2", "retired_pages.double_bit.count": "0",
                "retired_pages.pending": "No"}
        self.assertEqual(statuses(base)["retired-pages"], PASS)
        self.assertEqual(statuses({**base, "retired_pages.pending": "Yes"})["retired-pages"], FAIL)
        self.assertEqual(statuses({**base, "retired_pages.single_bit_ecc.count": "50"})["retired-pages"], FAIL)

    def test_degraded_pcie_warns(self):
        self.assertEqual(statuses({**HEALTHY, "pcie.link.width.current": "4"})["pcie"], WARN)

    def test_busy_and_throttled_warn(self):
        s = statuses({**HEALTHY, "memory.used": "40000", "clocks_throttle_reasons.hw_slowdown": "Active"})
        self.assertEqual(s["busy"], WARN)
        self.assertEqual(s["throttle"], WARN)

    def test_consumer_gpu_skips_ecc(self):
        g = {**HEALTHY, "ecc.mode.current": "[N/A]", "ecc.errors.uncorrected.volatile.total": "[N/A]",
             "ecc.errors.uncorrected.aggregate.total": "[N/A]"}
        self.assertEqual(statuses(g)["ecc"], checks.SKIP)


FAKE_SMI = textwrap.dedent("""\
    #!{python}
    import json, sys
    data = json.load(open({data!r}))
    arg = next(a for a in sys.argv[1:] if a.startswith("--query-"))
    kind, fields = arg[len("--query-"):].split("=")
    for f in fields.split(","):
        if f in data["unknown"]:
            print(f"Field \\"{{f}}\\" is not a valid field to query.", file=sys.stderr)
            sys.exit(2)
    for row in data[kind]:
        print(", ".join(row[f] for f in fields.split(",")))
""")


class FakeSmiTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patcher = mock.patch.dict(os.environ, {"PATH": self.dir.name + os.pathsep + os.environ["PATH"]})
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake(self, gpus, remaps=(), unknown=()):
        data = os.path.join(self.dir.name, "data.json")
        with open(data, "w") as f:
            json.dump({"gpu": gpus, "remapped-rows": list(remaps), "unknown": list(unknown)}, f)
        path = os.path.join(self.dir.name, "nvidia-smi")
        with open(path, "w") as f:
            f.write(FAKE_SMI.format(python=sys.executable, data=data))
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)

    def run_cli(self, *args):
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            code = cli.main(["preflight", "--quick", *args])
        return code, out.getvalue()

    def test_ok(self):
        self.fake([HEALTHY], [REMAP])
        code, out = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("VERDICT: OK", out)

    def test_dont_train(self):
        self.fake([HEALTHY, {**HEALTHY, "index": "1", "uuid": "GPU-def"}],
                  [REMAP, {**REMAP, "gpu_uuid": "GPU-def", "remapped_rows.failure": "Yes"}])
        code, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("DON'T TRAIN ON THIS MACHINE", out)
        self.assertIn("row-remap (GPU 1)", out)

    def test_unknown_field_falls_back(self):
        self.fake([HEALTHY], unknown=["clocks_throttle_reasons.hw_slowdown"])
        rows = smi.query(checks.GPU_FIELDS)
        self.assertIsNone(rows[0]["clocks_throttle_reasons.hw_slowdown"])
        self.assertEqual(rows[0]["name"], "NVIDIA A100-SXM4-80GB")

    def test_json(self):
        self.fake([HEALTHY], [REMAP])
        code, out = self.run_cli("--json")
        self.assertEqual(json.loads(out)["verdict"], "OK")

    def test_no_driver(self):
        with mock.patch.object(smi, "available", return_value=False):
            code, out = self.run_cli()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
