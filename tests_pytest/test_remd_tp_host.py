"""Compile and call production host utilities directly; CUDA is not required."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RemdHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="remd-host-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = Path(cls.temp.name) / "remd-host"
        command = shlex.split(os.environ.get("CXX", "c++"))
        command += ["-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror"]
        command += shlex.split(os.environ.get("REPLICA_HOST_CXXFLAGS", ""))
        command += ["-I", str(ROOT / "src"),
                    str(Path(__file__).with_name("remd_tp_host_harness.cpp")), "-o", str(cls.binary)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)


def add_case(name):
    def test(self):
        result = subprocess.run([str(self.binary), name], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    setattr(RemdHost, "test_" + name, test)


for case in ("nvt_schedule", "two_replicas", "grid_schedule", "pressure_schedule", "sheared",
             "invalid_grid", "logweights", "pressure_swap", "temperature_swap", "reverse_swap",
             "probability", "scr_moments", "scr_response", "scr_scaling", "state_roundtrip",
             "bad_state", "failed_state_preserves_original"):
    add_case(case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
