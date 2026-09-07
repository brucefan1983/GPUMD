"""GPU integration regressions for the standalone replica executable.

Run with: python3 tests/replica/test_replica.py
Set GPUMD_REPLICA to test a different executable. Requires a visible CUDA GPU.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("GPUMD_REPLICA", ROOT / "src/gpumd_replica")).resolve()


def metadata(directory, mode):
    return dict(
        line.split(maxsplit=1)
        for line in (directory / f"{mode}_restart.meta").read_text().splitlines()
        if line and not line.startswith("#")
    )


def events(directory):
    return [
        line.split()
        for line in (directory / "prd_events.out").read_text().splitlines()
        if line and not line.startswith("#")
    ]


class ReplicaRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise RuntimeError("Build the executable with make -C src/main_replica first.")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpumd-replica-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def prepare(self, name, charge=False, free=False):
        directory = self.root / name
        directory.mkdir()
        if free:
            # A zero-energy NEP gives exactly ballistic motion when BDP's
            # coupling is sufficiently weak. This fixture tests event ordering
            # and clocks, not the physical basin criterion or NEP accuracy.
            (directory / "nep.txt").write_text(
                "nep4 1 C\ncutoff 4 4 10 10\nn_max 0 0\n"
                "basis_size 0 0\nl_max 1 0 0\nANN 1 0\n"
                + "0\n" * 7 + "1\n" * 2
            )
            (directory / "model.xyz").write_text(
                '4\npbc="T T T" Lattice="30 0 0 0 30 0 0 0 30" '
                "Properties=species:S:1:pos:R:3\n"
                "C 4 4 4\nC 10 4 4\nC 4 10 4\nC 4 4 10\n"
            )
        else:
            shutil.copyfile(
                ROOT / "examples/gpumd_static_qnep/model.xyz", directory / "model.xyz"
            )
            potential = "qnep_train" if charge else "nep_train"
            shutil.copyfile(ROOT / f"examples/{potential}/nep.txt", directory / "nep.txt")
        return directory

    def run_md(self, directory, mode, steps, options, kspace="", resume=False,
               coupling=100, expected_error=None):
        final_temperature = 400 if mode == "remd" else 300
        restart = f" resume {mode}" if resume else ""
        commands = (
            "potential nep.txt\ntime_step 1\n"
            f"ensemble nvt_bdp 300 {final_temperature} {coupling}\n"
            f"multi_replica {mode} {options}{restart}\n"
        )
        if kspace:
            commands += f"kspace {kspace}\n"
        (directory / "run.in").write_text(commands + f"run {steps}\n")
        result = subprocess.run(
            [str(BINARY)], cwd=directory, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120,
        )
        (directory / "run.log").write_text(result.stdout)
        if expected_error:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stdout)
        else:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def prd_options(self, correlation=0, distance=0.15, interval=4):
        return (
            f"replicas 3 event {interval} dephase 0 1 correlate {correlation} "
            f"distance {distance} quench 0.001 2000"
        )

    def test_prd_initial_correlation_and_resume(self):
        directory = self.prepare("initial", free=True)
        options = self.prd_options(correlation=5, distance=1)
        self.run_md(directory, "prd", 2, options, coupling=1e20)
        state = metadata(directory, "prd")
        self.assertEqual(state["phase"], "correlation")
        self.assertEqual(state["clock"], "2")
        self.assertEqual(state["correlation_quiet_steps"], "2")
        self.assertEqual(state["correlation_winner"], "0")
        self.run_md(directory, "prd", 2, options, resume=True, coupling=1e20)
        self.assertEqual(metadata(directory, "prd")["clock"], "4")
        # The fifth serial step finishes correlation. The sixth step runs
        # all three replicas, so the total physical clock must be eight.
        self.run_md(directory, "prd", 2, options, resume=True, coupling=1e20)
        state = metadata(directory, "prd")
        self.assertEqual(state["clock"], "8")
        self.assertEqual(state["phase"], "parallel")
        self.assertEqual(state["correlation_winner"], "-1")
        self.assertEqual(state["correlation_quiet_steps"], "0")

    def test_prd_initial_exits_are_counted_without_acceleration(self):
        directory = self.prepare("initial_exits", free=True)
        self.run_md(
            directory, "prd", 2, self.prd_options(correlation=5, distance=1e-8),
            coupling=1e20,
        )
        rows = events(directory)
        self.assertEqual(len(rows), 2)
        self.assertEqual([int(row[1]) for row in rows], [1, 2])
        self.assertTrue(all(row[4] == "1" and row[6] == "0" for row in rows))
        self.assertEqual(metadata(directory, "prd")["correlation_quiet_steps"], "0")

    def test_prd_clock_and_coincident_exit_order(self):
        # All tests exit at step 2 or later, so both complete rounds and the
        # winner's position in the final round contribute to the clock.
        cases = [
            ("middle", [0.06, 0.1, 0.06], 1, 1, 5),
            ("last", [0.06, 0.06, 0.1], 2, 1, 6),
            ("tie", [0.1, 0.1, 0.06], 0, 2, 4),
            ("no_exit", [0.01, 0.01, 0.01], None, 0, 6),
        ]
        for interval in (1, 4):
            for name, speeds, winner, coincident, increment in cases:
                with self.subTest(interval=interval, case=name):
                    directory = self.prepare(f"{name}_{interval}", free=True)
                    options = self.prd_options(interval=interval)
                    self.run_md(directory, "prd", 1, options, coupling=1e20)
                    self.assertEqual(events(directory), [])
                    clock_before = int(metadata(directory, "prd")["clock"])
                    reference = (directory / "prd_basin_restart.xyz").read_text().splitlines()[2:]
                    for replica, speed in enumerate(speeds):
                        path = directory / f"prd_replica_{replica}_restart.xyz"
                        header = path.read_text().splitlines()[:2]
                        # Equal and opposite velocities preserve zero COM
                        # momentum, with a known 0.15 A displacement crossing.
                        velocities = [speed, -speed, 0, 0]
                        path.write_text("\n".join(header + [
                            f"{atom} {velocity} 0 0"
                            for atom, velocity in zip(reference, velocities)
                        ]) + "\n")
                    self.run_md(directory, "prd", 2, options, resume=True, coupling=1e20)
                    self.assertEqual(int(metadata(directory, "prd")["clock"]),
                                     clock_before + increment)
                    rows = events(directory)
                    if winner is None:
                        self.assertEqual(rows, [])
                    else:
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(int(rows[0][6]), winner)
                        self.assertEqual(int(rows[0][5]), coincident)
                        self.assertEqual(int(rows[0][1]), clock_before + increment)

    def test_qnep_comments_and_restart_solver_validation(self):
        for mode in ("remd", "prd"):
            options = ("replicas 2 exchange 2 temp 300 400" if mode == "remd" else
                       "replicas 2 event 2 dephase 0 1 correlate 5 distance 0.5 quench 0.001 2000")
            for solver in ("pppm", "ewald"):
                with self.subTest(mode=mode, solver=solver):
                    directory = self.prepare(f"{mode}_{solver}", charge=True)
                    self.run_md(directory, mode, 1, options,
                                kspace=f"{solver} # electrostatics")
                    self.assertEqual(metadata(directory, mode)["kspace"], solver)
                    self.run_md(directory, mode, 1, options, kspace=solver, resume=True)
                    before = (directory / f"{mode}_restart.meta").read_bytes()
                    other = "ewald" if solver == "pppm" else "pppm"
                    self.run_md(directory, mode, 1, options, kspace=other, resume=True,
                                expected_error=f"{mode.upper()} restart kspace does not match")
                    self.assertEqual((directory / f"{mode}_restart.meta").read_bytes(), before)
                    if solver == "pppm":
                        # Omitting kspace must resolve to the same default.
                        self.run_md(directory, mode, 1, options, resume=True)

    def test_split_runs_preserve_state(self):
        for mode in ("remd", "prd"):
            with self.subTest(mode=mode):
                directory = self.prepare(mode)
                options = ("replicas 2 exchange 3 temp 300 400" if mode == "remd" else
                           "replicas 2 event 2 dephase 1 2 correlate 5 distance 0.5 quench 0.001 2000")
                self.run_md(directory, mode, 2, options)
                split = self.root / f"{mode}_split"
                shutil.copytree(directory, split)
                self.run_md(directory, mode, 8, options, resume=True)
                self.run_md(split, mode, 4, options, resume=True)
                self.run_md(split, mode, 4, options, resume=True)
                self.assertEqual(metadata(directory, mode), metadata(split, mode))
                for replica in range(2):
                    filename = f"{mode}_replica_{replica}_restart.xyz"
                    first = (directory / filename).read_text().splitlines()[2:]
                    second = (split / filename).read_text().splitlines()[2:]
                    for a, b in zip(first, second):
                        for x, y in zip(a.split()[1:], b.split()[1:]):
                            self.assertAlmostEqual(float(x), float(y), delta=1e-10)

    def test_legacy_restart_is_rejected(self):
        for mode in ("remd", "prd"):
            with self.subTest(mode=mode):
                directory = self.prepare(mode, free=True)
                options = ("replicas 2 exchange 2 temp 300 400" if mode == "remd" else
                           self.prd_options())
                self.run_md(directory, mode, 1, options)
                path = directory / f"{mode}_restart.meta"
                path.write_text(path.read_text().replace("format_version 4", "format_version 3"))
                self.run_md(directory, mode, 1, options, resume=True,
                            expected_error=f"Unsupported {mode.upper()} restart format_version")


if __name__ == "__main__":
    unittest.main(verbosity=2)
