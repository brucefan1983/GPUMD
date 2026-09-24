"""Basic single-GPU integration tests for gpumd_replica.

Run directly with python3 tests_regression/replica/test_replica.py.
Set GPUMD_REPLICA to select the executable; see README.md in this directory.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BINARY = Path(os.environ.get("GPUMD_REPLICA", ROOT / "src/gpumd_replica")).resolve()
WORKDIR = os.environ.get("REPLICA_TEST_WORKDIR")
ENV = dict(os.environ)
# Keep a user-selected GPU (including UUIDs) while testing multiple streams on it.
ENV["CUDA_VISIBLE_DEVICES"] = ENV.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def metadata(directory, mode):
    state = {"bdp_rng": {}}
    for line in (directory / f"{mode}_restart.meta").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split(maxsplit=1)
        if key == "bdp_rng":
            replica, rng = value.split(maxsplit=1)
            replica = int(replica)
            if replica in state["bdp_rng"]:
                raise AssertionError(f"Repeated bdp_rng for replica {replica}")
            state["bdp_rng"][replica] = rng
        else:
            if key in state:
                raise AssertionError(f"Repeated restart key: {key}")
            state[key] = value
    if set(state["bdp_rng"]) != set(range(int(state["replicas"]))):
        raise AssertionError("Missing or unexpected replica RNG state")
    return state


def data_rows(path):
    return [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def events(directory):
    return data_rows(directory / "prd_events.out")


@unittest.skipUnless(
    __name__ == "__main__" or "GPUMD_REPLICA" in os.environ,
    "GPU tests: run this script directly or set GPUMD_REPLICA",
)
class ReplicaRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise RuntimeError(
                f"Replica executable not found: {BINARY}. "
                "Run make -C src or set GPUMD_REPLICA."
            )

    def setUp(self):
        if WORKDIR:
            parent = Path(WORKDIR).resolve()
            parent.mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName + "-", dir=parent))
            print(f"\nRetained test files: {self.root}", flush=True)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="gpumd-replica-test-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)

    def prepare(self, name, charge=False, free=False):
        directory = self.root / name
        directory.mkdir()
        if free:
            # Zero forces make the event clock predictable with weak BDP coupling.
            for filename in ("model.xyz", "nep.txt"):
                shutil.copyfile(FIXTURES / filename, directory / filename)
        else:
            shutil.copyfile(
                ROOT / "examples/gpumd_static_qnep/model.xyz", directory / "model.xyz"
            )
            potential = "qnep_train" if charge else "nep_train"
            shutil.copyfile(ROOT / f"examples/{potential}/nep.txt", directory / "nep.txt")
        return directory

    def assert_restart_xyz_equal(self, first, second):
        a, b = first.read_text().splitlines(), second.read_text().splitlines()
        self.assertEqual(a[:2], b[:2])
        self.assertEqual(len(a), int(a[0]) + 2)
        self.assertEqual(len(a), len(b))
        for row_a, row_b in zip(a[2:], b[2:]):
            tokens_a, tokens_b = row_a.split(), row_b.split()
            self.assertEqual(len(tokens_a), 8)
            self.assertEqual(len(tokens_b), 8)
            self.assertEqual(tokens_a[0], tokens_b[0])
            for x, y in zip(tokens_a[1:], tokens_b[1:]):
                # Text restarts and velocity unit conversions introduce roundoff.
                self.assertAlmostEqual(float(x), float(y), delta=1e-10)

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
            stderr=subprocess.STDOUT, timeout=120, env=ENV,
        )
        (directory / "run.log").write_text(result.stdout)
        if expected_error:
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn(expected_error, result.stdout)
        else:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def prd_options(self, correlation=0, distance=0.15, interval=4):
        return (
            f"replicas 3 event {interval} dephase 0 correlate {correlation} "
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
                       "replicas 2 event 2 dephase 0 correlate 5 distance 0.5 quench 0.001 2000")
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
                           "replicas 2 event 2 dephase 2 correlate 5 distance 0.5 quench 0.001 2000")
                self.run_md(directory, mode, 2, options)
                split = self.root / f"{mode}_split"
                shutil.copytree(directory, split)
                self.run_md(directory, mode, 8, options, resume=True)
                self.run_md(split, mode, 4, options, resume=True)
                self.run_md(split, mode, 4, options, resume=True)
                self.assertEqual(metadata(directory, mode), metadata(split, mode))
                for replica in range(2):
                    filename = f"{mode}_replica_{replica}_restart.xyz"
                    self.assert_restart_xyz_equal(directory / filename, split / filename)

    def test_legacy_restart_is_rejected(self):
        for mode in ("remd", "prd"):
            with self.subTest(mode=mode):
                directory = self.prepare(mode, free=True)
                options = ("replicas 2 exchange 2 temp 300 400" if mode == "remd" else
                           self.prd_options())
                self.run_md(directory, mode, 1, options)
                path = directory / f"{mode}_restart.meta"
                version = metadata(directory, mode)["format_version"]
                path.write_text(path.read_text().replace(
                    f"format_version {version}", f"format_version {int(version) - 1}"
                ))
                self.run_md(directory, mode, 1, options, resume=True,
                            expected_error=f"Unsupported {mode.upper()} restart format_version")

    def test_two_replica_exchange_rounds(self):
        directory = self.prepare("two_replica_rounds", free=True)
        options = "replicas 2 exchange 2 temp 300 400"
        self.run_md(directory, "remd", 6, options)
        self.run_md(directory, "remd", 4, options, resume=True)
        rows = data_rows(directory / "remd_exchange.out")
        self.assertEqual([int(row[0]) for row in rows], [2, 4, 6, 8, 10])

    def test_verbose_options_are_rejected(self):
        for mode in ("remd", "prd"):
            options = ("replicas 2 exchange 2 temp 300 400" if mode == "remd" else
                       self.prd_options())
            error = ("Unknown multi_replica option." if mode == "remd" else
                     "Unknown multi_replica prd option.")
            for option in ("verbose", "verbose_output"):
                with self.subTest(mode=mode, option=option):
                    directory = self.prepare(f"{mode}_{option}", free=True)
                    self.run_md(directory, mode, 1, f"{options} {option}",
                                expected_error=error)

    def test_dephase_input_validation(self):
        cases = [
            ("dephase 2 5", "accepts exactly one step count"),
            ("dephase -1", "one non-negative step count"),
            ("dephase", "one non-negative step count"),
            ("", "PRD requires replicas, event, dephase, correlate, and distance"),
        ]
        for i, (dephase, error) in enumerate(cases):
            with self.subTest(dephase=dephase):
                directory = self.prepare(f"invalid_dephase_{i}", free=True)
                options = f"replicas 3 event 1 {dephase} correlate 0 distance 1"
                self.run_md(directory, "prd", 1, options, expected_error=error)

    def test_dephase_retry_limit(self):
        directory = self.prepare("dephase_retry_limit", free=True)
        self.run_md(directory, "prd", 1,
                    "replicas 3 event 1 dephase 5 correlate 0 distance 1e-12 max_dephase_retries 2",
                    expected_error="PRD dephasing exceeded max_dephase_retries")

    def test_multi_replica_exchange_schedule(self):
        pairs = {
            4: [[(0, 1), (2, 3)], [(1, 2)]],
            8: [[(0, 1), (2, 3), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)]],
        }
        for replicas, rounds in pairs.items():
            with self.subTest(replicas=replicas):
                directory = self.prepare(f"remd_{replicas}", free=True)
                options = f"replicas {replicas} exchange 2 temp 300 400"
                self.run_md(directory, "remd", 4, options)
                self.run_md(directory, "remd", 4, options, resume=True)
                rows = data_rows(directory / "remd_exchange.out")
                expected = [(step, low, high)
                            for step, matching in zip((2, 4, 6, 8), rounds * 2)
                            for low, high in matching]
                self.assertEqual(
                    [(int(row[0]), int(row[3]), int(row[4])) for row in rows], expected
                )
                # Equal potential energies imply unit acceptance probability.
                label_to_replica = list(range(replicas))
                for row in rows:
                    low, high = int(row[3]), int(row[4])
                    self.assertEqual([int(row[1]), int(row[2])],
                                     [label_to_replica[low], label_to_replica[high]])
                    self.assertEqual(float(row[9]), 0.0)
                    self.assertEqual(float(row[10]), 1.0)
                    self.assertEqual(int(row[11]), 1)
                    label_to_replica[low], label_to_replica[high] = (
                        label_to_replica[high], label_to_replica[low]
                    )
                state = metadata(directory, "remd")
                self.assertEqual(list(map(int, state["temperature_to_replica"].split())),
                                 label_to_replica)
                self.assertEqual(list(map(int, state["replica_to_temperature"].split())),
                                 [label_to_replica.index(r) for r in range(replicas)])

    def test_restart_parameter_mismatch_is_rejected(self):
        cases = [
            ("remd", "replicas 2 exchange 2 temp 300 400", "exchange 2", "exchange 3",
             "REMD restart exchange_interval does not match"),
            ("prd", "replicas 3 event 2 dephase 5 correlate 0 distance 1",
             "dephase 5", "dephase 6", "PRD restart dephase_steps does not match"),
        ]
        for mode, options, old, new, error in cases:
            with self.subTest(mode=mode):
                directory = self.prepare(f"mismatch_{mode}", free=True)
                self.run_md(directory, mode, 2, options)
                restart = directory / f"{mode}_restart.meta"
                before = restart.read_bytes()
                self.run_md(directory, mode, 1, options.replace(old, new),
                            resume=True, expected_error=error)
                self.assertEqual(restart.read_bytes(), before)


if __name__ == "__main__":
    print(f"Replica executable: {BINARY}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={ENV['CUDA_VISIBLE_DEVICES']}", flush=True)
    unittest.main(verbosity=2)
