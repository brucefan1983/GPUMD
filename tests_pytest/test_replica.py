"""Basic single-GPU integration tests for replica dynamics (multi_replica in gpumd).

Run with pytest from tests_pytest/, or directly with python3 tests_pytest/test_replica.py.
Set GPUMD to select another gpumd executable; see "Replica dynamics" in README.md.
"""

import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "replica"
BINARY = Path(os.environ.get("GPUMD", ROOT / "src/gpumd")).resolve()
WORKDIR = os.environ.get("REPLICA_TEST_WORKDIR")
ENV = dict(os.environ)
# Keep a user-selected GPU (including UUIDs) while testing multiple streams on it.
ENV["CUDA_VISIBLE_DEVICES"] = ENV.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
# Batched integration is chosen automatically unless a test forces it.
ENV.pop("GPUMD_REPLICA_BATCH", None)


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


class ReplicaRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise RuntimeError(
                f"gpumd executable not found: {BINARY}. "
                "Run make -C src gpumd or set GPUMD."
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

    def prepare_large(self, name, charge=False, repeat=2):
        """repeat x repeat x repeat supercell of the PbTe example (2000 atoms for repeat 2,
        triclinic, thickness about 38 A) so that the NEP or qNEP model uses the large-box
        path, which batched integration requires."""
        directory = self.root / name
        directory.mkdir()
        lines = (ROOT / "examples/gpumd_static_qnep/model.xyz").read_text().splitlines()
        number = int(lines[0])
        lattice = [float(v) for v in lines[1].split('Lattice="')[1].split('"')[0].split()]
        a, b, c = lattice[0:3], lattice[3:6], lattice[6:9]
        atoms = []
        for i in range(repeat):
            for j in range(repeat):
                for k in range(repeat):
                    shift = [i * a[d] + j * b[d] + k * c[d] for d in range(3)]
                    for row in lines[2:2 + number]:
                        tokens = row.split()
                        atoms.append((tokens[0], [float(tokens[1 + d]) + shift[d] for d in range(3)]))
        header = " ".join(f"{repeat * v!r}" for v in lattice)
        with open(directory / "model.xyz", "w") as output:
            output.write(f"{len(atoms)}\n")
            output.write(f'pbc="T T T" Lattice="{header}" Properties=species:S:1:pos:R:3\n')
            for symbol, position in atoms:
                output.write(f"{symbol} {position[0]!r} {position[1]!r} {position[2]!r}\n")
        potential = "qnep_train" if charge else "nep_train"
        shutil.copyfile(ROOT / f"examples/{potential}/nep.txt", directory / "nep.txt")
        return directory

    def copy_restart(self, source, name, mode):
        directory = self.root / name
        directory.mkdir()
        for filename in ("model.xyz", "nep.txt"):
            shutil.copyfile(source / filename, directory / filename)
        for path in source.glob(f"{mode}_*restart*"):
            shutil.copyfile(path, directory / path.name)
        return directory

    def assert_files_identical(self, first, second, names):
        for name in names:
            self.assertEqual(
                (first / name).read_bytes(), (second / name).read_bytes(),
                f"{name} differs between {first.name} and {second.name}")

    def restart_files(self, directory, mode):
        return sorted(path.name for path in directory.glob(f"{mode}_*restart*"))

    def assert_restart_state(self, directory, mode, steps):
        state = metadata(directory, mode)
        step_key = "completed_steps" if mode == "remd" else "completed_dynamics_steps"
        self.assertEqual(int(state[step_key]), steps)
        replicas = int(state["replicas"])
        if mode == "remd":
            mapping = list(map(int, state["replica_to_temperature"].split()))
            inverse = list(map(int, state["temperature_to_replica"].split()))
            self.assertEqual(sorted(mapping), list(range(replicas)))
            self.assertEqual(sorted(inverse), list(range(replicas)))
            self.assertEqual([inverse[label] for label in mapping], list(range(replicas)))
            self.assertEqual(int(state["exchange_count"]), steps // int(state["exchange_interval"]))
            temperatures = list(map(float, state["temperatures"].split()))
        else:
            self.assertGreaterEqual(int(state["clock"]), steps)
            self.assertIn(state["phase"], ("dephase", "parallel", "correlation"))
        for replica in range(replicas):
            lines = (directory / f"{mode}_replica_{replica}_restart.xyz").read_text().splitlines()
            self.assertEqual(len(lines), int(lines[0]) + 2)
            fields = dict(field.split("=", 1) for field in shlex.split(lines[1]))
            self.assertEqual(fields["Properties"], "species:S:1:pos:R:3:mass:R:1:vel:R:3")
            self.assertEqual(fields["replica_mode"], mode)
            self.assertEqual(int(fields["replica"]), replica)
            if mode == "remd":
                self.assertEqual(int(fields["restart_step"]), steps)
                self.assertEqual(int(fields["temperature_label"]), mapping[replica])
                self.assertEqual(float(fields["target_temperature"]), temperatures[mapping[replica]])
                self.assertTrue(math.isfinite(float(fields["potential_energy"])))
            else:
                self.assertEqual(int(fields["dynamics_step"]), steps)
                self.assertEqual(int(fields["clock_step"]), int(state["clock"]))
            for row in lines[2:]:
                values = row.split()
                self.assertEqual(len(values), 8)
                self.assertTrue(all(math.isfinite(float(value)) for value in values[1:]))
                self.assertGreater(float(values[4]), 0)
        return state

    def run_md(self, directory, mode, steps, options, kspace="", resume=False,
               coupling=100, expected_error=None, batch=None):
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
        env = dict(ENV)
        if batch is not None:
            env["GPUMD_REPLICA_BATCH"] = str(batch)
        result = subprocess.run(
            [str(BINARY)], cwd=directory, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120, env=env,
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
        # The two-step correlation block (interval 4) is checked once, at its end:
        # one event, counted serially (clock 2, not accelerated by the 3 replicas).
        rows = events(directory)
        self.assertEqual(len(rows), 1)
        self.assertEqual([int(row[1]) for row in rows], [2])
        self.assertTrue(all(row[4] == "1" and row[6] == "0" for row in rows))
        self.assertEqual(metadata(directory, "prd")["correlation_quiet_steps"], "0")

    def test_prd_clock_and_coincident_exit_order(self):
        # All tests exit at step 2 or later, so both complete rounds and the
        # winner's position in the final round contribute to the clock.
        # Increments are for step-resolved checks (interval 1). With interval 4
        # the two-step run is a single block of s = 2 steps that is checked only
        # at its end, so an exit adds s * (winner + 1) and no exit adds 3 * s.
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
                        # Set known velocities in A/fs.
                        # Equal and opposite velocities preserve zero COM
                        # momentum, with a known 0.15 A displacement crossing.
                        velocities = [speed, -speed, 0, 0]
                        path.write_text("\n".join(header + [
                            f"{atom} {velocity} 0 0"
                            for atom, velocity in zip(reference, velocities)
                        ]) + "\n")
                    self.run_md(directory, "prd", 2, options, resume=True, coupling=1e20)
                    if interval > 1:
                        increment = 2 * (3 if winner is None else winner + 1)
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

    # Batched integration (GPUMD_REPLICA_BATCH=1) must reproduce the per-slot path bit for bit.
    BATCH_OPTIONS = {
        "remd": "replicas 4 exchange 10 temp 300 400",
        "prd": "replicas 3 event 5 dephase 10 correlate 5 distance 0.15 quench 0.01 3000",
    }

    def check_batched_matches_per_slot(self, mode):
        options = self.BATCH_OPTIONS[mode]
        start = self.prepare_large(f"{mode}_start")
        self.run_md(start, mode, 20, options, batch=0)
        runs = {}
        for batch in (0, 1):
            directory = self.copy_restart(start, f"{mode}_batch{batch}", mode)
            result = self.run_md(directory, mode, 40, options, resume=True, batch=batch)
            if batch == 1:
                self.assertIn("batched integration: on", result.stdout)
                self.assertIn("Batched integration:", result.stdout)
            runs[batch] = directory
        names = self.restart_files(runs[0], mode)
        self.assertEqual(names, self.restart_files(runs[1], mode))
        names += ["remd_exchange.out", "remd_temperature_status.out", "remd_thermo.out"] \
            if mode == "remd" else ["prd_events.out"]
        self.assert_files_identical(runs[0], runs[1], names)

    def test_batched_matches_per_slot_remd(self):
        self.check_batched_matches_per_slot("remd")

    def test_batched_matches_per_slot_prd(self):
        self.check_batched_matches_per_slot("prd")

    def test_batch_choice(self):
        # On when one GPU holds at least two replicas and either at most 5000 atoms per
        # replica or at most 30000 atoms in total.
        on = "on (one concatenated system per GPU)"
        cases = [
            ("on", 2, 4, None, on),
            ("on_small_replicas", 2, 16, None, on),  # 2000 atoms, 32000 in total
            ("on_total", 3, 4, None, on),  # 6750 atoms, 27000 in total
            ("off_atoms", 3, 5, None,
             "off (6750 atoms per replica > 5000 and 33750 atoms per GPU > 30000)"),
            ("off_forced", 2, 4, 0, "off (GPUMD_REPLICA_BATCH=0)"),
        ]
        for name, repeat, replicas, batch, message in cases:
            with self.subTest(name=name):
                directory = self.prepare_large(f"batch_choice_{name}", repeat=repeat)
                options = f"replicas {replicas} exchange 10 temp 300 400"
                result = self.run_md(directory, "remd", 10, options, batch=batch)
                self.assertIn("batched integration: " + message, result.stdout)
        directory = self.prepare_large("batch_keyword")
        self.run_md(directory, "remd", 10, "replicas 4 exchange 10 temp 300 400 batch 1",
                    expected_error="Unknown multi_replica option.")

    def test_remd_resume_preserves_schedule(self):
        # A checkpoint between exchanges must retain the mapping and global schedule.
        options = self.BATCH_OPTIONS["remd"]
        start = self.prepare_large("split_start")
        self.run_md(start, "remd", 20, options)
        initial = self.assert_restart_state(start, "remd", 20)
        for batch in (0, 1):
            with self.subTest(batch=batch):
                split = self.copy_restart(start, f"split_{batch}", "remd")
                self.run_md(split, "remd", 15, options, resume=True, batch=batch)
                self.assert_restart_state(split, "remd", 35)
                first = data_rows(split / "remd_temperature_status.out")[0]
                self.assertEqual(first, ["20"] + initial["replica_to_temperature"].split())
                self.assertEqual({int(row[0]) for row in data_rows(split / "remd_exchange.out")}, {30})
                self.run_md(split, "remd", 25, options, resume=True, batch=batch)
                self.assert_restart_state(split, "remd", 60)
                self.assertEqual({int(row[0]) for row in data_rows(split / "remd_exchange.out")},
                                 {30, 40, 50, 60})

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

    def assert_restart_xyz_close(self, first, second, tolerance):
        a, b = first.read_text().splitlines(), second.read_text().splitlines()
        self.assertEqual(a[0], b[0])
        self.assertEqual(len(a), len(b))
        # The comment line also stores the potential energy, which is compared numerically.
        comments = []
        for line in (a[1], b[1]):
            fields = line.split()
            energy = [f for f in fields if f.startswith("potential_energy=")]
            self.assertEqual(len(energy), 1)
            comments.append(([f for f in fields if f not in energy],
                             float(energy[0].split("=", 1)[1])))
        self.assertEqual(comments[0][0], comments[1][0])
        self.assertAlmostEqual(comments[0][1], comments[1][1],
                               delta=1e-9 * abs(comments[0][1]))
        largest = 0.0
        for row_a, row_b in zip(a[2:], b[2:]):
            tokens_a, tokens_b = row_a.split(), row_b.split()
            self.assertEqual(tokens_a[0], tokens_b[0])
            for x, y in zip(tokens_a[1:], tokens_b[1:]):
                largest = max(largest, abs(float(x) - float(y)))
        self.assertLess(largest, tolerance, f"{first.name}: largest difference {largest}")

    def test_qnep_large_box_concurrent_resume(self):
        # Two qNEP replicas on one GPU use the large-box path concurrently, each on its own
        # stream. Compare repeated runs from the same checkpoint to catch stream races.
        # PPPM charge spreading uses floating-point atomics, so allow roundoff there.
        # Split continuation checks the saved state, without requiring the same trajectory.
        options = "replicas 2 exchange 5 temp 300 400"
        for solver in ("ewald", "pppm"):
            with self.subTest(solver=solver):
                start = self.prepare_large(f"qnep_large_{solver}", charge=True)
                self.run_md(start, "remd", 5, options, kspace=solver)
                runs = []
                for name in ("whole", "again"):
                    whole = self.copy_restart(start, f"qnep_large_{solver}_{name}", "remd")
                    self.run_md(whole, "remd", 10, options, kspace=solver, resume=True)
                    self.assert_restart_state(whole, "remd", 15)
                    runs.append(whole)
                split = self.copy_restart(start, f"qnep_large_{solver}_split", "remd")
                self.run_md(split, "remd", 4, options, kspace=solver, resume=True)
                self.assert_restart_state(split, "remd", 9)
                self.run_md(split, "remd", 6, options, kspace=solver, resume=True)
                self.assert_restart_state(split, "remd", 15)
                names = self.restart_files(runs[0], "remd")
                if solver == "ewald":
                    self.assert_files_identical(runs[0], runs[1], names + ["remd_exchange.out"])
                else:
                    for name in names:
                        if name.endswith(".xyz"):
                            self.assert_restart_xyz_close(
                                runs[0] / name, runs[1] / name, tolerance=1e-6)
                    accepted = [[row[-1] for row in data_rows(run / "remd_exchange.out")]
                                for run in runs]
                    self.assertEqual(accepted[0], accepted[1])

    def test_qnep_matches_plain_gpumd(self):
        # Replica dynamics has its own copy of qNEP, Ewald and PPPM. On the small-box and
        # the large-box path, its initial potential energy equals that of an ordinary gpumd
        # run (PPPM charge spreading uses floating-point atomics, hence the tolerance).
        for solver in ("ewald", "pppm"):
            for size in ("small", "large"):
                with self.subTest(solver=solver, size=size):
                    name = f"qnep_plain_{solver}_{size}"
                    if size == "large":
                        directory = self.prepare_large(name, charge=True)
                    else:
                        directory = self.prepare(name, charge=True)
                    plain = self.root / f"{name}_plain"
                    shutil.copytree(directory, plain)

                    self.run_md(directory, "remd", 1, "replicas 2 exchange 1 temp 300 400",
                                kspace=solver)
                    row = data_rows(directory / "remd_thermo.out")[0]
                    replica_energies = [float(row[2]), float(row[5])]

                    # A zero time step keeps the positions: the energy is the initial one.
                    (plain / "run.in").write_text(
                        f"potential nep.txt\nkspace {solver}\nvelocity 300\nensemble nve\n"
                        "time_step 0\ndump_thermo 1\nrun 1\n")
                    result = subprocess.run(
                        [str(BINARY)], cwd=plain, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=120, env=ENV,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                    plain_energy = float(data_rows(plain / "thermo.out")[0][2])
                    for energy in replica_energies:
                        self.assertAlmostEqual(
                            energy, plain_energy, delta=1e-6 * abs(plain_energy))

    def test_species_subset_matches_plain_gpumd(self):
        # For a model.xyz with only some species of a multi-element NEP, gpumd evaluates a
        # compact copy of the model with renumbered types. A replica run must use the same
        # model: its initial potential energy equals that of an ordinary gpumd run.
        directory = self.prepare("subset")
        lines = (directory / "model.xyz").read_text().splitlines()
        number = int(lines[0])
        lattice = lines[1].split('Lattice="')[1].split('"')[0]
        rows = [row.split() for row in lines[2:2 + number] if row.split()[0] == "Pb"]
        with open(directory / "model.xyz", "w") as output:
            output.write(f"{len(rows)}\n")
            output.write(f'pbc="T T T" Lattice="{lattice}" Properties=species:S:1:pos:R:3:vel:R:3\n')
            for tokens in rows:
                output.write(" ".join(tokens[:4]) + " 0 0 0\n")
        plain = self.root / "subset_plain"
        shutil.copytree(directory, plain)

        result = self.run_md(directory, "remd", 1, "replicas 2 exchange 1 temp 300 400")
        self.assertIn("type 0 = Pb", result.stdout)
        replica_energy = float(data_rows(directory / "remd_thermo.out")[0][2])

        # Zero velocities and a tiny time step: the energy after one step is the initial one.
        (plain / "run.in").write_text(
            "potential nep.txt\ntime_step 0.0001\nensemble nve\ndump_thermo 1\nrun 1\n")
        result = subprocess.run(
            [str(BINARY)], cwd=plain, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120, env=ENV,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        plain_energy = float(data_rows(plain / "thermo.out")[0][2])
        self.assertAlmostEqual(replica_energy, plain_energy, delta=1e-6 * abs(plain_energy))

    def test_pppm_spacing_restart_validation(self):
        for mode in ("remd", "prd"):
            options = ("replicas 2 exchange 2 temp 300 400" if mode == "remd" else
                       "replicas 2 event 2 dephase 0 correlate 5 distance 0.5 quench 0.001 2000")
            with self.subTest(mode=mode):
                directory = self.prepare(f"{mode}_spacing", charge=True)
                self.run_md(directory, mode, 1, options, kspace="pppm 0.5")
                self.assertEqual(metadata(directory, mode)["kspace"], "pppm 0.5")
                self.run_md(directory, mode, 1, options, kspace="pppm 0.5", resume=True)
                before = (directory / f"{mode}_restart.meta").read_bytes()
                for kspace in ("pppm 1.0", "pppm"):
                    self.run_md(directory, mode, 1, options, kspace=kspace, resume=True,
                                expected_error=f"{mode.upper()} restart kspace does not match")
                self.assertEqual((directory / f"{mode}_restart.meta").read_bytes(), before)

    def test_restart_velocity_layouts(self):
        for mode in ("remd", "prd"):
            start = self.prepare(f"{mode}_velocity_start", free=True)
            options = ("replicas 2 exchange 10 temp 300 400" if mode == "remd" else
                       "replicas 2 event 2 dephase 0 correlate 0 distance 1")
            self.run_md(start, mode, 1, options, coupling=1e20)
            self.assert_restart_state(start, mode, 1)
            for natural in (False, True):
                with self.subTest(mode=mode, natural_velocity=natural):
                    directory = self.copy_restart(start, f"{mode}_velocity_{natural}", mode)
                    positions = []
                    for replica in range(2):
                        path = directory / f"{mode}_replica_{replica}_restart.xyz"
                        lines = path.read_text().splitlines()
                        positions.append([float(row.split()[1]) for row in lines[2:]])
                        if natural:
                            lines[1] = lines[1].replace("vel:R:3", "vel:R:3:vel_natural:R:3")
                        for atom, speed in enumerate((0.01, -0.01, 0, 0)):
                            fields = lines[atom + 2].split()[:5]
                            if natural:
                                # The displayed vel is deliberately zero: older files must
                                # use vel_natural. 0.1018051 internal units = 0.01 A/fs.
                                fields += ["0", "0", "0", repr(speed * 10.18051), "0", "0"]
                            else:
                                fields += [repr(speed), "0", "0"]
                            lines[atom + 2] = " ".join(fields)
                        path.write_text("\n".join(lines) + "\n")
                    self.run_md(directory, mode, 1, options, resume=True, coupling=1e20)
                    self.assert_restart_state(directory, mode, 2)
                    for replica in range(2):
                        rows = (directory / f"{mode}_replica_{replica}_restart.xyz").read_text().splitlines()[2:]
                        for atom, speed in enumerate((0.01, -0.01, 0, 0)):
                            self.assertAlmostEqual(float(rows[atom].split()[1]),
                                                   positions[replica][atom] + speed, delta=1e-12)

    def test_incompatible_restart_version_is_rejected(self):
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
