"""Single-GPU REMD-TP integration regressions; run this file explicitly.

Set REPLICA_SAMPLING=1 to include the longer ideal-gas distribution test.
"""
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import unittest

import test_replica as base

pytestmark = base.pytestmark

K_B = 8.617343e-5
PRESSURE_UNIT = 1.602177e2
TIME_UNIT = 1.018051e1


def metadata(directory):
    result = {}
    for line in (directory / 'remd_tp_restart.meta').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        tokens = line.split()
        indexed = tokens[0] in ('bdp_rng', 'scr_state', 'pair_statistics')
        key = tuple(tokens[:2] if indexed else tokens[:1])
        if key in result:
            raise AssertionError(f'duplicate metadata record: {key}')
        result[key] = tokens[2:] if indexed else tokens[1:]
    return result


def xyz_frames(path):
    with path.open() as stream:
        while line := stream.readline():
            n = int(line)
            header = dict(token.split('=', 1) for token in shlex.split(stream.readline()))
            atoms = [stream.readline().split() for _ in range(n)]
            yield header, atoms


def volume(header):
    h = list(map(float, header['Lattice'].split()))
    return h[0] * h[4] * h[8]


class RemdTPRegression(unittest.TestCase):
    setUp = base.ReplicaRegression.setUp
    prepare = base.ReplicaRegression.prepare

    @classmethod
    def setUpClass(cls):
        if not base.BINARY.is_file():
            raise RuntimeError(f'Replica executable not found: {base.BINARY}; build with make -C src gpumd')

    def prepare_orthogonal_pbte(self, name, repeat=2):
        directory = self.prepare(name)
        # The original PbTe fixture is triclinic and is deliberately unsupported
        # by REMD-TP. Use a 2x2x2 conventional rocksalt cell with the same NEP.
        lattice = 6.570392
        basis = [('Te', 0, 0, 0), ('Te', 0, .5, .5), ('Te', .5, 0, .5), ('Te', .5, .5, 0),
                 ('Pb', .5, 0, 0), ('Pb', 0, .5, 0), ('Pb', 0, 0, .5), ('Pb', .5, .5, .5)]
        atoms = [f'{symbol} {(i + x) * lattice:.12g} {(j + y) * lattice:.12g} {(k + z) * lattice:.12g}'
                 for i in range(repeat) for j in range(repeat) for k in range(repeat)
                 for symbol, x, y, z in basis]
        length = repeat * lattice
        (directory / 'model.xyz').write_text(
            f'{len(atoms)}\npbc="T T T" Lattice="{length} 0 0 0 {length} 0 0 0 {length}" '
            'Properties=species:S:1:pos:R:3\n' + '\n'.join(atoms) + '\n')
        return directory

    def run_tp(self, directory, steps, *, temp=(300, 400, 2), pressure=(0, 0, 1),
               exchange=10, interval=10, couple='xyz', modulus=2, tau_p=1000,
               tau_t=100, equilibrate=0, resume=False, grid=None, dump=0, error=None,
               timeout=120):
        t1, t2, nt = temp
        p1, p2, np = pressure
        p_options = f'pressure_grid {grid}' if grid else f'pressure {p1} {p2} {np}'
        commands = (
            'potential nep.txt\ntime_step 0.5\n'
            f'ensemble npt_scr {t1} {t2} {tau_t} {p1} {p2} {modulus} {tau_p} '
            f'couple {couple} pcouple_interval {interval}\n'
            f'multi_replica remd_tp replicas {nt * np} exchange {exchange} '
            f'temp {t1} {t2} {nt} {p_options}'
            + (f' equilibrate {equilibrate}' if equilibrate else '')
            + (' resume remd_tp' if resume else '') + '\n'
        )
        if dump:
            commands += f'dump_xyz -1 0 {dump} trajectory.xyz velocity\n'
        (directory / 'run.in').write_text(commands + f'run {steps}\n')
        result = subprocess.run([str(base.BINARY)], cwd=directory, env=base.ENV,
                                capture_output=True, text=True, timeout=timeout)
        (directory / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 1 if error else 0, result.stdout + result.stderr)
        if error:
            self.assertIn(error, result.stdout + result.stderr)
        return result

    def assert_cached_pressure(self, directory, replica):
        header, atoms = next(xyz_frames(directory / f'remd_tp_replica_{replica}_restart.xyz'))
        state = metadata(directory)[('scr_state', str(replica))]
        self.assertEqual(state[2], '1')  # Pending update at the checkpoint.
        kinetic = [sum(float(row[4]) * (float(row[5 + d]) * TIME_UNIT) ** 2
                       for row in atoms) for d in range(3)]
        expected_temperature = sum(kinetic) / ((3 * len(atoms) - 3) * K_B)
        self.assertAlmostEqual(float(state[3]), expected_temperature, delta=1e-8)
        for d in range(3):
            expected_pressure = kinetic[d] / volume(header)  # Zero virial fixture.
            self.assertAlmostEqual(float(state[5 + d]), expected_pressure,
                                   delta=max(1e-15, abs(expected_pressure) * 1e-10))

    def test_coincident_temperature_exchange_keeps_barostat(self):
        directory = self.prepare('coincident', free=True)
        self.run_tp(directory, 100)
        rows = base.data_rows(directory / 'remd_tp_exchange.out')
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row[-1] == '1' for row in rows))
        for replica in range(2):
            self.assert_cached_pressure(directory, replica)
            header, _ = next(xyz_frames(directory / f'remd_tp_replica_{replica}_restart.xyz'))
            self.assertNotEqual(volume(header), 27000)
        before = [next(xyz_frames(directory / f'remd_tp_replica_{r}_restart.xyz'))[0]
                  for r in range(2)]
        self.run_tp(directory, 1, resume=True)
        for r in range(2):
            header, _ = next(xyz_frames(directory / f'remd_tp_replica_{r}_restart.xyz'))
            self.assertNotEqual(volume(header), volume(before[r]))
            self.assertEqual(metadata(directory)[('scr_state', str(r))][2], '0')

    def test_pressure_is_measured_after_bdp_without_exchange(self):
        directory = self.prepare('bdp_pressure', free=True)
        self.run_tp(directory, 10, exchange=1000, tau_t=1)
        for replica in range(2):
            self.assert_cached_pressure(directory, replica)

    def test_exchange_between_barostat_steps_adds_no_update(self):
        directory = self.prepare('off_interval', free=True)
        self.run_tp(directory, 9, exchange=3)
        for r in range(2):
            header, _ = next(xyz_frames(directory / f'remd_tp_replica_{r}_restart.xyz'))
            self.assertEqual(volume(header), 27000)
            self.assertEqual(metadata(directory)[('scr_state', str(r))][2], '0')

    def test_coupled_directions(self):
        for dirs in ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'):
            with self.subTest(couple=dirs):
                directory = self.prepare(dirs, free=True)
                self.run_tp(directory, 31, couple=dirs)
                for r in range(2):
                    header, _ = next(xyz_frames(directory / f'remd_tp_replica_{r}_restart.xyz'))
                    h = list(map(float, header['Lattice'].split()))
                    for d, name in enumerate('xyz'):
                        if name in dirs:
                            self.assertNotEqual(h[4 * d], 30)
                        else:
                            self.assertEqual(h[4 * d], 30)

    def test_pressure_and_sheared_grid_exchange_rules(self):
        for sheared, nt, np in ((False, 1, 3), (False, 3, 3), (True, 3, 3)):
            with self.subTest(sheared=sheared, nt=nt):
                directory = self.prepare(f'grid_{nt}_{sheared}', free=True)
                grid = None
                if sheared:
                    grid = 'pressures.dat'
                    (directory / grid).write_text('0 .001 .002\n.0001 .0011 .0021\n.0002 .0012 .0022\n')
                options = dict(temp=(300, 450 if nt > 1 else 300, nt),
                               pressure=(0, .0022 if sheared else .002, np), grid=grid)
                self.run_tp(directory, 40, **options)
                rows = base.data_rows(directory / 'remd_tp_exchange.out')
                phases = [('T', 0), ('P', 0), ('T', 1), ('P', 1)] if nt > 1 else [('P', 0), ('P', 1)] * 2
                expected = []
                for step, (direction, parity) in zip((10, 20, 30, 40), phases):
                    if direction == 'T':
                        expected += [(step, direction, p * nt + t, p * nt + t + 1)
                                     for p in range(np) for t in range(parity, nt - 1, 2)]
                    else:
                        expected += [(step, direction, p * nt + t, (p + 1) * nt + t)
                                     for t in range(nt) for p in range(parity, np - 1, 2)]
                self.assertEqual([(int(r[0]), r[1], int(r[4]), int(r[5])) for r in rows], expected)
                mapping = list(range(nt * np))
                for row in rows:
                    i, j, a, b = map(int, row[2:6])
                    self.assertEqual([mapping[i], mapping[j]], [a, b])
                    ta, tb, pa, pb, ei, ej, vi, vj = map(float, row[6:14])
                    log_ratio = ((ei - ej) * (1 / ta - 1 / tb) / K_B +
                                 (pa / ta - pb / tb) * (vi - vj) / (K_B * PRESSURE_UNIT))
                    self.assertAlmostEqual(float(row[14]), log_ratio, delta=1e-7)
                    self.assertAlmostEqual(float(row[15]), math.exp(min(0, log_ratio)), delta=1e-8)
                    if row[-1] == '1':
                        mapping[i], mapping[j] = mapping[j], mapping[i]
                self.assertEqual(list(map(int, metadata(directory)[('replica_to_label',)])), mapping)

    def assert_checkpoint_valid(self, directory, steps, count):
        state = metadata(directory)
        self.assertEqual(int(state[('completed_steps',)][0]), steps)
        self.assertEqual(sorted(map(int, state[('replica_to_label',)])), list(range(count)))
        for r in range(count):
            header, atoms = next(xyz_frames(directory / f'remd_tp_replica_{r}_restart.xyz'))
            self.assertEqual(int(header['step']), steps)
            self.assertGreater(volume(header), 0)
            self.assertTrue(all(math.isfinite(float(x)) for row in atoms for x in row[1:]))

    def test_split_continuations_with_pending_and_equilibration_offset(self):
        for free in (False, True):
            for equilibration in (5, 7):
                with self.subTest(free=free, equilibration=equilibration):
                    name = f'continuous_{free}_{equilibration}'
                    directory = self.prepare(name, free=True) if free else self.prepare_orthogonal_pbte(name)
                    options = dict(temp=(300, 400, 2), pressure=(0, .001, 2),
                                   exchange=7, interval=5, modulus=100, dump=3)
                    self.run_tp(directory, 10, equilibrate=equilibration, **options)
                    split = self.root / f'split_{free}_{equilibration}'
                    shutil.copytree(directory, split)
                    self.run_tp(directory, 23, resume=True, **options)
                    self.run_tp(split, 5, resume=True, **options)
                    self.run_tp(split, 18, resume=True, **options)
                    self.assert_checkpoint_valid(directory, 33, 4)
                    self.assert_checkpoint_valid(split, 33, 4)
                    self.assertEqual(metadata(directory)[('exchange_count',)],
                                     metadata(split)[('exchange_count',)])
                    for r in range(4):
                        self.assertEqual([h['step'] for h, _ in xyz_frames(directory / f'trajectory_replica_{r}.xyz')],
                                         [h['step'] for h, _ in xyz_frames(split / f'trajectory_replica_{r}.xyz')])

    def test_missing_and_duplicate_restart_metadata(self):
        directory = self.prepare('metadata_validation', free=True)
        self.run_tp(directory, 10)
        path = directory / 'remd_tp_restart.meta'
        lines = [x for x in path.read_text().splitlines() if x and not x.startswith('#')]
        for index, line in enumerate(lines):
            for duplicate in (False, True):
                with self.subTest(key=line.split()[:2], duplicate=duplicate):
                    edited = lines + [line] if duplicate else lines[:index] + lines[index + 1:]
                    path.write_text('\n'.join(edited) + '\n')
                    before = path.read_bytes()
                    self.run_tp(directory, 1, resume=True, error='REMD-TP')
                    self.assertEqual(path.read_bytes(), before)

    def test_restart_version_and_parameters(self):
        directory = self.prepare('restart_parameters', free=True)
        self.run_tp(directory, 10)
        path = directory / 'remd_tp_restart.meta'
        original = path.read_text()
        version = metadata(directory)[('format_version',)][0]
        path.write_text(original.replace(f'format_version {version}', 'format_version 2'))
        self.run_tp(directory, 1, resume=True, error='Unsupported REMD-TP restart format_version')
        path.write_text(original)
        for options in (dict(exchange=11), dict(interval=5), dict(modulus=3), dict(tau_p=1100),
                        dict(tau_t=101), dict(temp=(300, 401, 2)), dict(couple='z')):
            with self.subTest(options=options):
                self.run_tp(directory, 1, resume=True, error='REMD-TP restart', **options)
        # Singular/plural messages differ for grid vs other parameter checks;
        # potential identity is also checked independently of the XYZ files.
        potential = directory / 'nep.txt'
        potential.write_text(potential.read_text() + '\n')
        self.run_tp(directory, 1, resume=True, error='potential files do not match')

    def test_restart_xyz_validation(self):
        directory = self.prepare('xyz_validation', free=True)
        self.run_tp(directory, 10, couple='z')
        path = directory / 'remd_tp_replica_0_restart.xyz'
        original = path.read_text()
        changes = [
            ('pbc="T T T"', 'pbc="F F T"'),
            ('Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3',
             'Properties=species:S:1:vel:R:3:mass:R:1:pos:R:3'),
            ('replica=0', 'replica=1'), ('step=10', 'step=9'),
        ]
        for old, new in changes:
            with self.subTest(change=new):
                path.write_text(original.replace(old, new))
                self.run_tp(directory, 1, resume=True, couple='z', error='REMD-TP')
        for lattice in ('0 0 0 0 30 0 0 0 30', '30 0 0 5 30 0 0 0 30',
                        '30 0 0 0 30 0 0 0 30 1'):
            with self.subTest(lattice=lattice):
                path.write_text(re.sub(r'Lattice="[^"]+"', f'Lattice="{lattice}"', original))
                self.run_tp(directory, 1, resume=True, couple='z', error='REMD-TP')
        path.write_text(original)
        model = directory / 'model.xyz'
        model.write_text(model.read_text().replace('pbc="T T T"', 'pbc="F F T"'))
        self.run_tp(directory, 1, resume=True, couple='z', error='pbc does not match')

    def test_reject_nonfinite_barostat_parameters(self):
        directory = self.prepare('nonfinite', free=True)
        for option in ('tau_t', 'modulus', 'tau_p'):
            for value in ('nan', 'inf', '-inf'):
                with self.subTest(option=option, value=value):
                    self.run_tp(directory, 1, error='npt_scr', **{option: value})

    def test_reject_qnep_and_nonperiodic_coupling(self):
        directory = self.prepare('charge', charge=True)
        self.run_tp(directory, 1, error='energy NEP models only')
        directory = self.prepare('nonperiodic', free=True)
        path = directory / 'model.xyz'
        path.write_text(path.read_text().replace('pbc="T T T"', 'pbc="F T T"'))
        self.run_tp(directory, 1, error='directions must be periodic')

    def test_species_subset_matches_plain_gpumd(self):
        directory = self.prepare_orthogonal_pbte('subset')
        lines = (directory / 'model.xyz').read_text().splitlines()
        atoms = [line + ' 0 0 0' for line in lines[2:] if line.startswith('Pb ')]
        header = lines[1].replace('pos:R:3', 'pos:R:3:vel:R:3')
        (directory / 'model.xyz').write_text(f'{len(atoms)}\n{header}\n' + '\n'.join(atoms) + '\n')
        plain = self.root / 'plain'
        shutil.copytree(directory, plain)
        self.run_tp(directory, 1)
        energy = float(base.data_rows(directory / 'remd_tp_thermo.out')[0][2])
        (plain / 'run.in').write_text(
            'potential nep.txt\ntime_step 0.0001\nensemble nve\ndump_thermo 1\nrun 1\n')
        result = subprocess.run([str(base.BINARY)], cwd=plain, env=base.ENV,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = float(base.data_rows(plain / 'thermo.out')[0][2])
        self.assertAlmostEqual(energy, expected, delta=max(1e-7, 1e-6 * abs(expected)))

    def test_deformed_large_box_energy_matches_plain_gpumd(self):
        # L=26.28 A exceeds the large-box threshold for the 8 A NEP cutoff.
        # The barostat must invalidate cached neighbors as the box changes.
        directory = self.prepare_orthogonal_pbte('large_box', repeat=4)
        self.run_tp(directory, 31, interval=1, modulus=20, tau_p=100)
        for replica in range(2):
            header, _ = next(xyz_frames(directory / f'remd_tp_replica_{replica}_restart.xyz'))
            self.assertNotEqual(volume(header), (4 * 6.570392) ** 3)
            plain = self.root / f'large_plain_{replica}'
            plain.mkdir()
            shutil.copyfile(directory / 'nep.txt', plain / 'nep.txt')
            shutil.copyfile(directory / f'remd_tp_replica_{replica}_restart.xyz', plain / 'model.xyz')
            (plain / 'run.in').write_text(
                'potential nep.txt\ntime_step 0\nensemble nve\ndump_thermo 1\nrun 1\n')
            result = subprocess.run([str(base.BINARY)], cwd=plain, env=base.ENV,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            expected = float(base.data_rows(plain / 'thermo.out')[0][2])
            self.assertAlmostEqual(float(header['potential_energy']), expected,
                                   delta=1e-8 * max(1, abs(expected)))

    @unittest.skipUnless(os.environ.get('REPLICA_SAMPLING') == '1', 'set REPLICA_SAMPLING=1 for long sampling test')
    def test_ideal_gas_volume_and_kinetic_distributions(self):
        directory = self.prepare('ideal_gas_sampling', free=True)
        n, temperature, length = 32, 300, 60
        atoms = [f'C {5 + 10 * x} {5 + 10 * y} {5 + 10 * z}'
                 for z in range(2) for y in range(4) for x in range(4)]
        (directory / 'model.xyz').write_text(
            f'{n}\npbc="T T T" Lattice="{length} 0 0 0 {length} 0 0 0 {length}" '
            'Properties=species:S:1:pos:R:3\n' + '\n'.join(atoms) + '\n')
        pressure = n * K_B * temperature / length ** 3 * PRESSURE_UNIT
        self.run_tp(directory, int(os.environ.get('REPLICA_SAMPLING_STEPS', '2000000')), temp=(temperature, temperature, 1),
                    pressure=(pressure, pressure * 1.2, 2), exchange=100, interval=1,
                    modulus=.002, tau_p=1000, tau_t=10, equilibrate=100000, dump=100,
                    timeout=int(os.environ.get('REPLICA_SAMPLING_TIMEOUT', '3600')))
        volumes, energies = [[], []], [[], []]
        for r in range(2):
            for header, rows in xyz_frames(directory / f'trajectory_replica_{r}.xyz'):
                label = int(header['condition_label'])
                volumes[label].append((int(header['step']), volume(header)))
                # Trajectory columns contain positions then velocities (no mass).
                kinetic = .5 * 12.011 * sum((float(x) * TIME_UNIT) ** 2 for row in rows for x in row[4:7])
                energies[label].append((int(header['step']), kinetic / (K_B * temperature)))
        # Use the mass actually written by this build rather than assuming a table value.
        _, restart_atoms = next(xyz_frames(directory / 'remd_tp_replica_0_restart.xyz'))
        mass_correction = float(restart_atoms[0][4]) / 12.011
        for label in range(2):
            energies[label] = [(step, value * mass_correction) for step, value in energies[label]]
            v = [value for _, value in sorted(volumes[label])]
            k = [value for _, value in sorted(energies[label])]
            expected_v = n * K_B * temperature * PRESSURE_UNIT / (pressure * (1 if label == 0 else 1.2))
            # With zero COM momentum, Pint=(N-1)kT/V and p(V) is Gamma(N,kT/P).
            self.assert_batch_moment(v, expected_v, 'volume mean')
            self.assert_batch_moment([(value - expected_v) ** 2 for value in v], expected_v ** 2 / n, 'volume variance')
            self.assert_batch_moment(k, (3 * n - 3) / 2, 'kinetic mean')
            self.assert_batch_moment([(value - (3 * n - 3) / 2) ** 2 for value in k], (3 * n - 3) / 2, 'kinetic variance')

    def assert_batch_moment(self, samples, expected, name):
        blocks = [statistics.mean(samples[i:i + 200]) for i in range(0, len(samples), 200)
                  if len(samples[i:i + 200]) == 200]
        self.assertGreaterEqual(len(blocks), 50)
        error = 6 * statistics.stdev(blocks) / math.sqrt(len(blocks)) + .01 * abs(expected)
        self.assertAlmostEqual(statistics.mean(samples), expected, delta=error, msg=name)


if __name__ == '__main__':
    print(f'Replica executable: {base.BINARY}', flush=True)
    unittest.main(verbosity=2)
