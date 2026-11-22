"""Single-GPU NPT extension of REMD: integration regressions; run this file explicitly.

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
    for line in (directory / 'remd_restart.meta').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        tokens = line.split()
        indexed = tokens[0] in ('bdp_rng', 'scr_state')
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


class RemdNPTRegression(unittest.TestCase):
    setUp = base.ReplicaRegression.setUp
    prepare = base.ReplicaRegression.prepare

    @classmethod
    def setUpClass(cls):
        if not base.BINARY.is_file():
            raise RuntimeError(f'Replica executable not found: {base.BINARY}; build with make -C src gpumd')

    def prepare_orthogonal_pbte(self, name, repeat=2, strain=1):
        directory = self.prepare(name)
        # The original PbTe fixture is triclinic and is deliberately unsupported
        # by NPT REMD. Use a 2x2x2 conventional rocksalt cell with the same NEP.
        lattice = 6.570392 * strain
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

    def run_npt(self, directory, steps, *, temp=(300, 400, 2), pressure=(0, 0, 1),
               exchange=10, interval=10, modulus=2, tau_p=1000, tau_t=100,
               equilibrate=0, resume=False, grid=None, ladder=True, dump=0, error=None,
               timeout=120, dt=0.5, properties='velocity'):
        t1, t2, nt = temp
        p1, p2, np = pressure
        # The ensemble uses the isotropic npt_scr syntax of ordinary MD; its pressure
        # is the lowest REMD pressure. Without a ladder, every replica uses it.
        p_options = f' pressure_grid {grid}' if grid else f' pressure {p1} {p2} {np}'
        commands = (
            f'potential nep.txt\ntime_step {dt}\n'
            f'ensemble npt_scr {t1} {t2} {tau_t} {p1} {modulus} {tau_p}\n'
            f'multi_replica remd replicas {nt * np} exchange {exchange} '
            f'temp {t1} {t2} {nt}' + (p_options if ladder else '')
            + (f' pcouple_interval {interval}' if interval != 10 else '')
            + (f' equilibrate {equilibrate}' if equilibrate else '')
            + (' resume remd' if resume else '') + '\n'
        )
        if dump:
            commands += f'dump_xyz -1 0 {dump} trajectory.xyz {properties}\n'
        (directory / 'run.in').write_text(commands + f'run {steps}\n')
        result = subprocess.run([str(base.BINARY)], cwd=directory, env=base.ENV,
                                capture_output=True, text=True, timeout=timeout)
        (directory / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 1 if error else 0, result.stdout + result.stderr)
        if error:
            self.assertIn(error, result.stdout + result.stderr)
        return result

    def frame_properties(self, header, rows):
        layout = header['Properties'].split(':')
        result = {}
        offset = 0
        for name, kind, count in zip(layout[::3], layout[1::3], layout[2::3]):
            count = int(count)
            result[name] = [row[offset:offset + count] if kind == 'S' else
                            list(map(float, row[offset:offset + count])) for row in rows]
            offset += count
        return result

    def plain_snapshot(self, directory, header, rows, name):
        plain = self.root / name
        plain.mkdir()
        shutil.copyfile(directory / 'nep.txt', plain / 'nep.txt')
        values = self.frame_properties(header, rows)
        lattice = header['Lattice']
        with (plain / 'model.xyz').open('w') as output:
            output.write(f'{len(rows)}\npbc="T T T" Lattice="{lattice}" '
                         'Properties=species:S:1:pos:R:3:vel:R:3\n')
            for symbol, position in zip(values['species'], values['pos']):
                output.write(symbol[0] + ' ' + ' '.join(format(x, '.17g') for x in position) + ' 0 0 0\n')
        (plain / 'run.in').write_text(
            'potential nep.txt\ntime_step 0\nensemble nve\ndump_thermo 1\n'
            'dump_xyz 1 reference.xyz precision double mass velocity force potential virial\nrun 1\n')
        run = subprocess.run([str(base.BINARY)], cwd=plain, env=base.ENV,
                             capture_output=True, text=True, timeout=120)
        (plain / 'run.log').write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        reference_header, reference_rows = next(xyz_frames(plain / 'reference.xyz'))
        return self.frame_properties(reference_header, reference_rows)

    def compare_array(self, actual, expected, name, tolerance=3e-6):
        self.assertEqual(len(actual), len(expected), name)
        for atom, (a, b) in enumerate(zip(actual, expected)):
            for component, (x, y) in enumerate(zip(a, b)):
                self.assertTrue(math.isclose(x, y, rel_tol=tolerance, abs_tol=tolerance),
                                f'{name}: atom={atom}, component={component}, actual={x}, expected={y}')

    def assert_cached_pressure(self, directory, replica):
        header, atoms = next(xyz_frames(directory / f'remd_replica_{replica}_restart.xyz'))
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
        self.run_npt(directory, 100)
        rows = base.data_rows(directory / 'remd_npt_exchange.out')
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row[-1] == '1' for row in rows))
        for replica in range(2):
            self.assert_cached_pressure(directory, replica)
            header, _ = next(xyz_frames(directory / f'remd_replica_{replica}_restart.xyz'))
            self.assertNotEqual(volume(header), 27000)
        before = [next(xyz_frames(directory / f'remd_replica_{r}_restart.xyz'))[0]
                  for r in range(2)]
        self.run_npt(directory, 1, resume=True)
        for r in range(2):
            header, _ = next(xyz_frames(directory / f'remd_replica_{r}_restart.xyz'))
            self.assertNotEqual(volume(header), volume(before[r]))
            self.assertEqual(metadata(directory)[('scr_state', str(r))][2], '0')

    def test_pressure_is_measured_after_bdp_without_exchange(self):
        directory = self.prepare('bdp_pressure', free=True)
        self.run_npt(directory, 10, exchange=1000, tau_t=1)
        for replica in range(2):
            self.assert_cached_pressure(directory, replica)

    def test_exchange_between_barostat_steps_adds_no_update(self):
        directory = self.prepare('off_interval', free=True)
        self.run_npt(directory, 9, exchange=3)
        for r in range(2):
            header, _ = next(xyz_frames(directory / f'remd_replica_{r}_restart.xyz'))
            self.assertEqual(volume(header), 27000)
            self.assertEqual(metadata(directory)[('scr_state', str(r))][2], '0')

    def test_isotropic_box_scaling(self):
        directory = self.prepare('isotropic', free=True)
        self.run_npt(directory, 31)
        for r in range(2):
            header, _ = next(xyz_frames(directory / f'remd_replica_{r}_restart.xyz'))
            h = list(map(float, header['Lattice'].split()))
            self.assertNotEqual(h[0], 30)
            for d in (1, 2):
                self.assertAlmostEqual(h[4 * d] / h[0], 1, delta=1e-13)
            self.assertEqual([h[i] for i in (1, 2, 3, 5, 6, 7)], [0] * 6)

    def test_pressure_defaults_to_npt_scr_pressure(self):
        directory = self.prepare('default_pressure', free=True)
        self.run_npt(directory, 10, temp=(300, 400, 2), pressure=(.001, .001, 1), ladder=False)
        state = metadata(directory)
        self.assertEqual(state[('grid',)], ['2', '1'])
        self.assertEqual([float(x) for x in state[('pressure_table',)]], [.001, .001])

    def test_npt_scr_uses_ordinary_md_syntax(self):
        directory = self.prepare('ordinary_syntax', free=True)
        self.run_npt(directory, 1)
        original = (directory / 'run.in').read_text()
        ensemble = 'ensemble npt_scr 300 400 100 0 2 1000'
        self.assertIn(ensemble, original)
        cases = [
            # Former NPT REMD form with two pressures and extra options.
            (ensemble, 'ensemble npt_scr 300 400 100 0 0 2 1000', 'isotropic npt_scr form'),
            (ensemble, ensemble + ' couple z', 'isotropic npt_scr form'),
            (ensemble, ensemble + ' pcouple_interval 10', 'isotropic npt_scr form'),
            # Independent box lengths and triclinic cell changes of ordinary MD.
            (ensemble, 'ensemble npt_scr 300 400 100 0 0 0 2 2 2 1000', 'isotropic npt_scr form'),
            (ensemble, 'ensemble npt_scr 300 400 100 0 0 0 0 0 0 2 2 2 2 2 2 1000',
             'isotropic npt_scr form'),
            # A modulus above 2000 GPa disables pressure control in npt_scr.
            (ensemble, 'ensemble npt_scr 300 400 100 0 2001 1000', 'active barostat'),
            (ensemble, 'ensemble npt_scr 300 400 100 0.001 2 1000', 'lowest NPT REMD pressure'),
            (ensemble, 'ensemble npt_scr 300 401 100 0 2 1000', 'first and last NPT REMD'),
            ('multi_replica remd ', 'multi_replica remd pcouple_interval 600 ', 'twice pcouple_interval'),
            ('multi_replica remd ', 'multi_replica remd pcouple_interval 0 ', 'pcouple_interval'),
        ]
        for old, new, error in cases:
            with self.subTest(change=new):
                (directory / 'run.in').write_text(original.replace(old, new))
                result = subprocess.run([str(base.BINARY)], cwd=directory, env=base.ENV,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(error, result.stdout + result.stderr)

    def test_pressure_and_sheared_grid_exchange_rules(self):
        for sheared, nt, np in ((False, 1, 3), (False, 3, 3), (True, 3, 3), (False, 1, 2), (False, 2, 1)):
            with self.subTest(sheared=sheared, nt=nt, np=np):
                directory = self.prepare(f'grid_{nt}x{np}_{sheared}', free=True)
                grid = None
                if sheared:
                    grid = 'pressures.dat'
                    (directory / grid).write_text('0 .001 .002\n.0001 .0011 .0021\n.0002 .0012 .0022\n')
                options = dict(temp=(300, 450 if nt > 1 else 300, nt),
                               pressure=(0, (.0022 if sheared else .002) if np > 1 else 0, np), grid=grid)
                self.run_npt(directory, 40, **options)
                rows = base.data_rows(directory / 'remd_npt_exchange.out')
                # Temperature and pressure edges alternate, each with even and odd phases;
                # an axis with two conditions has only the even phase.
                sizes = {'T': nt, 'P': np}
                cycle = [(d, 0 if sizes[d] == 2 else parity)
                         for parity in (0, 1) for d in ('T', 'P') if sizes[d] > 1]
                phases = [cycle[r % len(cycle)] for r in range(4)]
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
                self.assertEqual(list(map(int, metadata(directory)[('replica_to_temperature',)])), mapping)

    def test_reject_invalid_pressure_conditions(self):
        directory = self.prepare('invalid_conditions', free=True)
        ranges = [dict(pressure=(.002, .001, 3)), dict(pressure=(0, .001, 1)), dict(temp=(400, 300, 2))]
        for options in ranges:
            with self.subTest(options=options):
                self.run_npt(directory, 1, error='REMD range needs min < max', **options)
        # Two temperatures and two pressures per temperature.
        grids = [
            ('0 .001\n.0001\n', 'rows must have the same length'),
            ('0 .001\n', 'one row per temperature'),
            ('0 .001\n.0001 .0011\n.0002 .0012\n', 'dimensions exceed the replica count'),
            ('0 nan\n.0001 .0011\n', 'must contain finite pressures'),
            ('0 0\n.0001 .0011\n', 'increasing within each temperature row'),
            ('0 .001 .002\n.0001 .0011 .0021\n', 'dimensions must multiply to the replica count'),
        ]
        for text, error in grids:
            with self.subTest(grid=text):
                (directory / 'pressures.dat').write_text(text)
                self.run_npt(directory, 1, pressure=(0, .001, 2), grid='pressures.dat', error=error)

    def assert_checkpoint_valid(self, directory, steps, count, equilibration=0):
        state = metadata(directory)
        self.assertEqual(int(state[('completed_steps',)][0]), steps)
        mapping = list(map(int, state[('replica_to_temperature',)]))
        self.assertEqual(sorted(mapping), list(range(count)))
        nt, np = map(int, state[('grid',)])
        temperatures = list(map(float, state[('temperatures',)]))
        pressures = list(map(float, state[('pressure_table',)]))
        interval = int(state[('barostat',)][2])
        for r in range(count):
            header, atoms = next(xyz_frames(directory / f'remd_replica_{r}_restart.xyz'))
            self.assertEqual(int(header['restart_step']), steps)
            label = mapping[r]
            self.assertEqual(int(header['condition_label']), label)
            self.assertEqual(int(header['temperature_label']), label)
            self.assertAlmostEqual(float(header['target_temperature']), temperatures[label % nt])
            self.assertAlmostEqual(float(header['target_pressure']), pressures[(label % nt) * np + label // nt])
            scr = state[('scr_state', str(r))]
            self.assertEqual(int(scr[1]), steps + equilibration)
            self.assertEqual(int(scr[2]), (steps + equilibration) % interval == 0)
            self.assertTrue(all(math.isfinite(float(x)) for x in scr[3:8]))
            self.assertTrue(state[('bdp_rng', str(r))])
            self.assertGreater(len(scr[8:]), 600)
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
                    self.run_npt(directory, 10, equilibrate=equilibration, **options)
                    split = self.root / f'split_{free}_{equilibration}'
                    shutil.copytree(directory, split)
                    self.run_npt(directory, 23, resume=True, **options)
                    self.run_npt(split, 5, resume=True, **options)
                    self.run_npt(split, 18, resume=True, **options)
                    self.assert_checkpoint_valid(directory, 33, 4, equilibration)
                    self.assert_checkpoint_valid(split, 33, 4, equilibration)
                    self.assertEqual(metadata(directory)[('exchange_count',)],
                                     metadata(split)[('exchange_count',)])
                    for r in range(4):
                        self.assertEqual([h['step'] for h, _ in xyz_frames(directory / f'trajectory_replica_{r}.xyz')],
                                         [h['step'] for h, _ in xyz_frames(split / f'trajectory_replica_{r}.xyz')])

    def test_repeated_continuation_restores_full_state(self):
        for steps in (6, 10):  # checkpoint without and with a pending update
            with self.subTest(checkpoint_steps=steps):
                directory = self.prepare_orthogonal_pbte(f'repeat_{steps}', strain=.97)
                options = dict(exchange=5, interval=5, modulus=100, dump=1)
                self.run_npt(directory, steps, **options)
                checkpoint = metadata(directory)
                repeats = []
                for index in range(2):
                    path = self.root / f'repeated_{steps}_{index}'
                    shutil.copytree(directory, path)
                    self.run_npt(path, 11, resume=True, **options)
                    repeats.append(path)
                    self.assert_checkpoint_valid(path, steps + 11, 2)
                    state = metadata(path)
                    self.assertEqual(state[('mode',)], ['remd'])
                    self.assertEqual(state[('ensemble',)], ['npt_scr'])
                    for replica in range(2):
                        before = checkpoint[('scr_state', str(replica))]
                        after = state[('scr_state', str(replica))]
                        self.assertEqual(int(after[1]), int(before[1]) + 11)
                        self.assertEqual(int(after[2]), int(after[1]) % 5 == 0)
                        self.assertNotEqual(state[('bdp_rng', str(replica))],
                                            checkpoint[('bdp_rng', str(replica))])
                a, b = map(metadata, repeats)
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    if key[0] == 'scr_state':
                        self.assertEqual(a[key][:3], b[key][:3])
                        self.assertEqual(a[key][8:], b[key][8:])  # RNG and cached normal
                        for x, y in zip(a[key][3:8], b[key][3:8]):
                            self.assertTrue(math.isclose(float(x), float(y), rel_tol=1e-8, abs_tol=1e-9))
                    elif key[0] == 'exchange_probability_sum':
                        for x, y in zip(a[key], b[key]):
                            self.assertAlmostEqual(float(x), float(y), delta=1e-7)
                    else:
                        self.assertEqual(a[key], b[key], key)
                for replica in range(2):
                    ah, ar = next(xyz_frames(repeats[0] / f'remd_replica_{replica}_restart.xyz'))
                    bh, br = next(xyz_frames(repeats[1] / f'remd_replica_{replica}_restart.xyz'))
                    self.compare_array([list(map(float, ah['Lattice'].split()))],
                                       [list(map(float, bh['Lattice'].split()))], 'restored box', 1e-9)
                    self.compare_array([list(map(float, row[1:])) for row in ar],
                                       [list(map(float, row[1:])) for row in br], 'restored atoms', 1e-8)

    def test_missing_and_duplicate_restart_metadata(self):
        directory = self.prepare('metadata_validation', free=True)
        self.run_npt(directory, 10)
        path = directory / 'remd_restart.meta'
        lines = [x for x in path.read_text().splitlines() if x]
        for index, line in enumerate(lines):
            for duplicate in (False, True):
                with self.subTest(key=line.split()[:2], duplicate=duplicate):
                    edited = lines + [line] if duplicate else lines[:index] + lines[index + 1:]
                    path.write_text('\n'.join(edited) + '\n')
                    before = path.read_bytes()
                    self.run_npt(directory, 1, resume=True, error='REMD')
                    self.assertEqual(path.read_bytes(), before)

    def test_restart_version_and_parameters(self):
        directory = self.prepare('restart_parameters', free=True)
        self.run_npt(directory, 10)
        path = directory / 'remd_restart.meta'
        original = path.read_text()
        version = metadata(directory)[('format_version',)][0]
        path.write_text(original.replace(f'format_version {version}', 'format_version 2'))
        self.run_npt(directory, 1, resume=True, error='Unsupported REMD restart format_version')
        path.write_text(original)
        for options in (dict(exchange=11), dict(interval=5), dict(modulus=3), dict(tau_p=1100),
                        dict(tau_t=101), dict(temp=(300, 401, 2))):
            with self.subTest(options=options):
                self.run_npt(directory, 1, resume=True, error='REMD restart', **options)
        # Singular/plural messages differ for grid vs other parameter checks;
        # potential identity is also checked independently of the XYZ files.
        potential = directory / 'nep.txt'
        potential.write_text(potential.read_text() + '\n')
        self.run_npt(directory, 1, resume=True, error='potential file does not match')

    def test_corrupted_scr_state_is_rejected(self):
        directory = self.prepare('scr_state_validation', free=True)
        self.run_npt(directory, 10)  # an update is pending at the checkpoint
        path = directory / 'remd_restart.meta'
        original = path.read_text()
        line = next(x for x in original.splitlines() if x.startswith('scr_state 0 '))
        # scr_state <replica> <version> <counter> <pending> <T> <U> <p_xx> <p_yy> <p_zz> <RNG...>
        tokens = line.split()

        def edited(index, value):
            return ' '.join(tokens[:index] + [value] + tokens[index + 1:])

        cases = {'version': edited(2, '2'), 'negative counter': edited(3, '-1'),
                 'pending flag': edited(4, '0'), 'nonfinite pressure': edited(7, 'nan'),
                 'trailing token': line + ' 0'}
        for name, replacement in cases.items():
            with self.subTest(case=name):
                path.write_text(original.replace(line, replacement))
                before = path.read_bytes()
                self.run_npt(directory, 1, resume=True, error='Invalid SCR state in the REMD restart')
                self.assertEqual(path.read_bytes(), before)
        path.write_text(original)
        self.run_npt(directory, 1, resume=True)

    def test_restart_xyz_validation(self):
        directory = self.prepare('xyz_validation', free=True)
        self.run_npt(directory, 10)
        path = directory / 'remd_replica_0_restart.xyz'
        original = path.read_text()
        changes = [
            ('pbc="T T T"', 'pbc="F F T"', 'pbc does not match'),
            ('Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3',
             'Properties=species:S:1:vel:R:3:mass:R:1:pos:R:3', 'REMD'),
            ('replica=0', 'replica=1', 'REMD'), ('restart_step=10', 'restart_step=9', 'REMD'),
        ]
        for old, new, error in changes:
            with self.subTest(change=new):
                path.write_text(original.replace(old, new))
                self.run_npt(directory, 1, resume=True, error=error)
        for lattice in ('0 0 0 0 30 0 0 0 30', '30 0 0 5 30 0 0 0 30',
                        '30 0 0 0 30 0 0 0 30 1'):
            with self.subTest(lattice=lattice):
                path.write_text(re.sub(r'Lattice="[^"]+"', f'Lattice="{lattice}"', original))
                self.run_npt(directory, 1, resume=True, error='REMD')
        # A nonperiodic model is rejected before the restart files are read.
        path.write_text(original)
        model = directory / 'model.xyz'
        model.write_text(model.read_text().replace('pbc="T T T"', 'pbc="F F T"'))
        self.run_npt(directory, 1, resume=True, error='periodic boundaries in all directions')

    def test_reject_nonfinite_barostat_parameters(self):
        directory = self.prepare('nonfinite', free=True)
        for option in ('tau_t', 'modulus', 'tau_p'):
            for value in ('nan', 'inf', '-inf'):
                with self.subTest(option=option, value=value):
                    self.run_npt(directory, 1, error='npt_scr', **{option: value})

    def test_reject_qnep_and_nonperiodic_coupling(self):
        directory = self.prepare('charge', charge=True)
        self.run_npt(directory, 1, error='energy NEP model')
        directory = self.prepare('nonperiodic', free=True)
        path = directory / 'model.xyz'
        path.write_text(path.read_text().replace('pbc="T T T"', 'pbc="F T T"'))
        self.run_npt(directory, 1, error='periodic boundaries in all directions')

    def test_species_subset_matches_plain_gpumd(self):
        directory = self.prepare_orthogonal_pbte('subset')
        lines = (directory / 'model.xyz').read_text().splitlines()
        atoms = [line + ' 0 0 0' for line in lines[2:] if line.startswith('Pb ')]
        header = lines[1].replace('pos:R:3', 'pos:R:3:vel:R:3')
        (directory / 'model.xyz').write_text(f'{len(atoms)}\n{header}\n' + '\n'.join(atoms) + '\n')
        plain = self.root / 'plain'
        shutil.copytree(directory, plain)
        self.run_npt(directory, 1)
        energy = float(base.data_rows(directory / 'remd_npt_thermo.out')[0][2])
        (plain / 'run.in').write_text(
            'potential nep.txt\ntime_step 0.0001\nensemble nve\ndump_thermo 1\nrun 1\n')
        result = subprocess.run([str(base.BINARY)], cwd=plain, env=base.ENV,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = float(base.data_rows(plain / 'thermo.out')[0][2])
        self.assertAlmostEqual(energy, expected, delta=max(1e-7, 1e-6 * abs(expected)))

    def test_deformed_force_and_virial_match_plain_gpumd(self):
        for repeat in (2, 4):  # small-box and large-box paths
            with self.subTest(repeat=repeat):
                directory = self.prepare_orthogonal_pbte(f'deform_{repeat}', repeat=repeat, strain=.97)
                self.run_npt(directory, 11, interval=2, modulus=100, tau_p=1000,
                            dump=1, properties='mass velocity force potential virial', exchange=1000)
                for replica in range(2):
                    frames = list(xyz_frames(directory / f'trajectory_replica_{replica}.xyz'))
                    # Step 2 precedes an update, step 3 follows deformation/rebuild,
                    # step 4 exercises reuse, step 11 follows several updates.
                    for step in (2, 3, 4, 11):
                        header, rows = frames[step - 1]
                        actual = self.frame_properties(header, rows)
                        reference = self.plain_snapshot(directory, header, rows,
                            f'reference_{repeat}_{replica}_{step}')
                        for name in ('forces', 'energy_atom', 'virial'):
                            self.compare_array(actual[name], reference[name], name)
                        self.assertGreater(sum(abs(x) for row in reference['virial'] for x in row), 1e-5)
                        self.assertAlmostEqual(sum(row[0] for row in actual['energy_atom']),
                                               float(header['potential_energy']), delta=1e-7)
                    original_volume = (repeat * 6.570392 * .97) ** 3
                    self.assertNotEqual(volume(frames[-1][0]), original_volume)

    def check_interacting_pressure(self, directory, replica):
        header, rows = list(xyz_frames(directory / f'trajectory_replica_{replica}.xyz'))[-1]
        values = self.frame_properties(header, rows)
        state = metadata(directory)[('scr_state', str(replica))]
        self.assertEqual(state[2], '1')
        self.assertGreater(sum(abs(x) for row in values['virial'] for x in row), 1e-5)
        kinetic = [sum(m[0] * (v[d] * TIME_UNIT) ** 2 for m, v in zip(values['mass'], values['vel']))
                   for d in range(3)]
        for d in range(3):
            virial = sum(row[4 * d] for row in values['virial'])
            expected = (kinetic[d] + virial) / volume(header)
            self.assertAlmostEqual(float(state[5 + d]), expected,
                                   delta=max(1e-10, abs(expected) * 1e-8))
        self.assertAlmostEqual(float(state[3]), sum(kinetic) / ((3 * len(rows) - 3) * K_B), delta=1e-7)

    def test_nonzero_virial_pressure_after_bdp_and_exchange(self):
        for exchange in (1000, 1):
            with self.subTest(exchange=exchange):
                directory = self.prepare_orthogonal_pbte(f'pressure_{exchange}', strain=.97)
                options = dict(exchange=exchange, interval=1, tau_t=1, modulus=100, tau_p=1000,
                               dump=1, properties='mass velocity force potential virial')
                for attempt in range(32):
                    self.run_npt(directory, 1, resume=attempt > 0, **options)
                    if exchange == 1000 or base.data_rows(directory / 'remd_npt_exchange.out')[-1][-1] == '1':
                        break
                else:
                    self.fail('No accepted temperature exchange in 32 attempts')
                for replica in range(2):
                    self.check_interacting_pressure(directory, replica)

    def test_interacting_exchange_has_nontrivial_probability(self):
        directory = self.prepare_orthogonal_pbte('nontrivial')
        options = dict(exchange=1, interval=10, tau_t=1e20, modulus=100, tau_p=1e20, dt=1e-6)
        self.run_npt(directory, 1, **options)
        header, rows = next(xyz_frames(directory / 'remd_replica_0_restart.xyz'))
        reference_a = self.plain_snapshot(directory, header, rows, 'energy_a')
        shifted = [row[:] for row in rows]
        shifted[0][1] = str(float(shifted[0][1]) + .03)
        reference_b = self.plain_snapshot(directory, header, shifted, 'energy_b')
        ea = sum(row[0] for row in reference_a['energy_atom'])
        eb = sum(row[0] for row in reference_b['energy_atom'])
        self.assertGreater(abs(ea - eb), 1e-7)
        low, high = (rows, shifted) if ea < eb else (shifted, rows)
        mapping = list(map(int, metadata(directory)[('replica_to_temperature',)]))
        for replica, label in enumerate(mapping):
            path = directory / f'remd_replica_{replica}_restart.xyz'
            lines = path.read_text().splitlines()
            positions = low if label == 0 else high
            for i, row in enumerate(positions):
                atom = lines[i + 2].split()
                atom[1:4] = row[1:4]
                lines[i + 2] = ' '.join(atom)
            path.write_text('\n'.join(lines) + '\n')
        self.run_npt(directory, 1, resume=True, **options)
        row = base.data_rows(directory / 'remd_npt_exchange.out')[-1]
        ta, tb, pa, pb, ei, ej, vi, vj = map(float, row[6:14])
        expected = ((1 / ta - 1 / tb) * (ei - ej) +
                    (pa / ta - pb / tb) * (vi - vj) / PRESSURE_UNIT) / K_B
        self.assertGreater(abs(ei - ej), 1e-7)
        self.assertAlmostEqual(float(row[14]), expected, delta=1e-7)
        self.assertGreater(float(row[15]), 0)
        self.assertLess(float(row[15]), 1)
        self.assertAlmostEqual(float(row[15]), math.exp(expected), delta=1e-8)

    def test_unified_entry_and_ensemble_compatibility(self):
        directory = self.prepare('entry', free=True)
        self.run_npt(directory, 1)
        original = (directory / 'run.in').read_text()
        invalid = [original.replace('ensemble npt_scr 300 400 100 0 2 1000',
                                    'ensemble nvt_bdp 300 400 100').replace('temp 300 400 2', 'temp 300 400'),
                   original.replace('multi_replica remd ', 'multi_replica remd pcouple_interval 5 ')
                   .replace('ensemble npt_scr 300 400 100 0 2 1000', 'ensemble nvt_bdp 300 400 100')
                   .replace('temp 300 400 2 pressure 0 0 1', 'temp 300 400')]
        for commands in invalid:
            (directory / 'run.in').write_text(commands)
            result = subprocess.run([str(base.BINARY)], cwd=directory, env=base.ENV,
                                    capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(os.environ.get('GPUMD_BASELINE'), 'set GPUMD_BASELINE for upstream checkpoint compatibility')
    def test_nvt_checkpoint_from_upstream_master(self):
        baseline = Path(os.environ['GPUMD_BASELINE']).resolve()
        self.assertTrue(baseline.is_file())
        directory = self.prepare_orthogonal_pbte('legacy_nvt')
        commands = ('potential nep.txt\ntime_step 0.5\nensemble nvt_bdp 300 400 100\n'
                    'multi_replica remd replicas 2 exchange 5 temp 300 400\n'
                    'dump_xyz -1 0 1 trajectory.xyz velocity\nrun 7\n')
        (directory / 'run.in').write_text(commands)
        run = subprocess.run([str(baseline)], cwd=directory, env=base.ENV,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(metadata(directory)[('format_version',)], ['5'])
        candidate = self.root / 'legacy_nvt_candidate'
        shutil.copytree(directory, candidate)
        resumed = commands.replace('temp 300 400\n', 'temp 300 400 resume remd\n').replace('run 7\n', 'run 11\n')
        for path, binary in ((directory, baseline), (candidate, base.BINARY)):
            (path / 'run.in').write_text(resumed)
            run = subprocess.run([str(binary)], cwd=path, env=base.ENV,
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            state = metadata(path)
            self.assertEqual(state[('format_version',)], ['5'])
            self.assertEqual(state[('mode',)], ['remd'])
            self.assertEqual(state[('completed_steps',)], ['18'])
        a, b = metadata(directory), metadata(candidate)
        for key in a:
            if key[0] == 'exchange_probability_sum':
                for x, y in zip(a[key], b[key]):
                    self.assertAlmostEqual(float(x), float(y), delta=1e-7)
            else:
                self.assertEqual(a[key], b[key], key)
        for replica in range(2):
            ah, ar = next(xyz_frames(directory / f'remd_replica_{replica}_restart.xyz'))
            bh, br = next(xyz_frames(candidate / f'remd_replica_{replica}_restart.xyz'))
            self.assertEqual(ah.keys(), bh.keys())
            self.compare_array([list(map(float, row[1:])) for row in ar],
                               [list(map(float, row[1:])) for row in br], 'upstream NVT continuation', 1e-7)

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
        self.run_npt(directory, int(os.environ.get('REPLICA_SAMPLING_STEPS', '2000000')), temp=(temperature, temperature, 1),
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
        _, restart_atoms = next(xyz_frames(directory / 'remd_replica_0_restart.xyz'))
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
