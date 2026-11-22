# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD. If not, see <https://www.gnu.org/licenses/>.
# Major contributors: Zheyong Fan (NEP multi-GPU release tests).


"""CPU-only parser, physical-oracle and temporary-runner checks."""

import contextlib
import io
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import cases
import checks
import run_tests as runner

# A zero-force temporary executable exercises controller behavior, not NEP physics.
EMULATOR = r'''
import os, re
from pathlib import Path
if os.environ.get('NEP_TEST_EMULATOR_FAIL'):
    print('synthetic loader failure')
    raise SystemExit(127)
gpus = len(os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(','))
print('number of GPUs =', gpus)
partitions = 3 if Path(__file__).name.endswith('_zheyong') else gpus
if partitions > 1:
    print('Try to use %d GPUs for the NEP part.' % partitions)
lines = Path('model.xyz').read_text().splitlines()
box = re.search(r'Lattice="([^"]+)"', lines[1]).group(1)
atoms = [line.split() for line in lines[2:]]
if 'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1' in lines[1]:
    atoms = [a[:4] + a[5:8] + [a[4], a[8]] for a in atoms]
dt, t = .05, 0.
dumps, thermo, restart = [], 0, 0

def xyz(path, chosen, time, is_restart=False, positions_only=False):
    schema = 'species:S:1:pos:R:3'
    if not positions_only:
        schema += ':mass:R:1:vel:R:3:group:I:1' if is_restart else (
            ':mass:R:1:vel:R:3:forces:R:3:energy_atom:R:1:virial:R:9:group:I:1:unwrapped_position:R:3')
    meta = 'pbc="T T T" Lattice="' + box + '" Properties=' + schema
    if not is_restart:
        meta += ' Time=%.8f energy=0 virial="0 0 0 0 0 0 0 0 0" stress="0 0 0 0 0 0 0 0 0"' % time
    with open(path, 'w' if is_restart else 'a') as out:
        out.write(str(len(chosen)) + '\n' + meta + '\n')
        for a in chosen:
            # Input schema: species pos vel mass group. Restart order differs.
            mass, group = a[7], a[8]
            row = a[:4]
            if not positions_only:
                row += [mass] + ['0'] * 3
                if not is_restart:
                    row += ['0'] * 13
                row += [group]
                if not is_restart:
                    row += a[1:4]
            out.write(' '.join(row) + '\n')

for line in Path('run.in').read_text().splitlines():
    p = line.split()
    if not p:
        continue
    if p[0] == 'replicate':
        factor = int(p[1]) * int(p[2]) * int(p[3])
        atoms *= factor
    elif p[0] == 'time_step':
        dt = float(p[1])
    elif p[0] == 'dump_xyz':
        dumps.append(p)
    elif p[0] == 'dump_thermo':
        thermo = int(p[1])
    elif p[0] == 'dump_restart':
        restart = int(p[1])
    elif p[0] == 'run':
        steps = int(p[1])
        for step in range(1, steps + 1):
            t += dt
            if thermo and step % thermo == 0:
                with open('thermo.out', 'a') as out:
                    out.write(' '.join(['0'] * 9) + ' ' + box + '\n')
            for dump in dumps:
                if step % int(dump[1]) == 0:
                    chosen = [a for a in atoms if a[8] == '1'] if 'group' in dump else atoms
                    xyz(dump[2], chosen, t, positions_only=len(dump) == 3)
            if restart and step % restart == 0:
                xyz('restart.xyz', atoms, t, is_restart=True)
        print('Run %d steps.' % steps)
        elapsed = .02 if partitions == 3 else .01
        print('Time used for this run = %g second.' % elapsed)
        print('Speed of this run = %g atom*step/second.' % (len(atoms) * steps / elapsed))
        dumps, thermo, restart = [], 0, 0
'''


def timing(build='single', rates=(100, 2000), steps=(10, 100), atoms=10):
    text = 'number of GPUs = 1\n'
    if build == 'zheyong':
        text += 'Try to use 3 GPUs for the NEP part.\n'
    for steps_i, rate in zip(steps, rates):
        text += ('Run {} steps.\nTime used for this run = {} second.\n'
                 'Speed of this run = {} atom*step/second.\n').format(steps_i, atoms * steps_i / rate, rate)
    return text


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tiny = {'box': [[30, 0, 0], [0, 30, 0], [0, 0, 30]], 'pbc': [True] * 3,
                     'atoms': [['Cu', [1, 2, 3], [0, 0, 0], 63.546],
                               ['Cu', [5, 6, 7], [0, 0, 0], 63.546001]]}

    def xyz(self, text):
        path = self.root / 'sample.xyz'
        path.write_text(text)
        return path

    def executable(self, name):
        path = self.root / name
        path.write_text('#!' + sys.executable + '\n' + EMULATOR)
        path.chmod(0o755)
        return path

    def sample(self):
        directory = self.root / 'sample'
        directory.mkdir()
        cases.write_state(directory / 'model.xyz', self.tiny)
        text, spec = cases.correctness('axis_x')
        (directory / 'run.in').write_text(text)
        subprocess.run([str(self.executable('emulator'))], cwd=directory, check=True,
                       stdout=subprocess.DEVNULL, env=runner.device_environment('cuda', '0'))
        return directory, cases.only_frame(directory / 'model.xyz'), spec

    def rewrite_xyz(self, path, change):
        frames = list(checks.frames(path))
        change(frames)
        with path.open('w') as stream:
            for frame in frames:
                stream.write(str(frame['n']) + '\n' + ' '.join(
                    '{}="{}"'.format(k, v) for k, v in frame['meta'].items()) + '\n')
                for i in range(frame['n']):
                    values = [str(frame['data'][k][i * width + j])
                              for k, (_, width) in frame['schema'].items() for j in range(width)]
                    stream.write(' '.join(values) + '\n')

    def test_fixed_case_and_performance_contract(self):
        self.assertEqual(len(set(cases.NAMES)), 18)
        self.assertEqual(sum(name.startswith('energy_') for name in cases.NAMES), 2)
        self.assertIn('water', cases.NAMES)
        for name in cases.NAMES:
            text, spec = cases.correctness(name)
            self.assertLessEqual(sum(spec['steps']) * spec['executions'], 100)
            self.assertEqual([int(x) for x in re.findall(r'^run (\d+)$', text, re.M)], spec['steps'])
        self.assertEqual((cases.WARMUP, cases.STEPS, cases.REPEATS), (10, 100, 3))
        self.assertEqual(9216 * math.prod(cases.PERFORMANCE['solid']), 1105920)
        self.assertEqual(63 * math.prod(cases.PERFORMANCE['water']), 1102248)
        for phase in cases.PERFORMANCE:
            text = cases.performance(phase)
            self.assertEqual(re.findall(r'^run (\d+)$', text, re.M), ['10', '100'])
            self.assertNotIn('dump_xyz', text)
            self.assertTrue(text.endswith('dump_thermo 100\nrun 100\n'))

    def test_numerical_bounds_are_preserved(self):
        self.assertEqual(checks.TOLERANCES, {
            'pos': (2e-5, 0), 'unwrapped_position': (2e-5, 0), 'vel': (2e-5, 0),
            'force': (5e-4, 2e-5), 'potential': (5e-5, 0), 'virial': (.002, 2e-5),
            'box': (1e-6, 0), 'temperature': (.003, 2e-5), 'pressure': (.003, 2e-5),
            'energy_solid': (5e-4, 0), 'energy_water': (.002, 0)})

    def test_water_composition_geometry_and_atom_identity(self):
        water = cases.water(runner.REPO)
        self.assertEqual(len(water['atoms']), 6048)
        self.assertEqual(sum(a[0] == 'H' for a in water['atoms']), 4032)
        self.assertEqual(len(set(a[3] for a in water['atoms'])), 6048)
        self.assertGreaterEqual(int(water['box'][0][0] / 3) // 3, 10)
        for d in range(3):
            self.assertAlmostEqual(math.fsum(a[3] * a[2][d] for a in water['atoms']), 0, places=10)
        for axis in range(3):
            crystal = cases.crystal(axis=axis)
            self.assertEqual(len(crystal['atoms']), 9216)
            self.assertGreaterEqual(int(crystal['box'][axis][axis] / 3) // 3, 10)

    def test_xyz_aliases_and_malformed_records(self):
        frame = cases.only_frame(self.xyz('1\nProperties=forces:R:3:energy_atom:R:1:group:I:1\n1 2 3 4 0\n'))
        self.assertEqual(list(frame['data']['force']), [1, 2, 3])
        self.assertEqual(list(frame['data']['potential']), [4])
        bad = ['1\nProperties=pos:R:3\n' + row + '\n'
               for row in ('1 2', '1 2 3 4', '1 2 nan', '1 2 inf', '1 2 1e999')]
        bad += ['1\nProperties=group:I:1\n0.5\n', '1\nProperties=pos:R:1:pos:R:1\n1 2\n',
                '1\nProperties=pos:R:bad\n1\n', '2\nProperties=pos:R:1\n1\n']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(checks.CheckError):
                list(checks.frames(self.xyz(text)))
        with self.assertRaises(checks.CheckError):
            cases.only_frame(self.xyz('1\nProperties=pos:R:1\n1\n' * 2))

    def test_thermo_shape_count_and_finite_values(self):
        path = self.root / 'thermo.out'
        row = ' '.join(['1'] * 18) + '\n'
        path.write_text('# format_version 1\n' + row)
        self.assertEqual(len(checks.thermo_rows(path, 1)), 1)
        for text in ('', row + row, ' '.join(['1'] * 17), row.replace('1', 'nan', 1)):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(checks.CheckError):
                checks.thermo_rows(path, 1)

    def test_metric_uses_absolute_and_relative_bounds(self):
        result = checks.metric([1, 2.1, -3.2], [1, 2, -3], .05)
        self.assertEqual(result['exceedances'], 2)
        self.assertAlmostEqual(result['max_abs'], .2)
        self.assertAlmostEqual(result['rms'], math.sqrt(.05 / 3))
        self.assertEqual(checks.metric([102], [100], .5, .02)['exceedances'], 0)
        for left, right in (([], []), ([1], [1, 2]), ([float('nan')], [0])):
            with self.assertRaises(checks.CheckError):
                checks.metric(left, right, 1)

    def test_logs_require_one_gpu_correct_partition_and_complete_runs(self):
        for build in ('single', 'zheyong'):
            text = timing(build)
            self.assertEqual(runner.log_rates(text, build, [10, 100])[0], [100, 2000])
            bad = [text.replace('number of GPUs = 1', 'number of GPUs = 3'),
                   text.replace('Run 100 steps.', 'Run 99 steps.'), text + 'Run 1 steps.\n',
                   text.replace('Time used for this run = 0.5 second.\n', ''),
                   text.replace('= 2000 atom*step', '= nan atom*step'),
                   text.replace('= 2000 atom*step', '= 0 atom*step')]
            bad.append(timing('zheyong' if build == 'single' else 'single'))
            bad.append(text.replace('Try to use 3 GPUs', 'Try to use 2 GPUs') if build == 'zheyong'
                       else text + 'Try to use 2 GPUs for the NEP part.\n')
            for log in bad:
                with self.subTest(build=build, log=log), self.assertRaises(checks.CheckError):
                    runner.log_rates(log, build, [10, 100])

    def test_device_aliases_inherited_selection_and_invalid_masks(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4,5', 'HIP_VISIBLE_DEVICES': '2,3',
                                     'ROCR_VISIBLE_DEVICES': '6,7'}):
            cuda = runner.device_environment('cuda', None)
            hip = runner.device_environment('hip', None)
            self.assertEqual(cuda['CUDA_VISIBLE_DEVICES'], '4')
            self.assertEqual(hip['HIP_VISIBLE_DEVICES'], '2')
            self.assertNotIn('HIP_VISIBLE_DEVICES', cuda)
            self.assertNotIn('CUDA_VISIBLE_DEVICES', hip)
            self.assertEqual(hip['ROCR_VISIBLE_DEVICES'], '6,7')
        for device in ('', '-1', '0,1'):
            with self.assertRaises(checks.CheckError):
                runner.device_environment('cuda', device)

    def test_independent_output_checks_and_missing_files(self):
        directory, initial, spec = self.sample()
        self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'PASS')
        for name in ('state.xyz', 'selected.xyz', 'restart.xyz', 'thermo.out'):
            path = directory / name
            data = path.read_bytes()
            path.unlink()
            self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'ERROR')
            path.write_bytes(data)

    def test_wrong_identity_group_time_and_frame_count_fail(self):
        directory, initial, spec = self.sample()
        path = directory / 'state.xyz'
        original = path.read_bytes()
        changes = [lambda f: f[0]['data']['mass'].reverse(),
                   lambda f: f[0]['data']['group'].__setitem__(0, 1),
                   lambda f: f[0]['meta'].__setitem__('time', '9'), lambda f: f.pop()]
        for change in changes:
            path.write_bytes(original)
            self.rewrite_xyz(path, change)
            self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'FAIL')
        path.write_bytes(original)
        self.rewrite_xyz(directory / 'selected.xyz', lambda f: f[0]['data']['pos'].__setitem__(0, 99))
        self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'FAIL')

    def test_wrong_energy_and_virial_sums_fail_independently(self):
        directory, initial, spec = self.sample()
        path = directory / 'state.xyz'
        original = path.read_bytes()
        for key in ('potential', 'virial'):
            path.write_bytes(original)
            self.rewrite_xyz(path, lambda f: f[0]['data'][key].__setitem__(0, 1))
            status, details = runner.assess(checks.check_run, directory, initial, spec)
            self.assertEqual(status, 'FAIL')
        path.write_bytes(original)
        self.rewrite_xyz(path, lambda f: f[0]['meta'].__setitem__('energy', '1'))
        self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'FAIL')

    def test_berendsen_kinetic_energy_is_sampled_before_rescaling(self):
        directory, initial, _ = self.sample()
        def velocities(frames):
            for frame in frames:
                for i, mass in enumerate(frame['data']['mass']):
                    frame['data']['vel'][3 * i] = math.sqrt(3 * 8.617343e-5 * 201 / mass) / 10.18051
        for name in ('state.xyz', 'selected.xyz', 'restart.xyz'):
            self.rewrite_xyz(directory / name, velocities)
        row = [200, 3 * 8.617343e-5 * 200] + [0] * 7 + [x for r in self.tiny['box'] for x in r]
        (directory / 'thermo.out').write_text((' '.join(map(str, row)) + '\n') * 2)
        _, spec = cases.correctness('nvt')
        self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'PASS')
        spec.pop('berendsen')
        self.assertEqual(runner.assess(checks.check_run, directory, initial, spec)[0], 'FAIL')

    def test_nve_drift_is_normalized_per_atom(self):
        initial = {'n': 2}
        _, spec = cases.correctness('energy_solid')
        path = self.root / 'thermo.out'
        def write_energy(total_drift):
            rows = [[0, 0, total_drift * i / 9] + [0] * 15 for i in range(10)]
            path.write_text('\n'.join(' '.join(map(str, row)) for row in rows) + '\n')
        write_energy(.0008)
        self.assertEqual(runner.assess(checks.check_run, self.root, initial, spec)[0], 'PASS')
        write_energy(.0012)
        self.assertEqual(runner.assess(checks.check_run, self.root, initial, spec)[0], 'FAIL')

    def test_performance_uses_measured_run_and_atom_normalization(self):
        path = self.root / 'thermo.out'
        path.write_text(' '.join(['0'] * 18) + '\n')
        sample = ([999, 2000], [.1, .5])
        self.assertEqual(runner.performance_rate(self.root, sample, 10), 2000)
        with self.assertRaises(checks.CheckError):
            runner.performance_rate(self.root, sample, 11)
        path.unlink()
        with self.assertRaises(OSError):
            runner.performance_rate(self.root, sample, 10)
        a, acv = runner.rate_stats([80, 100, 120])
        b, bcv = runner.rate_stats([30, 50, 70])
        self.assertEqual((a, b, acv, bcv, b / a), (100, 50, 20, 40, .5))
        for rates in ([1, 2], [1, 2, float('nan')], [1, 2, 0]):
            with self.assertRaises(checks.CheckError):
                runner.rate_stats(rates)

    def controller(self, mode, fail=False):
        single, zheyong = self.executable('emulator'), self.executable('emulator_zheyong')
        potential = self.root / cases.POTENTIAL
        potential.parent.mkdir(parents=True, exist_ok=True)
        potential.write_text('nep 89 Cu\nformat_version 1\nmodel nep\nzbl 1 2\ncutoff 6 5 274 158\n')
        created = []
        def temporary(**kwargs):
            temporary = tempfile.TemporaryDirectory(dir=self.root, **kwargs)
            created.append(Path(temporary.name))
            return temporary
        output = io.StringIO()
        # Fixture sizing and the fixed production inventory are tested separately.
        names = ('axis_x', 'append', 'restart', 'water', 'energy_solid', 'energy_water')
        with patch.object(runner, 'REPO', self.root), patch.object(runner, 'TemporaryDirectory', temporary), \
                patch.object(cases, 'NAMES', names), patch.object(cases, 'crystal', return_value=self.tiny), \
                patch.object(cases, 'water', return_value=self.tiny), \
                patch.object(cases, 'PERFORMANCE', {'solid': (2, 1, 1), 'water': (1, 2, 1)}), \
                patch.dict(os.environ, {'NEP_TEST_EMULATOR_FAIL': '1' if fail else ''}), \
                contextlib.redirect_stdout(output):
            code = runner.main(['--single', str(single), '--zheyong', str(zheyong), '--mode', mode])
        self.assertTrue(created)
        self.assertTrue(all(not path.exists() for path in created))
        self.assertFalse((self.root / 'reports').exists())
        return code, output.getvalue()

    def test_controller_all_mode_and_temporary_cleanup(self):
        code, output = self.controller('all')
        self.assertEqual(code, 0, output)
        self.assertIn('Correctness: PASS (8/8)', output)
        self.assertIn('Performance: COMPLETE', output)
        self.assertEqual(output.count('0.500x'), 2)
        self.assertNotIn('FAIL', output)
        self.assertNotIn('ERROR', output)

    def test_completed_performance_survives_later_system_failure(self):
        execute = runner.Runner.execute
        completed = []
        def fail_water(instance, label, build, *args):
            if label.startswith('performance/water/'):
                raise checks.CheckError('synthetic water failure')
            result = execute(instance, label, build, *args)
            if label.startswith('performance/solid/'):
                completed.append(build)
            return result
        with patch.object(runner.Runner, 'execute', autospec=True, side_effect=fail_water):
            code, output = self.controller('performance')
        self.assertEqual(completed.count('single'), 3)
        self.assertEqual(completed.count('zheyong'), 3)
        self.assertEqual(code, 1)
        self.assertRegex(output, r'(?m)^solid\s+4\s+4\.000000e\+04\s+2\.000000e\+04\s+0\.500x')
        self.assertNotRegex(output, r'(?m)^water\s')
        self.assertIn('ERROR: synthetic water failure', output)
        self.assertIn('Correctness: NOT_RUN. Performance: ERROR', output)

    def test_process_failure_is_nonzero_and_blocks_performance(self):
        code, output = self.controller('performance', fail=True)
        self.assertEqual(code, 1)
        self.assertIn('exited with code 127', output)
        self.assertIn('synthetic loader failure', output)
        self.assertIn('Performance blocked', output)
        self.assertIn('Correctness: NOT_RUN. Performance: NOT_RUN', output)

    def test_first_case_execution_error_stops_remaining_correctness(self):
        code, output = self.controller('all', fail=True)
        self.assertEqual(code, 1)
        self.assertIn('Correctness: FAIL (0/8)', output)
        self.assertIn('remaining checks NOT_RUN', output)
        self.assertNotIn('repeat_single', output)
        self.assertNotIn('energy_water', output)
        self.assertIn('Performance blocked', output)

    def test_invalid_original_skips_its_repeat_and_restart(self):
        check_run = runner.check_run
        def invalid_single(directory, *args):
            if directory.parent.name == 'axis_x' and directory.name == 'single':
                raise checks.CheckError('synthetic invalid original')
            return check_run(directory, *args)
        with patch.object(runner, 'check_run', side_effect=invalid_single):
            code, output = self.controller('correctness')
        self.assertEqual(code, 1)
        self.assertRegex(output, r'repeat_single\s+SKIP\s+-\s+SKIP')
        self.assertRegex(output, r'repeat_zheyong\s+-\s+PASS\s+PASS')
        self.assertRegex(output, r'restart\s+SKIP\s+SKIP\s+SKIP')
        self.assertRegex(output, r'energy_water\s+PASS\s+PASS\s+PASS')
        self.assertIn('Correctness: FAIL (5/8)', output)

    def test_numerical_failure_blocks_performance_and_cleans_up(self):
        verdict = {'force': checks.metric([1], [0], 0)}
        with patch.object(runner, 'compare_runs', return_value=verdict):
            code, output = self.controller('all')
        self.assertEqual(code, 1)
        self.assertIn('Correctness: FAIL', output)
        self.assertIn('Performance blocked', output)
        self.assertNotIn('PERFORMANCE: 10', output)
        self.assertIn('max_abs=', output)


if __name__ == '__main__':
    unittest.main()
