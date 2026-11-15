#!/usr/bin/env python3
"""Deterministic GLSD spin/lattice timestep convergence through the real GPU executable."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from spin_analytic_model import REFERENCE_MODEL, structure
from validate_spin_glsd_runtime import damping
from validate_spin_multigpu import GATES, frame, sha


def initial_state():
    lines = structure().splitlines()
    velocities = [[.02, -.01, .005], [-.015, .006, -.003],
                  [.01, .008, -.004], [-.005, -.004, .006]]
    for i, velocity in enumerate(velocities, 2):
        row = lines[i].split()
        row[4:7] = map(str, velocity)
        lines[i] = ' '.join(row)
    return '\n'.join(lines) + '\n'


def run(root, binary, name, kind, dt, xyz):
    case = root / name
    case.mkdir()
    (case / 'nep.txt').write_bytes(REFERENCE_MODEL.read_bytes())
    (case / 'model.xyz').write_text(xyz)
    # Coupling periods are in timesteps: hold their physical durations fixed.
    tau = 1e8 if kind == 'weak-nhc' else 200.
    prefix = (f'nvt_glsd 300 300 {tau/dt:.17g} lattice on' if kind != 'mttk' else
              f'npt_glsd temp 300 300 iso 0 0 tperiod {tau/dt:.17g} pperiod {800/dt:.17g}')
    steps = round(40 / dt)
    (case / 'run.in').write_text(
        'potential nep.txt\n' +
        f'ensemble {prefix} {damping(50, 3)} gamma 3 stemp -1 seed 731 '
        'midpoint_tol 1e-8 midpoint_max_iter 100\n' +
        f'time_step {dt:.17g}\n'
        f'dump_xyz {steps} state.xyz precision double unwrapped_position_only velocity '
        'spin mforce force potential virial\n' + f'run {steps}\n')
    env = os.environ.copy()
    if env.get('CUDA_VISIBLE_DEVICES'):
        env['CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES'].split(',')[0]
    result = subprocess.run([str(binary)], cwd=case, env=env, capture_output=True, text=True)
    (case / 'run.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (case, result.stdout[-3000:], result.stderr)
    assert 'steps accepted at iteration cap before convergence: 0 of' in result.stdout, case
    state = frame(case / 'state.xyz')
    header = (case / 'state.xyz').read_text().splitlines()[1]
    state['cell'] = np.fromstring(re.search(r'Lattice="([^"]+)"', header)[1], sep=' ')
    assert all(np.isfinite(value).all() for value in state.values()), case
    return state


def convergence(root, binary, prefix, kind, xyz):
    ref = run(root, binary, prefix + '-ref', kind, .125, xyz)
    fine = run(root, binary, prefix + '-fine', kind, .0625, xyz)
    states = [run(root, binary, prefix + f'-dt{dt}', kind, dt, xyz) for dt in (4., 2., 1.)]
    fields = ('pos', 'vel', 'spin') + (('cell',) if kind == 'mttk' else ())
    report = {}
    for key in fields:
        errors = [float(np.max(np.abs(state[key] - fine[key]))) for state in states]
        refinement = float(np.max(np.abs(ref[key] - fine[key])))
        ratios = [a/b for a, b in zip(errors, errors[1:])]
        report[key] = {'max_abs_errors': errors, 'halving_ratios': ratios,
                       'reference_refinement': refinement}
    return report


def periodic_crossing(root, binary, xyz):
    lines = xyz.splitlines()
    shifts = []
    for i in range(2, len(lines)):
        row = lines[i].split()
        old = float(row[1])
        row[1] = str((old + 19.7) % 20)
        shifts.append([float(row[1]) - old, 0., 0.])
        lines[i] = ' '.join(row)
    actual = run(root, binary, 'periodic-crossing', 'weak-nhc', 1., '\n'.join(lines) + '\n')
    expected = frame(root / 'weak-nhc-dt1.0/state.xyz')
    assert actual['pos'][0, 0] > 20, 'the first atom must cross the periodic boundary'
    errors = {}
    for key in ('pos', 'vel', 'spin', 'force', 'mforce', 'potential', 'virial'):
        shifted = actual[key] - np.array(shifts) if key == 'pos' else actual[key]
        errors[key] = float(np.max(np.abs(shifted - expected[key])))
        atol, rtol = GATES[key]
        np.testing.assert_allclose(shifted, expected[key], atol=atol, rtol=rtol, err_msg=key)
    return errors


def validate(root, binary, reference=None):
    root.mkdir(parents=True, exist_ok=True)
    xyz = initial_state()
    report = {'binary_sha256': sha(binary), 'model_sha256': sha(REFERENCE_MODEL),
              'dt_fs': [4., 2., 1.], 'duration_fs': 40., 'noise': False, 'cases': {}}
    for kind in ('weak-nhc', 'nhc', 'mttk'):
        current_xyz = xyz.replace('20 0 0 0 20 0 0 0 20', '20 0 0 2 20 0 0 1 20') if kind == 'mttk' else xyz
        metrics = convergence(root, binary, kind, kind, current_xyz)
        report['cases'][kind] = metrics
        if reference:
            report.setdefault('baseline', {})[kind] = convergence(
                root, reference, 'baseline-' + kind, kind, current_xyz)
        (root / 'report.json').write_text(json.dumps(report, indent=2))
        print(kind, json.dumps(metrics), flush=True)
        for key, values in metrics.items():
            assert all(3.0 < ratio < 5.5 for ratio in values['halving_ratios']), (kind, key, values)
            assert values['reference_refinement'] < .08 * values['max_abs_errors'][-1], (kind, key, values)
    if reference:
        baseline = report['baseline']['weak-nhc']
        assert any(max(value['halving_ratios']) < 2.7 for value in baseline.values()), baseline
        report['baseline_sha256'] = sha(reference)
    report['periodic_crossing'] = periodic_crossing(root, binary, xyz)
    report['passed'] = True
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    validate(args.output.resolve(), args.binary.resolve(),
             args.reference.resolve() if args.reference else None)
