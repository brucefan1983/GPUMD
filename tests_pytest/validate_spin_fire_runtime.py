#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spin FIRE analytic and endpoint regression, without third-party packages."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

FIXTURE = Path(__file__).parent / 'fixtures' / 'nep_spin3'
GPUMD = Path(os.environ.get('GPUMD_COMMAND', Path(__file__).parents[1] / 'src/gpumd')).resolve()


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def analytic_model(onsite):
    """E_i = tanh(0.1 |S_i|^2), or zero, using the actual Spin3 ANN path."""
    lines = (FIXTURE / 'nep4_spin3_o3c2_uniform.nep').read_text().splitlines()
    start = next(i+1 for i, s in enumerate(lines) if s.startswith('ANN '))
    # This fixture has 2 types, 30 neurons, 360 structural, 72 spin and 16 projection weights.
    dim, remainder = divmod(len(lines)-start-569, 61)
    assert remainder == 0 and dim > 30
    values = [0.0] * (len(lines)-start)
    values[-dim:] = [1.0] * dim
    if onsite:
        for t in range(2):
            offset = t * 30 * (dim+2)
            values[offset+30] = 0.1  # 5 radial + 5 * (4 + q222) structural channels
            values[offset+30*dim+30] = 1.0
    lines[2] = 'spin_baseline 0 0'
    return '\n'.join(lines[:start] + [str(x) for x in values]) + '\n'


def structure(spins=None):
    lines = (FIXTURE / 'model_large_box.xyz').read_text().splitlines()
    lines[1] = lines[1].replace(':spin:R:3', ':vel:R:3:spin:R:3:spin_vel:R:3')
    for i in range(4):
        f = lines[i+2].split()
        spin = spins[i] if spins is not None else f[4:7]
        lines[i+2] = ' '.join(f[:4] + ['0.001', '-0.002', '0.003'] + list(map(str, spin)) + ['0.02','-0.03','0.04'])
    return '\n'.join(lines)+'\n'


def read_frame(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    fields = re.search(r'Properties=([^\s]+)', lines[-n-1]).group(1).split(':')
    rows = [x.split() for x in lines[-n:]]
    result, offset = {}, 0
    for name, kind, width in zip(fields[::3], fields[1::3], fields[2::3]):
        width = int(width)
        result[name] = [r[offset:offset+width] if kind == 'S' else list(map(float,r[offset:offset+width])) for r in rows]
        offset += width
    return result


def run_case(root, name, potential, model, commands, dump=True, single_gpu=False):
    case = root / name
    case.mkdir()
    (case/'nep.txt').write_text(potential)
    (case/'model.xyz').write_text(model)
    suffix = ('ensemble nve\ntime_step 0\ndump_xyz 1 state.xyz precision double velocity spin mforce spin_velocity unwrapped_position\nrun 1\n' if dump else '')
    (case/'run.in').write_text('potential nep.txt\n'+commands+suffix)
    env = os.environ.copy()
    if single_gpu and env.get('CUDA_VISIBLE_DEVICES'):
        env['CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES'].split(',')[0]
    run = subprocess.run([str(GPUMD)], cwd=case, env=env, capture_output=True, text=True)
    (case/'stdout').write_text(run.stdout+run.stderr)
    return case, run


def check_run(case, run, tolerance=None, converged=True):
    assert run.returncode == 0, (case, run.stdout[-3000:], run.stderr)
    records = [(int(s),float(e),float(g),float(dt)) for s,e,g,dt in re.findall(
        r'spin_fire step (\d+) energy (\S+) max_mforce (\S+) dt (\S+)', run.stdout)]
    assert records, case
    assert all(math.isfinite(v) for r in records for v in r), case
    if converged:
        assert 'Spin FIRE finished: converged;' in run.stdout, (case,records[-1])
        assert records[-1][2] <= tolerance, (case, records[-1])
    state = read_frame(case/'state.xyz') if (case/'state.xyz').exists() else None
    if state:
        original = read_frame(case/'model.xyz')
        for field in ('pos','vel','spin_vel'):
            if field in original:
                assert max(abs(a-b) for x,y in zip(state['unwrapped_position' if field == 'pos' else field],original[field]) for a,b in zip(x,y)) < 1e-8, (case,field)
    return {'initial_energy':records[0][1], 'final_energy':records[-1][1],
            'initial_residual':records[0][2], 'final_residual':records[-1][2],
            'steps':records[-1][0], 'status':re.findall(r'Spin FIRE finished: (\w+)',run.stdout)[-1]}, state


def validate(root):
    summary = {}
    spins = [[1,0.2,0], [0.4,-0.3,0.7], [0,0,0], [0.6,0.1,-0.4]]
    model = structure(spins)
    field = 'add_mfield region 0 1 0 1 0 1 0 0 10000\n'
    case, run = run_case(root,'fixed_zeeman',analytic_model(False),model,
                         field+'minimize spin_fire 1e-7 3000 0\n')
    summary['fixed_zeeman'], state = check_run(case,run,1e-7)
    for before, after in zip(spins,state['spin']):
        assert abs(norm(before)-norm(after)) < 2e-8
        assert abs(after[2]-norm(before)) < 2e-7
    expected = -5.7883818060e-5*10000*sum(map(norm,spins))
    assert abs(summary['fixed_zeeman']['final_energy']-expected)<1e-9

    case,run = run_case(root,'variable_onsite',analytic_model(True),structure(),
                        'minimize spin_fire 1e-6 3000 1\n')
    summary['variable_onsite'],state = check_run(case,run,1e-6)
    assert max(map(norm,state['spin'])) < 6e-6
    initial = read_frame(case/'model.xyz')['spin']
    assert abs(summary['variable_onsite']['initial_energy']-sum(math.tanh(.1*norm(s)**2) for s in initial))<2e-7

    case,run = run_case(root,'zero_variable',analytic_model(True),structure([[0,0,0]]*4),
                        'add_mfield region 0 1 0 1 0 1 0 0 100\nminimize spin_fire 2e-6 3000 1\n')
    summary['zero_variable'],state = check_run(case,run,2e-6)
    for s in state['spin']:
        assert s[2] > .02
        assert abs(.2*s[2]/math.cosh(.1*s[2]**2)**2-5.7883818060e-3)<3e-6

    # A constant energy reference must not affect the FIRE trajectory or endpoint.
    shifted = analytic_model(True).replace('spin_baseline 0 0', 'spin_baseline 1024 1024')
    case,run = run_case(root,'shifted_onsite',shifted,structure(),
                        'minimize spin_fire 1e-6 3000 1\n')
    summary['shifted_onsite'],shifted_state = check_run(case,run,1e-6)
    original = read_frame(root/'variable_onsite/state.xyz')
    assert shifted_state['spin'] == original['spin']
    for key in ('steps', 'status', 'initial_residual', 'final_residual'):
        assert summary['shifted_onsite'][key] == summary['variable_onsite'][key]
    assert abs(summary['shifted_onsite']['final_energy'] - summary['variable_onsite']['final_energy'] - 4096) < 1e-10

    # Nonzero Zeeman force below representable spin updates must report a stall,
    # not convergence or an exhausted iteration budget.
    case,run = run_case(root,'unresolved_updates',analytic_model(False),structure([[0,0,1]]*4),
                        'add_mfield region 0 1 0 1 0 1 0 0 1e-12\nminimize spin_fire 1e-18 3000 1\n')
    summary['unresolved_updates'],state = check_run(case,run,converged=False)
    assert summary['unresolved_updates']['status'] == 'stalled'
    assert 'stagnation unresolved_spin_updates' in run.stdout
    assert summary['unresolved_updates']['final_residual'] > 1e-18
    assert state['spin'] == [[0,0,1]]*4

    # A pure Zeeman potential with a free magnitude has no stationary point:
    # spins keep moving and energy falls, but the force cannot converge.
    case,run = run_case(root,'residual_plateau',analytic_model(False),model,
                        'add_mfield region 0 1 0 1 0 1 0 0 100\nminimize spin_fire 1e-6 1500 1\n')
    summary['residual_plateau'],state = check_run(case,run,converged=False)
    record = summary['residual_plateau']
    assert record['status'] == 'stalled' and record['steps'] < 1500
    assert 'stagnation no_residual_progress' in run.stdout
    assert record['final_energy'] < record['initial_energy']
    assert record['final_residual'] > 1e-6
    assert state['spin'][0][2] > spins[0][2]+1

    case,run = run_case(root,'fixed_radial',analytic_model(True),structure(),
                        'minimize spin_fire 1e-7 10 0\nminimize spin_fire 1e-7 10 0\n')
    summary['fixed_radial'],state = check_run(case,run,1e-7)
    assert summary['fixed_radial']['steps']==0
    assert max(map(norm,state['mforce']))>.1

    case,run = run_case(root,'max_steps',analytic_model(False),model,
                        field+'minimize spin_fire 1e-7 1 0\n')
    summary['max_steps'],_ = check_run(case,run,converged=False)
    assert summary['max_steps']['status']=='maximum_steps'

    for i,command in enumerate(('minimize spin_fire -1 20 0', 'minimize spin_fire 1e-5 0 0',
                                'minimize spin_fire 1e-5 20 wrong', 'minimize spin_fire 1e-5 20',
                                'minimize spin_fire nan 20 0', 'minimize fire 1e-5 20',
                                'minimize spin_fire 1e-5 20 2', 'minimize spin_fire 1e-5 20 -1',
                                'minimize spin_fire 1e-5 20 fixed', 'minimize spin_fire 1e-5 20 variable')):
        case,run = run_case(root,'invalid_'+str(i),analytic_model(False),model,command+'\n',False)
        assert run.returncode != 0, case
    # Ordinary lattice FIRE remains available with a non-spin potential.
    # Its upstream multi-GPU backend does not support this small periodic cell.
    fixtures = FIXTURE.parent
    case, run = run_case(root, 'lattice_fire',
                        (fixtures/'models/nep_C.txt').read_text(),
                        (fixtures/'structures/C-nat16-rattled.xyz').read_text(),
                        'minimize fire 1e-3 20\n', False, single_gpu=True)
    assert run.returncode == 0, (case, run.stdout[-2000:])
    case, run = run_case(root, 'spin_fire_without_spin',
                        (fixtures/'models/nep_C.txt').read_text(),
                        (fixtures/'structures/C-nat16-rattled.xyz').read_text(),
                        'minimize spin_fire 1e-3 20 0\n', False, single_gpu=True)
    assert run.returncode != 0
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='spin-fire-') as temporary:
        root = args.output or Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        summary = validate(root)
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
