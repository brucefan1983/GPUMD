#!/usr/bin/env python3
"""Public executable, same-input Spin integration and endpoint parity on 1/2/4 GPUs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from spin_fixtures import FIXTURE
from validate_spin_multigpu import compare, frame, sha, structure, write_xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=('tspin', 'sib'), default=['tspin', 'sib'])
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    assert len(devices) >= 4, 'four allocated GPUs required'
    positions, spins, cell, velocity = structure((24, 4, 4))
    source = args.output / 'input.xyz'
    write_xyz(source, positions, spins, cell, velocity)
    report = {'binary_sha256': sha(args.binary), 'cases': {}}
    fields = ('pos', 'vel', 'spin', 'potential', 'force', 'mforce', 'virial')
    # A group field plus a region field checks additive Actions and endpoint energy.
    magnetic_field = ('add_mfield 0 1 4 -2 3\n'
                      'add_mfield region 0 1 0 1 0 1 7 -3 4\n')

    def run(name, count, ensemble, xyz, steps=8, dt=.01):
        directory = args.output / name
        directory.mkdir()
        shutil.copy(FIXTURE / 'nep_spin_o3c2_uniform.nep', directory / 'nep.txt')
        shutil.copy(xyz, directory / 'model.xyz')
        (directory / 'run.in').write_text(
            'potential nep.txt x\n' + magnetic_field + f'ensemble {ensemble}\ntime_step {dt}\n'
            f'dump_xyz {steps} state.xyz precision double velocity force potential virial spin mforce\n'
            f'dump_restart {steps}\nrun {steps}\n')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices[:count]))
        with (directory / 'run.log').open('w') as log:
            result = subprocess.run([str(args.binary)], cwd=directory, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (directory, (directory / 'run.log').read_text()[-3000:])
        values = frame(directory / 'state.xyz')
        assert all(np.isfinite(values[key]).all() for key in fields), directory
        return directory, values

    def record(name, left, right, keys):
        errors = compare(left, right, keys)
        report['cases'][name] = errors
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        assert not any(error['exceedances'] for error in errors.values()), (name, errors)
        print(name, 'passed', flush=True)

    for mode in args.modes:
        for lattice in ('nvt', 'npt'):
            prefix = (f'nvt_{mode} 300 320 100' if lattice == 'nvt' else
                      f'npt_{mode} temp 300 320 iso 0 .01 tperiod 100 pperiod 1000')
            ensemble = prefix + (' mass_factor 1 seed 731' if mode == 'tspin' else
                                 ' alpha .1 stemp 300 seed 731')
            baseline = None
            for count in (1, 2, 4):
                name = f'{lattice}-{mode}-g{count}'
                directory, values = run(name, count, ensemble, source)
                if baseline is None:
                    baseline = values
                else:
                    record(name, values, baseline, fields)
                if mode == 'sib':
                    _, endpoint = run(name + '-endpoint', count, 'nve', directory / 'restart.xyz', 1, 0)
                    record(name + '-endpoint', values, endpoint, ('potential', 'force', 'mforce', 'virial'))
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
