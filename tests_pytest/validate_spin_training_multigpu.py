#!/usr/bin/env python3
"""Compare fixed-seed Spin3 training on one, two, and four visible GPUs (population 12)."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import validate_main_nep_spin3_training as base


def read_restart(path):
    lines = path.read_text().splitlines()
    assert lines[0] == 'nep_restart', lines[0]
    reference = lines[1].split()
    assert reference[0] == 'energy_reference'
    assert int(reference[1]) == len(reference) - 2
    count = lines[2].split()
    assert count[0] == 'parameters'
    parameters = [[float(v) for v in row.split()] for row in lines[3:]]
    assert len(parameters) == int(count[1])
    assert all(len(row) == 2 for row in parameters)
    return parameters, list(map(float, reference[2:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    if len(visible) < 4:
        raise RuntimeError('This test requires four allocated CUDA devices.')
    config = base.NEP_IN.replace('generation 1', 'generation 3').replace('batch 3', 'batch 1')
    config = config.replace('population 10', 'population 12')
    config += 'lambda_spin_response 0.3\nnep_compile 0\nseed 20261021\n'
    with tempfile.TemporaryDirectory(prefix='spin-training-multigpu-') as temporary:
        root = args.output.resolve() if args.output else Path(temporary)
        if args.output:
            root.mkdir(parents=True, exist_ok=False)
        results = []
        for devices in (visible[:1], visible[:2], visible[:4]):
            case = root / str(len(devices))
            base.write_case(case, config)
            (case / 'response.xyz').write_text(base.response_xyz())
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
            run = subprocess.run([str(base.NEP)], cwd=case, env=env, text=True, capture_output=True)
            (case / 'run.log').write_text(run.stdout + run.stderr)
            if run.returncode:
                raise RuntimeError(run.stdout + run.stderr)
            parameters, references = read_restart(case / 'nep.restart')
            results.append({
                'restart': parameters,
                'energy_reference': references,
                'loss': [[float(v) for v in row.split()] for row in (case / 'loss.out').read_text().splitlines()
                         if row and not row.startswith('#')],
            })
        assert all(math.isfinite(v) for result in results for value in result.values()
                   for v in base.flatten(value))
        errors = {}
        for devices, result in zip([2, 4], results[1:]):
            error = {key: base.maximum_error(results[0][key], result[key]) for key in result}
            assert error['restart'] <= 2e-6, error
            assert error['loss'] <= 2e-5, error
            assert error['energy_reference'] <= 1e-12, error
            errors[str(devices)] = error
        report = {'passed': True, 'devices': [1, 2, 4], 'population': 12,
                  'generations': 3, 'energy_reference': results[0]['energy_reference'],
                  'max_abs': errors}
        (root / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
