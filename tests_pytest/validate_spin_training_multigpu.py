#!/usr/bin/env python3
"""Compare fixed-seed Spin3 training on one and four visible GPUs (population 12)."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import validate_main_nep_spin3_training as base


def main():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    if len(visible) < 4:
        raise RuntimeError('This test requires four allocated CUDA devices.')
    config = base.NEP_IN.replace('generation 1', 'generation 3').replace('batch 3', 'batch 1')
    config = config.replace('population 10', 'population 12')
    config += 'lambda_spin_response 0.3\nnep_compile 0\nseed 20261021\n'
    with tempfile.TemporaryDirectory(prefix='spin-training-multigpu-') as temporary:
        root = Path(temporary)
        results = []
        for devices in (visible[:1], visible[:4]):
            case = root / str(len(devices))
            base.write_case(case, config)
            (case / 'response.xyz').write_text(base.response_xyz())
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
            run = subprocess.run([str(base.NEP)], cwd=case, env=env, text=True, capture_output=True)
            if run.returncode:
                raise RuntimeError(run.stdout + run.stderr)
            results.append({
                'restart': base.columns(case / 'nep.restart', 2),
                'loss': [[float(v) for v in row.split()] for row in (case / 'loss.out').read_text().splitlines()
                         if row and not row.startswith('#')],
            })
        errors = {key: base.maximum_error(results[0][key], results[1][key]) for key in results[0]}
        assert errors['restart'] <= 2e-6, errors
        assert errors['loss'] <= 2e-5, errors
        print(json.dumps({'devices': [1, 4], 'population': 12, 'generations': 3, 'max_abs': errors}))


if __name__ == '__main__':
    main()
