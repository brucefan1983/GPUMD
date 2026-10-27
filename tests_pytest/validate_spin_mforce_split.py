#!/usr/bin/env python3
"""Validate independent Spin3 magnetic RMSE terms, restart, and GPU parity.

Run on one allocated CUDA device, or pass --devices 1 2 4 for GPU parity. Supply
--oracle pointing to the linked spin_mforce_split.cu numerical test executable.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import validate_main_nep_spin3_training as base
from validate_spin_training_multigpu import read_restart, run_nep


def loss_rows(case):
    lines = (case / 'loss.out').read_text().splitlines()
    header = next(line.split()[2:] for line in reversed(lines) if line.startswith('# columns '))
    rows = [[float(v) for v in line.split()] for line in lines if line and not line.startswith('#')]
    assert all(len(row) == len(header) and all(math.isfinite(v) for v in row) for row in rows)
    return [dict(zip(header, row)) for row in rows]


def independent_metrics(case):
    """Recompute unweighted active-Fe residuals from exported Cartesian forces."""
    atoms = [line.split() for line in base.TRAIN_XYZ.splitlines() if line.startswith(('Fe ', 'Ge '))]
    values = base.columns(case / 'mforce_test.out', 6)
    assert len(values) == len(atoms)
    raw = parallel = perp = 0.0
    count = 0
    for atom, row in zip(atoms, values):
        if atom[0] != 'Fe':
            continue
        spin = list(map(float, atom[7:10]))
        error = [row[i] - row[3+i] for i in range(3)]
        norm2 = sum(v*v for v in error)
        length = math.sqrt(sum(v*v for v in spin))
        projection = sum(e*s for e, s in zip(error, spin)) / length if length else 0
        p = projection**2 if length else norm2/3
        raw += norm2; parallel += p; perp += norm2-p; count += 1
    return {'rmse_mforce_test': math.sqrt(raw/(3*count)),
            'rmse_m_parallel_test': math.sqrt(parallel/count),
            'rmse_m_perp_test': math.sqrt(perp/(2*count))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--oracle', type=Path)
    parser.add_argument('--devices', nargs='+', type=int, choices=(1, 2, 4), default=[1])
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    assert args.devices[0] == 1 and len(set(args.devices)) == len(args.devices)
    assert len(visible) >= max(args.devices), 'Not enough allocated GPUs'
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = visible[0]
    common = base.NEP_IN.replace('population 10', 'population 12').replace('generation 1', 'generation 3')
    common += 'seed 20261021\nnep_compile 0\nlambda_1 0\nlambda_2 0\n'
    split = common.replace('lambda_m 1.0', 'lambda_m_parallel 2\nlambda_m_perp 0.5')
    with tempfile.TemporaryDirectory(prefix='spin-mforce-split-') as temp:
        root = args.output.resolve() if args.output else Path(temp)
        if args.output: root.mkdir(parents=True, exist_ok=False)
        report = {'passed': False}
        initial = root / 'initial'
        base.write_case(initial, common)
        run_nep(initial, env)
        if args.oracle:
            run = subprocess.run([str(args.oracle.resolve())], cwd=initial, env=env, capture_output=True, text=True)
            (root / 'oracle.log').write_text(run.stdout + run.stderr)
            assert run.returncode == 0, run.stdout + run.stderr
            report['numerical_oracle'] = 'passed'
        # Require clear parser errors, not an arbitrary runtime failure.
        invalid = [
            ('lambda_m_parallel 1', 'Specify both'),
            ('lambda_m_perp 1', 'Specify both'),
            ('lambda_m 1\nlambda_m_parallel 1\nlambda_m_perp 1', 'Do not combine'),
            ('lambda_m_parallel 1\nlambda_m_perp 1\nlambda_mforce 1', 'Do not combine'),
            ('lambda_m_parallel 1\nlambda_m_perp 1\nlambda_m_parallel 2', 'Duplicate'),
            ('lambda_m_parallel 1 2\nlambda_m_perp 1', '1 parameter'),
        ]
        for key in ('lambda_m_parallel', 'lambda_m_perp'):
            for bad in ('-1', 'nan', 'inf', '1e40'):
                other = 'lambda_m_perp' if key.endswith('parallel') else 'lambda_m_parallel'
                invalid.append((f'{key} {bad}\n{other} 1', 'finite and non-negative'))
        for key, message in [('lambda_tau', 'finite and non-negative'),
                             ('spin_cutoff', 'finite positive')]:
            for bad in ('nan', 'inf', '1e40'):
                case = root / f'invalid-{key}-{bad}'
                config = common.replace(f'{key} ' + ('0.5' if key == 'lambda_tau' else '6.0'), f'{key} {bad}')
                base.write_case(case, config)
                run = subprocess.run([str(base.NEP)], cwd=case, env=env, capture_output=True, text=True)
                assert run.returncode != 0 and message in run.stdout + run.stderr, (key, bad, run.stdout, run.stderr)
        report['nonfinite_torque_cutoff_rejections'] = 6
        for i, (keywords, message) in enumerate(invalid):
            case = root / f'invalid-{i}'
            base.write_case(case, common.replace('lambda_m 1.0', keywords))
            run = subprocess.run([str(base.NEP)], cwd=case, env=env, capture_output=True, text=True)
            assert run.returncode != 0 and message in run.stdout + run.stderr, (keywords, run.stdout, run.stderr)
        report['parser_rejections'] = len(invalid)
        # Freeze a common parameter vector, including its q scaler and references.
        restart = (initial / 'nep.restart').read_text().splitlines()
        frozen = '\n'.join(restart[:3]) + '\n' + ''.join(f'{row.split()[0]} 0\n' for row in restart[3:])
        objective = common.replace('lambda_tau 0.5', 'lambda_tau 0') + 'lambda_e 0\nlambda_f 0\nlambda_v 0\n'
        terms = [('parallel', 2, 0), ('perp', 0, .5), ('mixed', 2, .5), ('disabled', 0, 0)]
        fixed = {}; predictions = []
        for name, lp, lt in terms:
            config = objective.replace('lambda_m 1.0', f'lambda_m_parallel {lp}\nlambda_m_perp {lt}')
            case = root / name
            base.write_case(case, config)
            (case / 'test.xyz').write_text(base.TRAIN_XYZ)
            shutil.copyfile(initial / 'nep.txt', case / 'nep.txt')
            (case / 'nep.restart').write_text(frozen)
            run_nep(case, env)
            rows = loss_rows(case); row = rows[-1]
            oracle = independent_metrics(case)
            for key, value in oracle.items(): assert abs(row[key]-value) < 8e-6, (key, row[key], value)
            expected = lp * row['rmse_m_parallel_train'] + lt * row['rmse_m_perp_train']
            for result in rows[1:]:
                assert abs(result['total'] - expected) < 1.5e-5, (name, result['total'], expected)
            predictions.append(base.columns(case / 'mforce_test.out', 3))
            fixed[name] = {'total': row['total'], 'independent_rmse': oracle}
        assert all(base.maximum_error(predictions[0], values) == 0 for values in predictions[1:])
        report['frozen_objective_checks'] = fixed
        # Ordinary stochastic training, multi-structure batch, and actual saved-model energy.
        results = []
        for count in args.devices:
            case = root / f'gpu-{count}'
            base.write_case(case, split)
            (case / 'test.xyz').write_text(base.TRAIN_XYZ)
            run_env = env.copy(); run_env['CUDA_VISIBLE_DEVICES'] = ','.join(visible[:count])
            run_nep(case, run_env)
            rows = loss_rows(case); row = rows[-1]
            oracle = independent_metrics(case)
            for key, value in oracle.items(): assert abs(row[key]-value) < 8e-6, (key, row[key], value)
            for suffix in ('train', 'test'):
                m = row['rmse_mforce_'+suffix]; p = row['rmse_m_parallel_'+suffix]; t = row['rmse_m_perp_'+suffix]
                assert abs(3*m*m-p*p-2*t*t) < 1e-5
            energy = base.columns(case / 'energy_test.out', 2)
            assert base.maximum_error([r[1] for r in energy], [-1, -.5, -1.5]) < 1e-12
            raw_e = math.sqrt(sum((r[0]-r[1])**2 for r in energy)/3)
            assert raw_e > 1e-5 and abs(raw_e-row['rmse_energy_test']) < 6e-6
            parameters, reference = read_restart(case / 'nep.restart')
            results.append({'restart': parameters, 'references': reference,
                            'loss': [list(r.values()) for r in rows],
                            'energy': energy, 'mforce': base.columns(case / 'mforce_test.out', 6)})
        errors = []
        for result in results[1:]:
            error = {key: base.maximum_error(results[0][key], result[key]) for key in result}
            assert error['restart'] <= 2e-6 and error['references'] <= 1e-12, error
            assert error['loss'] <= 2e-5 and error['energy'] <= 2e-6 and error['mforce'] <= 2e-6, error
            errors.append(error)
        report['multigpu_max_abs'] = dict(zip(map(str, args.devices[1:]), errors))
        report['passed'] = True
        (root / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
