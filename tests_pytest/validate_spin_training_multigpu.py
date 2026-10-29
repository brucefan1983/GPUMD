#!/usr/bin/env python3
"""Check 1/2/4-GPU training parity and exported-model energy on three structures.

Keep batch=1 as a centered-loss negative control; batch=3 exercises energy fitting.
The short synthetic run is a regression check, not a production accuracy target.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import validate_main_nep_spin3_training as base


def run_nep(case, env):
    run = subprocess.run([str(base.NEP)], cwd=case, env=env, text=True, capture_output=True)
    (case / 'run.log').write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f'{case}:\n{run.stdout}{run.stderr}')


def check_exported_energy(case, config, env, loss, batch):
    prediction = case / 'prediction'
    prediction_config = re.sub(
        r'^(population|generation|output_interval|save_potential|lambda_spin_response).*$',
        '', config, flags=re.MULTILINE)
    prediction_config = prediction_config.replace('version 4', 'version 4\nprediction 1')
    base.write_case(prediction, prediction_config)
    shutil.copyfile(case / 'nep.txt', prediction / 'nep.txt')
    run_nep(prediction, env)
    values = base.columns(prediction / 'energy_train.out', 2)
    # Independent labels for the three structures in base.TRAIN_XYZ, in eV/atom.
    # This also catches reference energies accidentally left out of output labels.
    assert len(values) == 3 and all(len(row) == 2 for row in values), values
    assert all(math.isfinite(v) for v in base.flatten(values)), values
    assert base.maximum_error([row[1] for row in values], [-1.0, -0.5, -1.5]) < 1e-12
    errors = [predicted - label for predicted, label in values]
    # Do not subtract a fitted mean: check the saved model's actual predictions.
    raw_rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_error = sum(errors) / len(errors)
    logged = [row[4] for row in loss]
    if batch == 3:
        assert all(value > 1e-5 for value in logged), ('energy loss is inactive', logged)
        # loss.out has five decimal places; allow its rounding and model serialization.
        assert abs(raw_rmse - logged[-1]) <= 6e-6, (raw_rmse, logged[-1])
        assert abs(mean_error) <= 2e-6, mean_error
        # Bounded regression guard for this fixed-seed three-generation fixture.
        assert raw_rmse <= 0.1, raw_rmse
    else:
        assert all(value == 0 for value in logged), logged
        assert raw_rmse > 1e-5, 'The batch=1 negative control must expose hidden energy error'
    return values, {'logged_energy_rmse': logged, 'raw_energy_rmse': raw_rmse,
                    'mean_energy_error': mean_error, 'prediction_and_label': values}


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
    parser.add_argument('--compile', choices=('native', 'jit', 'both'), default='native',
                        help='Run native kernels, required JIT kernels, or compare both')
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    if len(visible) < 4:
        raise RuntimeError('This test requires four allocated CUDA devices.')
    common_config = base.NEP_IN.replace('generation 1', 'generation 3')
    common_config = common_config.replace('population 10', 'population 12')
    common_config += 'lambda_spin_response 0.3\nseed 20261021\n'
    with tempfile.TemporaryDirectory(prefix='spin-training-multigpu-') as temporary:
        root = args.output.resolve() if args.output else Path(temporary)
        if args.output:
            root.mkdir(parents=True, exist_ok=False)
        report = {'passed': False, 'devices': [1, 2, 4], 'population': 12,
                  'generations': 3, 'energy_units': 'eV/atom'}
        modes = ('native', 'jit') if args.compile == 'both' else (args.compile,)
        report['modes'] = {}
        report['native_jit'] = {}
        native_results = {}
        for mode in modes:
            compiled = int(mode == 'jit')
            report['modes'][mode] = {}
            for batch in (1, 3):
                config = common_config.replace('batch 3', f'batch {batch}')
                config += f'nep_compile {compiled}\n'
                batch_root = root / mode / f'batch-{batch}'
                batch_root.mkdir(parents=True)
                results = []
                energies = {}
                for devices in (visible[:1], visible[:2], visible[:4]):
                    case = batch_root / str(len(devices))
                    base.write_case(case, config)
                    (case / 'response.xyz').write_text(base.response_xyz())
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
                    run_nep(case, env)
                    if compiled:
                        output = (case / 'run.log').read_text()
                        assert 'Spin layout JIT ready:' in output, output
                        assert 'specialization disabled' not in output.lower(), output
                    parameters, references = read_restart(case / 'nep.restart')
                    loss = [[float(v) for v in row.split()]
                            for row in (case / 'loss.out').read_text().splitlines()
                            if row and not row.startswith('#')]
                    assert len(loss) == 3 and all(len(row) == 14 for row in loss), loss
                    predictions, energies[str(len(devices))] = check_exported_energy(
                        case, config, env, loss, batch)
                    results.append({'restart': parameters, 'energy_reference': references,
                                    'loss': loss, 'energy_predictions': predictions})
                assert all(math.isfinite(v) for result in results for value in result.values()
                           for v in base.flatten(value))
                errors = {}
                for devices, result in zip([2, 4], results[1:]):
                    error = {key: base.maximum_error(results[0][key], result[key]) for key in result}
                    assert error['restart'] <= 2e-6, error
                    assert error['loss'] <= 2e-5, error
                    assert error['energy_reference'] <= 1e-12, error
                    assert error['energy_predictions'] <= 2e-6, error
                    errors[str(devices)] = error
                report['modes'][mode][str(batch)] = {
                    'energy_reference': results[0]['energy_reference'],
                    'max_abs': errors, 'energy': energies}
                (root / 'report.json').write_text(json.dumps(report, indent=2))
                if mode == 'native':
                    native_results[batch] = results
                elif batch in native_results:
                    jit_errors = []
                    for native, jit in zip(native_results[batch], results):
                        errors = {key: base.maximum_error(native[key], jit[key]) for key in native}
                        for key, error in errors.items():
                            assert error <= base.parity_tolerance(native[key], jit[key]), (key, error)
                        jit_errors.append(errors)
                    report['native_jit'][str(batch)] = jit_errors
        report['passed'] = True
        (root / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
