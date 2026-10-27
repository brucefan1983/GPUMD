#!/usr/bin/env python3
"""Check native SNES restore/scalers using generated fixtures and RESTART_PROBE.

The probe is built from restart_parity.cu against the current NEP objects.
No private dataset or pre-refactor executable is required.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import validate_main_nep_spin3_training as base
from validate_spin_training_multigpu import read_restart


def execute(binary, case):
    result = base.run(binary, case)
    (case / 'run.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def values(path):
    return base.finite_values(list(map(float, path.read_text().split())))


def validate(root):
    probe = Path(os.environ['RESTART_PROBE']).resolve()
    report = {}
    for kind in ('spin', 'structural'):
        config = base.NEP_IN
        if kind == 'structural':
            config = '\n'.join(line for line in config.splitlines()
                               if not line.startswith(('spin_', 'lambda_m', 'lambda_tau'))) + '\n'
        config += 'nep_compile 0\nseed 20261021\n'
        config = config.replace('generation 1', 'generation 3')
        initial = root / (kind + '-initial')
        base.write_case(initial, config)
        execute(base.NEP, initial)
        restored = []
        for name, extra in (('restore', ''), ('restore-import', 'import_q_scaler 1\n')):
            case = root / (kind + '-' + name)
            base.write_case(case, config.replace('generation 3', 'generation 0') + extra)
            for file in ('nep.txt', 'nep.restart'):
                shutil.copyfile(initial / file, case / file)
            execute(probe, case)
            assert read_restart(case/'restored.restart') == read_restart(initial/'nep.restart')
            restored.append(case)
        for case in restored:
            for filename in ('scaler.txt', 'baseline.txt'):
                values(case / filename)
            for filename in ('mu.pred', 'elite.pred'):
                rows = (case / filename).read_text().splitlines()
                assert rows, filename
                for row in rows:
                    base.finite_values(list(map(float, row.split()[2:])))
        for file in ('scaler.txt', 'baseline.txt', 'mu.pred', 'elite.pred'):
            assert (restored[0]/file).read_text() == (restored[1]/file).read_text(), (kind, file)
        assert (restored[0]/'mu.pred').read_text() != (restored[0]/'elite.pred').read_text()
        if kind == 'spin':
            scales = values(restored[0]/'scaler.txt')
            imported = root/'fresh-import'
            base.write_case(imported, config.replace('generation 3', 'generation 0')+'import_q_scaler 1\n')
            shutil.copyfile(initial/'nep.txt', imported/'nep.txt')
            execute(probe, imported)
            assert values(imported/'scaler.txt') == scales
            assert read_restart(imported/'restored.restart')[0] != read_restart(initial/'nep.restart')[0]
            fresh = []
            for seed in (1, 2):
                case = root/f'fresh-{seed}'
                base.write_case(case, config.replace('generation 3', 'generation 0').replace('seed 20261021', f'seed {seed}'))
                execute(probe, case)
                fresh.append(case)
            assert values(fresh[0]/'scaler.txt') == values(fresh[1]/'scaler.txt')
            assert read_restart(fresh[0]/'restored.restart')[0] != read_restart(fresh[1]/'restored.restart')[0]
        report[kind] = {'restore': 'passed', 'import': 'passed', 'search_mean_distinct_from_elite': True}
    return report


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='restart-scalers-') as temp:
        print(json.dumps(validate(Path(temp)), indent=2))
