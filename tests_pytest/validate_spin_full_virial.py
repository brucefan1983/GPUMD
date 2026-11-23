#!/usr/bin/env python3
"""Check full NEP-Spin virials through the nep CLI (requires CUDA)."""

import argparse
import math
import re
import tempfile
from pathlib import Path

import validate_spin_training as base

PAIRS = [(i, j) for i in range(3) for j in range(3)]
CYCLIC = [(1, 2), (2, 0), (0, 1)]
TENSOR = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
VIRIAL = 'virial="0 0 0 0 0 0 0 0 0"'
FULL = 'virial="' + ' '.join(map(str, TENSOR)) + '"'
STRESS = 'stress="0 0 0 0 0 0 0 0 0"'
STRESS_ERROR = 'NEP-Spin does not accept stress labels. Use the nine-component virial field.'
CONFIG = base.NEP_IN + 'seed 20261122\nzbl 1.3\nlambda_e 1\nlambda_f 1\nlambda_v 1\n'


def check_near(actual, expected, tolerance):
    error = base.maximum_error(actual, expected)
    assert error <= tolerance, (actual, expected, error, tolerance)
    return error


def run_case(path, config, xyz, model=None, restart=None, test=None, error=None):
    base.write_case(path, config, xyz)
    for name, content in (('nep.txt', model), ('nep.restart', restart), ('test.xyz', test)):
        if content is not None:
            (path / name).write_text(content)
    result = base.run(base.NEP, path)
    output = result.stdout + result.stderr
    (path / 'run.log').write_text(output)
    if error:
        assert result.returncode and error in output, output
    else:
        assert result.returncode == 0, output
    return output


def physics(root, config, model, soc):
    def rotate(vec):
        x, y, z = map(float, vec)
        c, s = math.cos(0.6), math.sin(0.6)
        return [f'{v:.16g}' for v in (c*x - s*y, s*x + c*y, z)]

    # Joint rotation gives all three axes a resolvable torque with SOC.
    lines = base.TRAIN_XYZ.replace(VIRIAL, FULL, 1).splitlines()[:6]
    lattice = re.search(r'Lattice="([^"]+)"', lines[1])[1].split()
    cell = ' '.join(x for k in (0, 3, 6) for x in rotate(lattice[k:k+3]))
    lines[1] = re.sub(r'Lattice="[^"]+"', f'Lattice="{cell}"', lines[1])
    atoms = [line.split() for line in lines[2:]]
    for atom in atoms:
        atom[1:4] = rotate(atom[1:4])
        atom[7:10] = rotate(atom[7:10])
    frame = '\n'.join(lines[:2] + [' '.join(atom) for atom in atoms]) + '\n'
    run_case(root, config + 'prediction 1\nnep_compile 0\n', frame, model)
    assert not list(root.glob('virial_full_*.out'))
    virial = base.columns(root / 'virial_train.out', 19)
    assert len(virial) == 1 and virial[0][9:] == [x / 4 for x in TENSOR], virial
    w = [x * 4 for x in virial[0][:9]]
    forces = base.columns(root / 'force_train.out', 3, 4)
    moments = [list(map(float, row.split()[7:10])) for row in frame.splitlines()[2:]]
    mforces = base.columns(root / 'mforce_train.out', 3, 4)
    coords = [list(map(float, row.split()[1:4])) for row in frame.splitlines()[2:]]
    from_force = [sum(r[i] * f[j] for r, f in zip(coords, forces)) for i, j in PAIRS]
    skew = [w[PAIRS.index((i, j))] - w[PAIRS.index((j, i))] for i, j in CYCLIC]
    torque = [sum(s[i] * m[j] - s[j] * m[i] for s, m in zip(moments, mforces))
              for i, j in CYCLIC]
    result = {
        'force_moment': check_near(w, from_force, 3e-4),
        'spin_torque': check_near(skew, [-x for x in torque], 2e-6),
    }
    if soc:
        assert min(abs(x) for x in skew) > 1e-4, skew
    else:
        check_near(skew, [0.] * 3, 2e-6)
        check_near(torque, [0.] * 3, 2e-6)
    return result


def fixed_training(root, model, restart, check_jit):
    first, second, third, tail = base.TRAIN_XYZ.split(VIRIAL)
    symmetric = [1, 0, 3, 0, 4, 5, 3, 5, 6]
    perturbed = symmetric.copy()
    perturbed[1] = 1e-12
    targets = [symmetric, [0] * 9, perturbed]
    fields = ['virial="' + ' '.join(map(str, tensor)) + '"' for tensor in targets]
    unlabeled = '\n'.join(base.TRAIN_XYZ.splitlines()[:6]).replace(' ' + VIRIAL, '') + '\n'
    xyz = first + fields[0] + second + fields[1] + third + fields[2] + tail + unlabeled
    config = (CONFIG.replace('generation 1\n', 'generation 2\n').replace('batch 3\n', 'batch 4\n')
              + 'lambda_shear 0.5\n')
    output = {}
    for mode in (0, 1) if check_jit else (0,):
        case = root / f'fixed-{mode}'
        log = run_case(case, config + f'nep_compile {mode}\n', xyz, model, restart, xyz)
        if mode:
            assert 'Spin aggregate JIT ready:' in log and 'specialization disabled' not in log.lower()
        assert not list(case.glob('virial_full_*.out'))
        output[mode] = rows = base.columns(case / 'virial_test.out', 19)
        assert len(rows) == 4 and all(len(row) == 18 for row in rows), rows
        for row, tensor in zip(rows, targets):
            check_near(row[9:], [x / 4 for x in tensor], 1e-18)
        assert rows[3][9:] == [-1e6] * 9, rows[3]
        stress_rows = base.columns(case / 'stress_test.out', 13)
        assert len(stress_rows) == 4 and all(len(row) == 12 for row in stress_rows), stress_rows
        assert stress_rows[3][6:] == [-1e6] * 6, stress_rows[3]
        squared = [[(a - b) ** 2 for a, b in zip(row[:9], row[9:])] for row in rows[:3]]
        count = 9 * len(targets)
        loss = list(map(float, (case / 'loss.out').read_text().splitlines()[-1].split()))
        check_near(loss[11], math.sqrt(sum(map(sum, squared)) / count), 5e-5)
        weighted = sum(x * (1 if i in (0, 4, 8) else 0.25)
                       for frame in squared for i, x in enumerate(frame))
        total_without_virial = loss[2] + loss[3] + loss[4] + loss[5] + loss[7] + 0.5 * loss[8]
        check_near(loss[1] - total_without_virial, math.sqrt(weighted / count), 1e-4)
    return check_near(output[0], output[1], 5e-5) if check_jit else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-jit', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='nep-spin-full-virial-') as tmp:
        root = Path(tmp)
        for soc in (1, 0):
            initial = root / f'initial-{soc}'
            config = CONFIG.replace('spin_soc 1', f'spin_soc {soc}')
            run_case(initial, config + 'nep_compile 0\n', base.TRAIN_XYZ)
            model = (initial / 'nep.txt').read_text()
            print(f'spin_soc={soc}', physics(root / f'physics-{soc}', config, model, soc), flush=True)
            if soc:
                lines = (initial / 'nep.restart').read_text().splitlines()
                frozen = '\n'.join(lines[:3]) + '\n' + ''.join(
                    f'{row.split()[0]} 0\n' for row in lines[3:])
                print('nine-component supervision passed. JIT max error:',
                      fixed_training(root, model, frozen, args.check_jit), flush=True)
                for n, (label, error) in enumerate((
                    ('virial="1 2"', 'virial must contain exactly 9 values'),
                    (FULL + ' ' + VIRIAL, 'Duplicate virial field'),
                    (FULL.replace('1.0 ', '1e100 ', 1), 'within the float range'),
                    (STRESS, STRESS_ERROR),
                    (FULL + ' ' + STRESS, STRESS_ERROR),
                )):
                    xyz = base.TRAIN_XYZ.replace(VIRIAL, label, 1)
                    run_case(root / f'invalid-{n}', config + 'nep_compile 0\n', xyz, error=error)


if __name__ == '__main__':
    main()
