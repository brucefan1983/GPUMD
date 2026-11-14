#!/usr/bin/env python3
"""Compare generic and compiled spin at high spin order and high angular rank."""
import argparse
import json
import random
import re
from pathlib import Path
from validate_spin_training import maximum_error
from validate_spin_model import FIXTURE, INPUT, execute, training_xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    template = (FIXTURE/'c3l2s2.nep').read_text().split('@ parameters')[0]
    cases = json.loads((FIXTURE/'c3l2s2.json').read_text())['cases']
    xyz = ''.join(training_xyz(case) for case in cases.values())
    # FP64 autograd reference for the seeded models below: E/atom, first Fe
    # force and magnetic force, then the six symmetric virial components/atom.
    triclinic_reference = {
        3: [-.093570807, .006894585, .004953089, .004489590, .070759494, -.035416741,
            .097256480, -.014927060, -.011665854, -.003631408, .005918129, .004268369, .000251857],
        4: [-.067485760, .013377681, -.010326244, .001702646, -.042160576, .021577156,
            -.058922727, .017591940, .000237735, .012242427, -.000468419, -.000441563, .001200707],
    }
    report = {}
    for C,L,S,response,projection in [
        (2,1,4,14,62),
        (1,4,1,15,35),
        (1,3,2,16,60),
        (3,3,2,46,180),
        (3,4,2,58,237),
        (2,4,4,40,428),
    ]:
        dim = 30+response
        count = 3*(32*dim+64)+1+810+projection
        header = template
        for key,value in [('spin_compress',C),('spin_l_max',L),('spin_order',S)]:
            header = re.sub(rf'(?m)^{key} .*$', f'{key} {value}', header)
        rng = random.Random(3401)
        model = header + f'@ parameters {count}\n'
        model += ''.join(f'{rng.uniform(-.1,.1):.9g}\n' for _ in range(count))
        model += f'@ q_scaler {dim}\n' + '1\n'*dim
        dirs = []
        for jit in (0,1):
            directory = args.workdir/f'c{C}l{L}s{S}-jit{jit}'
            text = INPUT.format(C=C,L=L,S=S) + f'prediction 1\noutput_descriptor 1\nnep_compile {jit}\n'
            execute(directory, text, xyz, model)
            if jit:
                assert 'Spin aggregate JIT ready:' in (directory/'stdout.log').read_text()
            dirs.append(directory)
        errors = {}
        for file in ('energy_train.out','force_train.out','mforce_train.out',
                     'virial_train.out','descriptor.out'):
            a,b = [[float(x) for x in (d/file).read_text().split()] for d in dirs]
            error = maximum_error(a, b)
            assert error < 1e-4, (C,L,S,file,error)
            errors[file] = error
        if C == 3 and S == 2:
            frame = list(cases).index('triclinic')
            atom = sum(len(case['types']) for case in list(cases.values())[:frame])
            actual = []
            for field, row, width in [('energy',frame,1), ('force',atom,3),
                                      ('mforce',atom,3), ('virial',frame,6)]:
                lines = (dirs[1]/f'{field}_train.out').read_text().splitlines()
                actual.extend(map(float, lines[row].split()[:width]))
            errors['triclinic_FP64'] = maximum_error(actual, triclinic_reference[L])
            assert errors['triclinic_FP64'] < 2e-6, (L, actual)
        report[f'c{C}l{L}s{S}'] = errors
    (args.workdir/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
