#!/usr/bin/env python3
"""Short GPU checks of every declared shape, including the maximal 339-column ANN.

Uses real NEP prediction and GPUMD CLI paths. Above the old magnetic limit,
central differences independently check the force, magnetic-force and raw9
virial derivatives of the scalar energy. Requires numpy and an allocated GPU.
"""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
from test_spin_capacity import layout_rows
from spin_force_helpers import run_case, max_error
from validate_spin_model import execute, training_xyz
from validate_spin_training import columns
import validate_spin_nosoc as non_soc


def model_input(row):
    c, l, o, soc, magnetic, projection = row
    dim = 103 + magnetic
    common = (f'model nep_spin\nspin_version 1\ncutoff 6 5\nn_max 12 8\n'
              f'basis_size 0 8\nl_max 8 2 1\nspin_basis_size 8\nspin_cutoff 6\n'
              f'spin_type Fe\nspin_compress {c}\nspin_l_max {l}\nspin_order {o}\nspin_soc {soc}\n')
    count = 4*(dim+2)+1+13+81+projection
    rng = random.Random(3401)
    parameters = [rng.uniform(-.08, .08) for _ in range(count)]
    header = ('nep 1 Fe\nformat_version 1\n'+common.replace('cutoff 6 5\n',
              'cutoff 6 5 128 128\n')+'ann 4 0\nspin_scaler 1\nenergy_reference 0\n')
    model = header + f'@ parameters {count}\n'+''.join(f'{x:.9g}\n' for x in parameters)
    model += f'@ q_scaler {dim}\n'+''.join(f'{.5+i/dim:.9g}\n' for i in range(dim))
    text = ('type 1 Fe\n'+common+'neuron 4\nlambda_m 1\nlambda_v 1\n'
            'population 10\nbatch 1\ngeneration 1\noutput_interval 1\nseed 3401\nsave_potential 1 0 0\n')
    return model, text


def check_prediction(directory, state):
    n = len(state['force'])
    errors = {'energy': abs(n*columns(directory/'energy_train.out', 1)[0][0]-state['energy'])}
    for field in ('force', 'mforce'):
        errors[field] = max_error(columns(directory/f'{field}_train.out', 3), state[field])
    v = np.array(state['virial']).reshape(3, 3)
    symmetric = [(v[i,j]+v[j,i])/2 for i,j in ((0,0),(1,1),(2,2),(0,1),(1,2),(2,0))]
    errors['virial'] = max_error([n*x for x in columns(directory/'virial_train.out', 6)[0]], symmetric)
    assert max(errors.values()) < 5e-5, (directory, errors)
    return errors


def differences(root, name, case, model, state):
    rng = np.random.default_rng(619)
    directions = {'positions': rng.uniform(-1, 1, (4, 3)),
                  'spins': rng.uniform(-1, 1, (4, 3)), 'strain': rng.uniform(-1, 1, (3, 3))}
    errors = {}
    for field, direction in directions.items():
        gradient = np.array(state[{'positions':'force','spins':'mforce','strain':'virial'}[field]])
        if field == 'strain': gradient = gradient.reshape(3, 3)
        expected = float((gradient*direction).sum())
        # Richardson cancellation permits larger steps without amplifying FP32 energy rounding.
        derivatives = {}
        for h in (2e-2, 1e-2, 5e-3):
            energies = []
            for sign in (1, -1):
                shifted = copy.deepcopy(case)
                if field == 'strain':
                    deform = np.eye(3)+sign*h*direction
                    shifted['positions'] = (np.array(case['positions'])@deform).tolist()
                    shifted['cell'] = (np.array(case['cell']).reshape(3,3)@deform).ravel().tolist()
                else:
                    shifted[field] = (np.array(case[field])+sign*h*direction).tolist()
                got = run_case(root, f'{name}-{field}-{h}-{sign}', shifted, model)
                energies.append(got['energy'])
            derivatives[h] = -(energies[0]-energies[1])/(2*h)
        for h in (2e-2, 1e-2):
            actual = (4*derivatives[h/2]-derivatives[h])/3
            error = abs(actual-expected)
            assert error < 3e-6+3e-4*abs(expected), (name, field, h, expected, actual)
            errors[f'{field}-{h}'] = {'error':error, 'expected':expected, 'actual':actual}
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    root = args.workdir.resolve(); root.mkdir(parents=True, exist_ok=False)
    rows = layout_rows(root)
    case = dict(types=['Fe']*4, cell=[24.,0.,0.,.7,25.,0.,.2,.6,26.],
                positions=[[.2,.3,.4],[2.1,.5,.7],[.8,2.4,.6],[1.8,1.9,2.2]],
                spins=[[.8,-.4,1.1],[-.5,.9,.3],[.2,.8,-.4],[-.7,.1,.2]],
                energy=0., force=[[0.,0.,0.]]*4, mforce=[[0.,0.,0.]]*4, virial=[0.]*9)
    report = {}
    for row in rows:
        c,l,o,soc,magnetic,_ = row
        name = f'c{c}l{l}o{o}-soc{soc}'
        model, text = model_input(row)
        state = run_case(root, name+'-md', case, model)
        directory = root/(name+'-prediction')
        execute(directory, text+'prediction 1\nnep_compile 0\n', training_xyz(case), model)
        errors = check_prediction(directory, state)
        if magnetic > 96:
            errors['finite_difference'] = differences(root, name, case, model, state)
        # Compile a newly admitted SOC layout and the maximal non-SOC boundary.
        if (c,l,o,soc) in ((9,2,2,1), (9,4,4,0)):
            directory = root/(name+'-jit')
            execute(directory, text+'prediction 1\nnep_compile 1\n', training_xyz(case), model)
            assert 'Spin aggregate JIT ready:' in (directory/'stdout.log').read_text()
            errors['jit'] = check_prediction(directory, state)
            directory = root/(name+'-train')
            execute(directory, text+'nep_compile 0\n', training_xyz(case))
            trained_model = (directory/'nep.txt').read_text()
            trained_state = run_case(root, name+'-trained-md', case, trained_model)
            directory = root/(name+'-trained-prediction')
            execute(directory, text+'prediction 1\nnep_compile 0\n', training_xyz(case), trained_model)
            errors['trained'] = check_prediction(directory, trained_state)
            # Also exercise the multi-image small-box implementation at these boundaries.
            small = copy.deepcopy(case); small['cell'] = [4.4,0.,0.,.7,4.8,0.,.2,.6,4.5]
            small_state = run_case(root, name+'-small-md', small, model)
            directory = root/(name+'-small-prediction')
            execute(directory, text+'prediction 1\nnep_compile 0\n', training_xyz(small), model)
            errors['small'] = check_prediction(directory, small_state)
        report[name] = errors
        (root/'report.json').write_text(json.dumps(report, indent=2))
        print(name, 'passed', flush=True)
    assert len(report)==360
    # Independent FP64 Legendre addition theorem: no native CG/layout/kernel reuse.
    non_soc.np = np
    independent = {}
    for l,o in ((3,2), (4,4)):
        oracle = non_soc.Oracle(9, l, o)
        for label,cell,spins in (
            ('large', case['cell'], case['spins']),
            ('zero', case['cell'], np.zeros((4,3))),
            ('small', [4.4,0.,0.,.7,4.8,0.,.2,.6,4.5], case['spins'])):
            name = f'legendre-c9l{l}o{o}-{label}'
            reference = oracle.case(label, np.array(case['positions']), np.array(spins),
                                    np.array(cell).reshape(3,3))
            state = run_case(root, name, reference, oracle.model())
            errors = {field:max_error(state[field], reference[field])
                      for field in ('energy','potential','force','mforce','virial')}
            assert max(errors.values())<5e-5, (name, errors)
            directory = root/(name+'-prediction')
            text = (f'type 3 Fe Bi O\nmodel nep_spin\nspin_version 1\ncutoff 6 5\n'
                    f'n_max 4 4\nbasis_size 8 8\nl_max 4 2 0\nneuron 6\n'
                    f'spin_type Fe\nspin_cutoff 6 5 4\nspin_compress 9\nspin_l_max {l}\n'
                    f'spin_order {o}\nspin_soc 0\nprediction 1\nnep_compile 0\n')
            execute(directory, text, training_xyz(reference), oracle.model())
            errors['prediction'] = check_prediction(directory, reference)
            independent[name] = errors
    (root/'independent.json').write_text(json.dumps(independent, indent=2))
    print('360 GPU shapes and 55 extended-shape directional derivative checks passed', flush=True)


if __name__ == '__main__':
    main()
