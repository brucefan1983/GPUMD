#!/usr/bin/env python3
# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.

"""Native spin oracle, training/prediction, restart and type-mask checks."""
import argparse
import copy
import hashlib
import json
import math
import re
import struct
from pathlib import Path

from spin_force_helpers import run_case, max_error
from validate_spin_training import run, NEP, columns, flatten

FIXTURE = Path(__file__).parent / 'fixtures/nep_spin'
INPUT = '''model nep_spin
spin_version 1
type 3 Fe Bi O
cutoff 6 5
n_max 4 4
basis_size 8 8
l_max 4 2 0
neuron 32
spin_type Fe
spin_compress {C}
spin_l_max {L}
spin_order {S}
spin_cutoff 6
lambda_m 1
lambda_v 1
population 10
seed 3401
batch 8
generation 2
output_interval 1
save_potential 1 0 0
'''


def training_xyz(case):
    n = len(case['types'])
    cell = ' '.join(map(str, case['cell']))
    virial = ' '.join(map(str, case['virial']))
    lines = [str(n), f'Lattice="{cell}" energy={case["energy"]:.17g} virial="{virial}" '
             'Properties=species:S:1:pos:R:3:spin:R:3:force:R:3:mforce:R:3']
    for i, t in enumerate(case['types']):
        values = sum([case[k][i] for k in ('positions','spins','force','mforce')], [])
        lines.append(t + ' ' + ' '.join(f'{v:.17g}' for v in values))
    return '\n'.join(lines)+'\n'


def execute(directory, text, xyz, model=None):
    directory.mkdir()
    (directory/'nep.in').write_text(text)
    (directory/'train.xyz').write_text(xyz)
    if model: (directory/'nep.txt').write_text(model)
    result = run(NEP, directory)
    (directory/'stdout.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--require-jit', action='store_true',
                        help='Require compiled spin kernels for the small oracle layouts')
    args = parser.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    report = {}
    for fixture in sorted(FIXTURE.glob('c*l*s*.json')):
        name = fixture.stem
        ref = json.loads(fixture.read_text())
        model = fixture.with_suffix('.nep').read_text()
        assert hashlib.sha256(model.encode()).hexdigest() == ref['model_sha256']
        C,L,S = map(int,re.fullmatch(r'c(\d+)l(\d+)s(\d+)',name).groups())
        text = INPUT.format(C=C,L=L,S=S)
        errors = {}
        for case_name,case in ref['cases'].items():
            got = run_case(args.workdir,name+'-'+case_name,case,model)
            delta = {k:max_error(got[k],case[k]) for k in ('energy','potential','force','mforce','virial')}
            print(name,case_name,delta,flush=True)
            for k,err in delta.items():
                scale = max(abs(x) for x in flatten(case[k]))
                assert err < 5e-5 + 3e-5*scale, (name,case_name,k,err,scale)
            errors[case_name]=delta
        case=ref['cases']['large']
        masked=copy.deepcopy(case)
        masked['spins'][2:]=[[0.,0.,0.],[0.,0.,0.]]
        a=run_case(args.workdir,name+'-masked',masked,model)
        for field in ('energy','force','mforce','virial'):
            assert max_error(a[field],case[field]) < 5e-5
        assert a['mforce'][2:] == [[0.,0.,0.],[0.,0.,0.]]
        # Prediction checks the independent training neighbor/ANN/force path.
        xyz=''.join(training_xyz(c) for c in ref['cases'].values())
        prediction=args.workdir/(name+'-prediction')
        execute(prediction,text+'prediction 1\n',xyz,model)
        if args.require_jit and L <= 2:
            assert 'Spin aggregate JIT ready:' in (prediction/'stdout.log').read_text(), name
        expected=list(ref['cases'].values())
        e=columns(prediction/'energy_train.out',2)
        errors['training_prediction_energy']=max_error([v[0] for v in e],[v['energy']/4 for v in expected])
        assert errors['training_prediction_energy'] < 5e-5
        for field in ('force','mforce'):
            rows=columns(prediction/(field+'_train.out'),6)
            err=max_error([v[:3] for v in rows],sum([v[field] for v in expected],[]))
            errors['training_prediction_'+field]=err
            assert err < 1e-4, (name,field,err)
        report[name]=errors
    # Isolate spin geometry from the ordinary structural float descriptor.
    geometry_model = (FIXTURE/'c3l2s2.nep').read_text().splitlines()
    start = next(i+1 for i,line in enumerate(geometry_model) if line.startswith('@ parameters '))
    for t in range(3):
        for neuron in range(32):
            for d in range(30):
                geometry_model[start+t*32*(64+2)+neuron*64+d] = '0'
    geometry_model = '\n'.join(geometry_model)+'\n'
    original = json.loads((FIXTURE/'c3l2s2.json').read_text())['cases']['large']
    geometry = {}
    length = 300995.893159187
    rounded = {}
    for triclinic in (False, True):
        states = {}
        for jit in (0, 1):
            for index,shift in enumerate((-1., length/2)):
                case = copy.deepcopy(original)
                case['cell'] = [length,0.,0.,2. if triclinic else 0.,20.,0.,
                                0.,1. if triclinic else 0.,20.]
                for row in case['positions']:
                    row[0] = (row[0]+shift) % case['cell'][0]
                directory = args.workdir/f'geometry-{triclinic}-{jit}-{index}'
                execute(directory, INPUT.format(C=3,L=2,S=2)+f'prediction 1\nnep_compile {jit}\n',
                        training_xyz(case), geometry_model)
                states[jit,index] = sum((sum(columns(directory/(field+'_train.out'),width), [])
                    for field,width in (('energy',2),('force',6),('mforce',6),('virial',12))), [])
                if args.require_jit and jit:
                    assert 'Spin aggregate JIT ready:' in (directory/'stdout.log').read_text()
                if not triclinic and not jit:
                    # A float-rounding negative control must fail the translation gate.
                    quantized = copy.deepcopy(case)
                    f32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
                    quantized['cell'] = list(map(f32, case['cell']))
                    quantized['positions'] = [list(map(f32, row)) for row in case['positions']]
                    quantized_dir = args.workdir/f'geometry-rounded-{index}'
                    execute(quantized_dir, INPUT.format(C=3,L=2,S=2)+'prediction 1\nnep_compile 0\n',
                            training_xyz(quantized), geometry_model)
                    rounded[index] = columns(quantized_dir/'mforce_train.out', 6)
        errors = {str(key):max_error(states[0,0],value) for key,value in states.items()}
        assert max(errors.values()) < 2e-7, (triclinic, errors)
        geometry[str(triclinic)] = errors
    geometry['float_rounding_negative_control'] = max_error(rounded[0], rounded[1])
    assert geometry['float_rounding_negative_control'] > 2e-7, geometry
    report['raw_double_geometry'] = geometry
    # Joint rotations, spatial inversion and time reversal are model symmetries.
    reference = json.loads((FIXTURE/'c3l2s2.json').read_text())
    fixed_model = (FIXTURE/'c3l2s2.nep').read_text()
    original = reference['cases']['large']
    symmetry = {}
    for label in ('joint-rotation', 'inversion', 'time-reversal'):
        transformed = copy.deepcopy(original)
        if label == 'joint-rotation':
            rotate = lambda row: [-row[1], row[0], row[2]]
            for field in ('positions', 'spins', 'force', 'mforce'):
                transformed[field] = [rotate(row) for row in original[field]]
            transformed['cell'] = sum([rotate(original['cell'][k:k+3]) for k in (0,3,6)], [])
        elif label == 'inversion':
            for field in ('positions', 'force'):
                transformed[field] = [[-v for v in row] for row in original[field]]
        else:
            for field in ('spins', 'mforce'):
                transformed[field] = [[-v for v in row] for row in original[field]]
        got = run_case(args.workdir, label, transformed, fixed_model)
        errors = {field:max_error(got[field], transformed[field])
                  for field in ('energy','potential','force','mforce')}
        assert max(errors.values()) < 5e-5, (label,errors)
        symmetry[label] = errors
    # A nonmagnetic center must retain magnetic-energy response to neighbor Fe spins.
    changed = copy.deepcopy(original)
    changed['spins'][:2] = [[0.,0.,0.],[0.,0.,0.]]
    got = run_case(args.workdir,'nonmagnetic-center-response',changed,fixed_model)
    response = [abs(got['potential'][k]-original['potential'][k]) for k in (2,3)]
    assert min(response) > 1e-6, response
    symmetry['Bi_O_energy_response'] = response
    report['symmetry'] = symmetry
    # Model selection is independent of keyword order.
    fixed_text = INPUT.format(C=3,L=2,S=2)
    reverse = fixed_text.replace('model nep_spin\nspin_version 1',
                                 'spin_version 1\nmodel nep_spin')
    execute(args.workdir/'selector-order', reverse+'prediction 1\n',
            training_xyz(original), fixed_model)
    for label, old, new in [
        ('order','spin_order 2','spin_order 5'),
        ('rank','spin_l_max 2','spin_l_max 5'),
        ('soc','spin_order 2','spin_order 2\nspin_soc 2'),
        ('basis','spin_order 2','spin_order 2\nspin_basis_size 7'),
        ('degree','spin_order 2','spin_degree 2'),
        ('initial-para','spin_order 2','spin_order 2\ninitial_para 1'),
        ('version','spin_version 1','spin_version 2'),
    ]:
        directory = args.workdir/('invalid-'+label); directory.mkdir()
        (directory/'nep.in').write_text(fixed_text.replace(old,new))
        (directory/'train.xyz').write_text(training_xyz(original))
        result = run(NEP,directory)
        (directory/'stdout.log').write_text(result.stdout+result.stderr)
        assert result.returncode != 0, label
    # Train the default 64-column model, reload, and preserve zero-variance restart.
    ref=json.loads((FIXTURE/'c3l2s2.json').read_text())
    xyz=''.join(training_xyz(c) for c in ref['cases'].values())
    text=INPUT.format(C=3,L=2,S=2)
    train=args.workdir/'training'
    execute(train,text,xyz)
    model=(train/'nep.txt').read_text()
    assert '\nspin_version 1\n' in model and '@ parameters 7270\n' in model
    assert all(math.isfinite(float(x)) for line in (train/'loss.out').read_text().splitlines()
               if not line.startswith('#') for x in line.split())
    execute(args.workdir/'trained-prediction',text+'prediction 1\n',xyz,model)
    case=ref['cases']['large']
    md=run_case(args.workdir,'trained-gpumd',case,model)
    pred=args.workdir/'trained-prediction'
    assert max_error(md['energy']/4,columns(pred/'energy_train.out',2)[0][0]) < 1e-4
    for field in ('force','mforce'):
        assert max_error(md[field],[x[:3] for x in columns(pred/(field+'_train.out'),6)[:4]]) < 2e-4
    lines=model.splitlines()
    start=next(i+1 for i,l in enumerate(lines) if l.startswith('@ parameters '))
    parameters=lines[start:start+7270]
    restart=args.workdir/'restart';restart.mkdir()
    (restart/'nep.txt').write_text(model)
    (restart/'nep.in').write_text(text)
    (restart/'train.xyz').write_text(xyz)
    reference=next(l.split(' ',1)[1] for l in lines if l.startswith('energy_reference '))
    (restart/'nep.restart').write_text('nep_restart\nenergy_reference 3 '+reference+'\nparameters 7270\n'+''.join(v+' 0\n' for v in parameters))
    result=run(NEP,restart)
    (restart/'stdout.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0,result.stdout+result.stderr
    resumed=(restart/'nep.txt').read_text()
    resumed_lines = resumed.splitlines()
    assert resumed_lines[:start] == lines[:start], 'Restart changed model metadata'
    assert resumed_lines[start+7270:] == lines[start+7270:], 'Restart changed scalers'
    restart_error = max_error(list(map(float,parameters)),
                              list(map(float,resumed_lines[start:start+7270])))
    # SNES recombination can move a zero-variance mean by one FP32 ULP.
    assert restart_error < 1e-7, restart_error
    report['training']={'parameters':7270,'dimension':64,'restart_max_abs':restart_error,
                        'restart_scalers':'identical','generations':2}
    (args.workdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Spin native runtime, training, prediction and restart passed',flush=True)

if __name__ == '__main__': main()
