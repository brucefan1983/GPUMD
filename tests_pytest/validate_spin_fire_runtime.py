#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spin FIRE analytic and endpoint regression, without third-party packages."""
from spin_analytic_model import analytic_model, structure
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

FIXTURE = Path(__file__).parent / 'fixtures' / 'nep_spin3'
GPUMD = Path(os.environ.get('GPUMD_COMMAND', Path(__file__).parents[1] / 'src/gpumd')).resolve()


def norm(v):
    return math.sqrt(sum(x*x for x in v))






def read_frame(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    fields = re.search(r'Properties=([^\s]+)', lines[-n-1]).group(1).split(':')
    rows = [x.split() for x in lines[-n:]]
    result, offset = {}, 0
    for name, kind, width in zip(fields[::3], fields[1::3], fields[2::3]):
        width = int(width)
        result[name] = [r[offset:offset+width] if kind == 'S' else list(map(float,r[offset:offset+width])) for r in rows]
        offset += width
    return result


def run_case(root, name, potential, model, commands, dump=True, single_gpu=False):
    case = root / name
    case.mkdir()
    (case/'nep.txt').write_text(potential)
    (case/'model.xyz').write_text(model)
    suffix = ('ensemble nve\ntime_step 0\ndump_xyz 1 state.xyz precision double velocity force potential virial spin mforce spin_velocity unwrapped_position\nrun 1\n' if dump else '')
    (case/'run.in').write_text('potential nep.txt\n'+commands+suffix)
    env = os.environ.copy()
    if single_gpu and env.get('CUDA_VISIBLE_DEVICES'):
        env['CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES'].split(',')[0]
    run = subprocess.run([str(GPUMD)], cwd=case, env=env, capture_output=True, text=True)
    (case/'stdout').write_text(run.stdout+run.stderr)
    return case, run


def check_run(case, run, tolerance=None, converged=True):
    assert run.returncode == 0, (case, run.stdout[-3000:], run.stderr)
    records = [(int(s),float(e),float(g),float(dt)) for s,e,g,dt in re.findall(
        r'spin_fire step (\d+) energy (\S+) max_mforce (\S+) dt (\S+)', run.stdout)]
    assert records, case
    assert all(math.isfinite(v) for r in records for v in r), case
    if converged:
        assert 'Spin FIRE finished: converged;' in run.stdout, (case,records[-1])
        assert records[-1][2] <= tolerance, (case, records[-1])
    state = read_frame(case/'state.xyz') if (case/'state.xyz').exists() else None
    if state:
        original = read_frame(case/'model.xyz')
        for field in ('pos','vel','spin_vel'):
            if field in original:
                assert max(abs(a-b) for x,y in zip(state['unwrapped_position' if field == 'pos' else field],original[field]) for a,b in zip(x,y)) < 1e-8, (case,field)
    return {'initial_energy':records[0][1], 'final_energy':records[-1][1],
            'initial_residual':records[0][2], 'final_residual':records[-1][2],
            'steps':records[-1][0], 'status':re.findall(r'Spin FIRE finished: (\w+)',run.stdout)[-1]}, state


def validate(root):
    summary = {}
    spins = [[1,0.2,0], [0.4,-0.3,0.7], [0,0,0], [0.6,0.1,-0.4]]
    model = structure(spins)
    field = 'add_mfield region 0 1 0 1 0 1 0 0 10000\n'
    case, run = run_case(root,'fixed_zeeman',analytic_model(False),model,
                         field+'minimize spin_fire 1e-7 3000 0\n')
    summary['fixed_zeeman'], state = check_run(case,run,1e-7)
    for before, after in zip(spins,state['spin']):
        assert abs(norm(before)-norm(after)) < 2e-8
        assert abs(after[2]-norm(before)) < 2e-7
    expected = -5.7883818060e-5*10000*sum(map(norm,spins))
    assert abs(summary['fixed_zeeman']['final_energy']-expected)<1e-9

    case,run = run_case(root,'variable_onsite',analytic_model(True),structure(),
                        'minimize spin_fire 1e-6 3000 1\n')
    summary['variable_onsite'],state = check_run(case,run,1e-6)
    assert max(map(norm,state['spin'])) < 6e-6
    initial = read_frame(case/'model.xyz')['spin']
    assert abs(summary['variable_onsite']['initial_energy']-sum(math.tanh(.1*norm(s)**2) for s in initial))<2e-7

    case,run = run_case(root,'zero_variable',analytic_model(True),structure([[0,0,0]]*4),
                        'add_mfield region 0 1 0 1 0 1 0 0 100\nminimize spin_fire 2e-6 3000 1\n')
    summary['zero_variable'],state = check_run(case,run,2e-6)
    for s in state['spin']:
        assert s[2] > .02
        assert abs(.2*s[2]/math.cosh(.1*s[2]**2)**2-5.7883818060e-3)<3e-6

    # A constant energy reference must not affect the FIRE trajectory or endpoint.
    shifted = analytic_model(True).replace('spin_baseline 0 0', 'spin_baseline 1024 1024')
    case,run = run_case(root,'shifted_onsite',shifted,structure(),
                        'minimize spin_fire 1e-6 3000 1\n')
    summary['shifted_onsite'],shifted_state = check_run(case,run,1e-6)
    original = read_frame(root/'variable_onsite/state.xyz')
    assert shifted_state['spin'] == original['spin']
    for key in ('steps', 'status', 'initial_residual', 'final_residual'):
        assert summary['shifted_onsite'][key] == summary['variable_onsite'][key]
    assert abs(summary['shifted_onsite']['final_energy'] - summary['variable_onsite']['final_energy'] - 4096) < 1e-10

    # Nonzero Zeeman force below representable spin updates must report a stall,
    # not convergence or an exhausted iteration budget.
    case,run = run_case(root,'unresolved_updates',analytic_model(False),structure([[0,0,1]]*4),
                        'add_mfield region 0 1 0 1 0 1 0 0 1e-12\nminimize spin_fire 1e-18 3000 1\n')
    summary['unresolved_updates'],state = check_run(case,run,converged=False)
    assert summary['unresolved_updates']['status'] == 'stalled'
    assert 'stagnation unresolved_spin_updates' in run.stdout
    assert summary['unresolved_updates']['final_residual'] > 1e-18
    assert state['spin'] == [[0,0,1]]*4

    # A pure Zeeman potential with a free magnitude has no stationary point:
    # spins keep moving and energy falls, but the force cannot converge.
    case,run = run_case(root,'residual_plateau',analytic_model(False),model,
                        'add_mfield region 0 1 0 1 0 1 0 0 100\nminimize spin_fire 1e-6 1500 1\n')
    summary['residual_plateau'],state = check_run(case,run,converged=False)
    record = summary['residual_plateau']
    assert record['status'] == 'stalled' and record['steps'] < 1500
    assert 'stagnation no_residual_progress' in run.stdout
    assert record['final_energy'] < record['initial_energy']
    assert record['final_residual'] > 1e-6
    assert state['spin'][0][2] > spins[0][2]+1

    case,run = run_case(root,'fixed_radial',analytic_model(True),structure(),
                        'minimize spin_fire 1e-7 10 0\nminimize spin_fire 1e-7 10 0\n')
    summary['fixed_radial'],state = check_run(case,run,1e-7)
    assert summary['fixed_radial']['steps']==0
    assert max(map(norm,state['mforce']))>.1

    # Joint modes reduce to the known onsite problem when the potential has no
    # position dependence. Zero cell gradient must not introduce cell drift.
    for suffix in ('0', '0 0', '0 1', '0 1 0', '0 1 1'):
        name='joint_onsite_'+suffix.replace(' ','_')
        case,run = run_case(root,name,analytic_model(True),structure(),
                            'minimize spin_fire 1e-6 3000 1 '+suffix+'\n')
        summary[name],state = check_run(case,run,1e-6)
        assert max(map(norm,state['spin'])) < 6e-6
        assert 'max_force 0.000000000e+00 max_stress_GPa 0.000000000e+00' in run.stdout
        before=re.search(r'Lattice="([^"]+)"',(case/'model.xyz').read_text()).group(1).split()
        after=re.search(r'Lattice="([^"]+)"',(case/'state.xyz').read_text()).group(1).split()
        assert list(map(float,before)) == list(map(float,after))

    # Preserve unwrapped coordinates both when FIRE creates the image arrays
    # and when an earlier run has already populated them.
    shifted=structure().splitlines()
    for i in range(2,len(shifted)):
        row=shifted[i].split(); row[1]=str(float(row[1])+40); row[2]=str(float(row[2])-20)
        shifted[i]=' '.join(row)
    for existing_images in (False,True):
        prefix=('ensemble nve\ntime_step 0\ndump_xyz 1 before.xyz unwrapped_position\nrun 1\n'
                if existing_images else '')
        case,run=run_case(root,'joint_images_'+str(int(existing_images)),analytic_model(True),
                          '\n'.join(shifted)+'\n',prefix+'minimize spin_fire 1e-6 3000 1 0\n')
        summary[case.name],_=check_run(case,run,1e-6)

    # First-step oracle from independently dumped force, magnetic force and
    # nonsymmetric virial. At zero initial fictitious velocity, FIRE mixing
    # leaves v=dt*g; this checks simultaneous Cartesian and affine updates.
    potential=(FIXTURE/'nep4_spin3_o3c2_uniform.nep').read_text()
    skew=structure().replace('20 0 0 0 20 0 0 0 20','20 0 0 2 20 0 0 1 20')
    case,run=run_case(root,'joint_reference',potential,skew,'')
    assert run.returncode==0,run.stderr
    ref=read_frame(case/'state.xyz')
    h=[[20.,2.,0.],[0.,20.,1.],[0.,0.,20.]]
    w=[sum(row[k] for row in ref['virial']) for k in range(9)]
    strain=[[.01*w[3*j+i]/400 for j in range(3)] for i in range(3)]
    assert norm([v for row in strain for v in row])<.01
    assert 1e-10 < max(map(norm,ref['forces'])) < 10
    assert max(map(norm,ref['mforce'])) < 10
    case,run=run_case(root,'joint_one_step',potential,skew,'minimize spin_fire 1e-14 1 1 0 1\n')
    assert run.returncode==0,run.stderr
    final=read_frame(case/'state.xyz')
    errors={'position':0.,'spin':0.,'cell':0.}
    for p,f,s,g,q,t in zip(ref['unwrapped_position'],ref['forces'],ref['spin'],ref['mforce'],
                         final['unwrapped_position'],final['spin']):
        expected=[p[i]+.01*f[i]+sum(strain[i][j]*p[j] for j in range(3)) for i in range(3)]
        errors['position']=max(errors['position'],max(abs(x-y) for x,y in zip(expected,q)))
        errors['spin']=max(errors['spin'],max(abs(s[i]+.01*g[i]-t[i]) for i in range(3)))
    cell=list(map(float,re.search(r'Lattice="([^"]+)"',(case/'state.xyz').read_text()).group(1).split()))
    errors['cell']=max(abs(cell[3*j+i]-h[i][j]-sum(strain[i][k]*h[k][j] for k in range(3)))
                       for i in range(3) for j in range(3))
    assert max(errors.values())<1e-8,errors
    summary['joint_one_step_oracle']=errors

    # Explicit fixed_atoms=0 must retain the joint update, including cell motion.
    joint_cell=cell
    case,run=run_case(root,'explicit_moving_atoms',potential,skew,
                      'minimize spin_fire 1e-14 1 1 0 1 0\n')
    assert run.returncode==0,(case,run.stderr)
    assert read_frame(case/'state.xyz')==final
    cell=list(map(float,re.search(r'Lattice="([^"]+)"',(case/'state.xyz').read_text()).group(1).split()))
    assert cell==joint_cell
    summary['explicit_moving_atoms']={'matches_joint_one_step':True}

    # Nonzero atomic forces make this a meaningful freeze test. Both magnitude
    # modes must match the legacy spin-only path and preserve the skew cell.
    for mode in (0,1):
        endpoints=[]
        records=[]
        for suffix in ('', ' 1', ' 1 0', ' 1 0 0'):
            name='fixed_atoms_'+str(mode)+'_'+str(len(endpoints))
            case,run=run_case(root,name,potential,skew,
                              'minimize spin_fire 1e-14 1 '+str(mode)+suffix+'\n')
            record,state=check_run(case,run,converged=False)
            endpoints.append(state)
            records.append(record)
            cell=list(map(float,re.search(r'Lattice="([^"]+)"',(case/'state.xyz').read_text()).group(1).split()))
            assert cell==[h[i][j] for j in range(3) for i in range(3)]
            assert max(abs(a-b) for s,t in zip(ref['spin'],state['spin']) for a,b in zip(s,t))>1e-12
        assert all(state==endpoints[0] for state in endpoints[1:])
        assert all(record==records[0] for record in records[1:])
        summary['fixed_atoms_'+str(mode)]=dict(records[1],matches_legacy=True)

    case,run = run_case(root,'max_steps',analytic_model(False),model,
                        field+'minimize spin_fire 1e-7 1 0\n')
    summary['max_steps'],_ = check_run(case,run,converged=False)
    assert summary['max_steps']['status']=='maximum_steps'

    for i,command in enumerate(('minimize spin_fire -1 20 0', 'minimize spin_fire 1e-5 0 0',
                                'minimize spin_fire 1e-5 20 wrong', 'minimize spin_fire 1e-5 20',
                                'minimize spin_fire nan 20 0', 'minimize fire 1e-5 20',
                                'minimize spin_fire 1e-5 20 2', 'minimize spin_fire 1e-5 20 -1',
                                'minimize spin_fire 1e-5 20 fixed', 'minimize spin_fire 1e-5 20 variable',
                                'minimize spin_fire 1e-5 20 1 2', 'minimize spin_fire 1e-5 20 1 1 2',
                                'minimize spin_fire 1e-5 20 1 1 1', 'minimize spin_fire 1e-5 20 1 1 0 0 0',
                                'minimize spin_fire 1e-5 20 1 0 0 2', 'minimize spin_fire 1e-5 20 1 0 0 -1',
                                'minimize spin_fire 1e-5 20 1 0 0 wrong', 'minimize spin_fire 1e-5 20 1 1 0 1',
                                'minimize spin_fire 1e-5 20 1 0 0 1', 'minimize spin_fire 1e-5 20 1 0 2 0',
                                'minimize spin_fire 1e-5 20 1 -1', 'minimize spin_fire 1e-5 20 1 wrong')):
        case,run = run_case(root,'invalid_'+str(i),analytic_model(False),model,command+'\n',False)
        assert run.returncode != 0, case
    case,run=run_case(root,'joint_nonperiodic',analytic_model(True),
                      model.replace('pbc="T T T"','pbc="T F T"'),
                      'minimize spin_fire 1e-5 20 1 0 1\n',False)
    assert run.returncode != 0 and 'periodic boundaries' in run.stderr

    # Ordinary lattice FIRE remains available with a non-spin potential.
    # Its upstream multi-GPU backend does not support this small periodic cell.
    fixtures = FIXTURE.parent
    case, run = run_case(root, 'lattice_fire',
                        (fixtures/'models/nep_C.txt').read_text(),
                        (fixtures/'structures/C-nat16-rattled.xyz').read_text(),
                        'minimize fire 1e-3 20\n', False, single_gpu=True)
    assert run.returncode == 0, (case, run.stdout[-2000:])
    case, run = run_case(root, 'spin_fire_without_spin',
                        (fixtures/'models/nep_C.txt').read_text(),
                        (fixtures/'structures/C-nat16-rattled.xyz').read_text(),
                        'minimize spin_fire 1e-3 20 0\n', False, single_gpu=True)
    assert run.returncode != 0
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='spin-fire-') as temporary:
        root = args.output or Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        summary = validate(root)
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
