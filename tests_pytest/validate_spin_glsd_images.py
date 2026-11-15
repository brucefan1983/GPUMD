#!/usr/bin/env python3
"""Compare real GLSD GPU runs with accumulated images against image-free controls."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from spin_analytic_model import REFERENCE_MODEL, structure
from validate_spin_glsd_runtime import damping
from validate_spin_multigpu import GATES, frame, sha


def execute(root, binary, name, xyz, commands):
    case = root/name
    case.mkdir()
    (case/'nep.txt').write_bytes(REFERENCE_MODEL.read_bytes())
    (case/'model.xyz').write_text(xyz)
    (case/'run.in').write_text('potential nep.txt\n'+commands)
    env = os.environ.copy()
    if env.get('CUDA_VISIBLE_DEVICES'):
        env['CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES'].split(',')[0]
    result = subprocess.run([str(binary)], cwd=case, env=env, capture_output=True, text=True)
    (case/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0, (case, result.stdout[-3000:], result.stderr)
    assert 'steps accepted at iteration cap before convergence: 0 of' in result.stdout, case
    return case


def comparison(root, binary, name, skew, sign):
    cell = np.array([[20.,0.,0.],[2. if skew else 0.,20.,0.],
                     [1. if skew else 0.,1. if skew else 0.,20.]])
    lines = structure().splitlines()
    lines[1] = re.sub(r'Lattice="[^"]+"', 'Lattice="'+' '.join(map(str,cell.ravel()))+'"', lines[1])
    # One lattice vector per fs: five ordinary NVE steps accumulate +/-5 images
    # while returning near the initial wrapped configuration. No image injection.
    images_per_step = sign*np.array([[1,0,0],[-1,1,0],[0,-1,1],[1,0,-1]])
    for i, velocity in enumerate(images_per_step@cell, 2):
        row = lines[i].split(); row[4:7] = map(str,velocity); lines[i] = ' '.join(row)
    warmup = ('ensemble nve\ntime_step 1\n'
              'dump_xyz 5 warmup.xyz precision double unwrapped_position velocity spin\n'
              'dump_restart 5\nrun 5\n')
    # The deliberately fast warmup is only a way to make image history. Keep its
    # weak thermostat and subsequent deterministic probe identical in both runs.
    probe = (f'ensemble nvt_glsd 300 300 2e9 lattice on {damping(50,3)} gamma 3 '
             'stemp -1 midpoint_tol 1e-8 midpoint_max_iter 100\ntime_step .05\n'
             'dump_xyz 3 state.xyz precision double unwrapped_position_only velocity '
             'spin force mforce potential virial\nrun 3\n')
    actual = execute(root,binary,name+'-history','\n'.join(lines)+'\n',warmup+probe)
    warm = frame(actual/'warmup.xyz')
    images = np.rint((warm['unwrapped_position']-warm['pos'])@np.linalg.inv(cell)).astype(int)
    np.testing.assert_array_equal(images,5*images_per_step)
    # restart.xyz was written before GLSD, contains wrapped positions and full
    # precision velocities/spins, and intentionally carries no image history.
    control = execute(root,binary,name+'-control',(actual/'restart.xyz').read_text(),probe)
    a,b = frame(actual/'state.xyz'),frame(control/'state.xyz')
    assert all(np.isfinite(v).all() for state in (a,b) for v in state.values())
    header = (actual/'state.xyz').read_text().splitlines()[1]
    endpoint_cell = np.fromstring(re.search(r'Lattice="([^"]+)"',header)[1],sep=' ').reshape(3,3)
    a['pos'] -= images@endpoint_cell
    errors = {key:float(np.max(np.abs(a[key]-b[key])))
              for key in ('pos','vel','spin','force','mforce','potential','virial')}
    return {'images':images.tolist(),'errors':errors}


def validate(root, binary, reference=None):
    root.mkdir(parents=True,exist_ok=True)
    report = {'binary_sha256':sha(binary),'model_sha256':sha(REFERENCE_MODEL),'cases':{}}
    if reference:
        report['baseline_sha256'] = sha(reference)
    for skew in (False,True):
        for sign in (1,-1):
            name = f'{"triclinic" if skew else "orthogonal"}-{sign:+d}'
            result = comparison(root,binary,name,skew,sign)
            for key,error in result['errors'].items():
                atol = 1e-7 if key=='spin' else GATES[key][0]
                assert error<atol,(name,key,error,atol)
            report['cases'][name] = result
            if reference:
                old = comparison(root,reference,'baseline-'+name,skew,sign)
                # The old orthogonal PBC path removes only one image; the
                # triclinic public path already wraps arbitrary images.
                if not skew:
                    assert old['errors']['spin']>1e-7,(name,old)
                report.setdefault('baseline',{})[name] = old
            (root/'report.json').write_text(json.dumps(report,indent=2))
            print(name,result,flush=True)
    report['passed'] = True
    (root/'report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    validate(args.output.resolve(),args.binary.resolve(),args.reference.resolve() if args.reference else None)
