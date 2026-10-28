#!/usr/bin/env python3
"""Same-system 1/2/4 GPU NEP-Spin checks; run inside one four-GPU allocation.

Requires numpy. Keeps inputs, per-case logs, and a JSON error ledger under --output.
The single-GPU executable is an independent pre-decomposition build when --reference is supplied.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, time
from pathlib import Path
import numpy as np

# Componentwise gates: |candidate-reference| <= atol + rtol*|reference|.
# Energy tolerance is per atom and does not scale with the elemental baseline.
GATES = {'potential': (5e-6, 0), 'force': (5e-5, 1e-5),
         'mforce': (5e-6, 1e-5), 'virial': (5e-5, 1e-5),
         'pos': (2e-5, 0), 'vel': (2e-5, 0), 'spin': (3e-5, 0)}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frame(path):
    with path.open() as f:
        while True:
            line = f.readline()
            if not line: break
            n = int(line); header = f.readline()
            props = re.search(r'Properties=([^\s]+)', header).group(1).split(':')
            raw = [f.readline().split() for _ in range(n)]
            offset = 0; result = {}
            for i in range(0, len(props), 3):
                key, kind, width = props[i], props[i+1], int(props[i+2])
                key = {'forces': 'force', 'energy_atom': 'potential'}.get(key, key)
                if kind != 'S':
                    result[key] = np.array([r[offset:offset+width] for r in raw], dtype=float)
                offset += width
    return result

def compare(left, right, keys):
    results = {}
    for key in keys:
        x, y = left[key], right[key]
        assert x.shape == y.shape
        assert np.isfinite(x).all() and np.isfinite(y).all(), key
        err = np.abs(x-y); atol, rtol = GATES[key]
        flat = int(np.argmax(err)); idx = np.unravel_index(flat, err.shape)
        result = {'max_abs': float(err.max()), 'rms': float(np.sqrt(np.mean(err**2))),
                  'max_rel_floor_1e-3': float(np.max(err / np.maximum(np.abs(y), 1e-3))),
                  'worst_index': list(map(int, idx)), 'reference': float(y[idx]),
                  'candidate': float(x[idx]), 'atol': atol, 'rtol': rtol,
                  'exceedances': int(np.sum(err > atol+rtol*np.abs(y)))}
        if key in ('potential','virial','force','mforce'):
            result['sum_delta'] = (x.sum(axis=0)-y.sum(axis=0)).tolist()
        results[key] = result
    return results

def write_xyz(path, positions, spins, cell, velocity=None, species=None):
    if velocity is None: velocity = np.zeros_like(positions)
    n = len(positions)
    with path.open('w') as f:
        f.write(str(n)+'\nLattice="'+' '.join(map(str,cell.ravel()))+'" '
                'Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3:group:I:1 pbc="T T T"\n')
        for i,(r,v,s) in enumerate(zip(positions,velocity,spins)):
            symbol = species[i] if species is not None else 'Fe'
            f.write(symbol+' '+' '.join(f'{x:.15g}' for x in np.r_[r,v,s])+f' {i%2}\n')

def structure(dims=(48,12,12), variant='bcc'):
    rng = np.random.default_rng(813)
    a = 2.8402441439
    ijk = np.indices(dims).reshape(3,-1).T
    pos = np.vstack((ijk,ijk+0.5))*a
    cell = np.diag(np.array(dims)*a)
    pos += rng.uniform(-0.012,0.012,pos.shape)
    pos %= np.diag(cell)
    if variant == 'nonuniform':
        keep = (pos[:,0] < cell[0,0]*.42) | (rng.random(len(pos)) < .32)
        pos = pos[keep]
    if variant == 'triclinic':
        frac = pos / np.diag(cell)
        cell[1,0] = 5.4; cell[2,1] = 2.8
        pos = frac @ cell
    spin = np.tile([0.,0.,2.4],(len(pos),1)) + rng.uniform(-.25,.25,pos.shape)
    velocity = rng.normal(0,.0012,pos.shape)
    velocity -= velocity.mean(axis=0)
    return pos,spin,cell,velocity

def execute(args, name, xyz, gpus, ensemble='nve', steps=1, dt=0, field='', direction='', reference=False):
    path = args.output/name; path.mkdir()
    shutil.copyfile(args.model,path/'nep.txt'); shutil.copyfile(xyz,path/'model.xyz')
    run = (f'potential nep.txt{(" "+direction) if direction else ""}\n'
           f'ensemble {ensemble}\ntime_step {dt}\n'+field+
           f'dump_xyz {steps} state.xyz precision double velocity force potential spin mforce virial\nrun {steps}\n')
    (path/'run.in').write_text(run)
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = ','.join(args.devices[:gpus])
    exe = args.reference if reference and args.reference else args.binary
    start=time.perf_counter()
    with (path/'run.log').open('w') as log:
        proc=subprocess.run([str(exe)],cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode: raise RuntimeError(f'{name}: exit {proc.returncode}: {path}/run.log')
    return frame(path/'state.xyz'),time.perf_counter()-start

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for key in ['binary','reference','model','output']:
        if getattr(a,key):setattr(a,key,getattr(a,key).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    a.devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if len(a.devices)<4:raise RuntimeError('Requires four allocated GPUs')
    report={'gates':GATES,'binary_sha256':sha(a.binary),'model_sha256':sha(a.model),
            'reference_sha256':sha(a.reference) if a.reference else None,
            'cases':{}}
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
    def record(name,candidate,ref,keys):
        result=compare(candidate,ref,keys);report['cases'][name]=result;save()
        bad={k:v for k,v in result.items() if v['exceedances']}
        if bad:raise AssertionError((name,bad))
        print(name,'PASS',flush=True)
    xyz=a.output/'input.xyz'
    outputs=['potential','force','mforce','virial']
    original_model = a.model
    model = original_model.read_text()
    converted = original_model
    # Both elements appear on every slab; use a large cell to exercise the
    # actual decomposition rather than its small-box fallback.
    pos,spin,cell,vel=structure((48,8,8))
    symbols=original_model.read_text().splitlines()[0].split()[2:]
    species=np.array([symbols[i % len(symbols)] for i in range(len(pos))])
    write_xyz(xyz,pos,spin,cell,vel,species)
    ref,_=execute(a,'format-reference',xyz,1)
    reference_line=next(line for line in model.splitlines() if line.startswith('energy_reference '))
    original=np.array(list(map(float,reference_line.split()[1:])))
    shifts=np.array([(-1)**i*(100000.123456789+i*100000.864197532)
                     for i in range(len(symbols))])
    shifted=a.output/'shifted.nep'
    shifted.write_text(model.replace(reference_line,'energy_reference '+
                       ' '.join(f'{v:.17g}' for v in original+shifts),1))
    atom_shift=np.array([shifts[i % len(symbols)] for i in range(len(pos))])[:,None]
    for g in [1,2,4]:
        a.model=converted
        out,_=execute(a,f'format-block-g{g}',xyz,g)
        record(f'format-block-g{g}',out,ref,outputs)
        if g > 1:
            repeated,_=execute(a,f'format-repeat-g{g}',xyz,g)
            record(f'format-repeat-g{g}',repeated,out,outputs)
        a.model=shifted
        shifted_out,_=execute(a,f'energy-reference-g{g}',xyz,g)
        shifted_out['potential']-=atom_shift
        record(f'energy-reference-g{g}',shifted_out,out,outputs)
        # Multi-GPU cell-list ordering varies between launches. Keep the
        # established componentwise gates above and record a repeat control;
        # the single-GPU path also supports a strict derivative check.
        if g == 1:
            for key in ['force','mforce','virial']:
                if np.max(np.abs(shifted_out[key]-out[key]))>1e-10:
                    raise AssertionError(f'Energy reference changed {key} on {g} GPUs')
    a.model=converted
    for variant in ['bcc','nonuniform','triclinic','small','cutoff']:
        if variant=='cutoff':
            # Isolated pairs just inside/outside the 6 A cutoff, including slab/PBC faces.
            cell=np.diag([144.,24.,24.]);pos=[]
            for k,x in enumerate([0.,33.,69.,105.]):
                for j,eps in enumerate([-1e-5,0,1e-5]):
                    pos.extend([[x,4+8*j,4],[((x+6+eps)%144),4+8*j,4]])
            pos=np.array(pos);spin=np.tile([.3,.4,2.35],(len(pos),1));vel=np.zeros_like(pos)
        else:
            pos,spin,cell,vel=structure((2,2,2) if variant=='small' else (48,12,12),variant)
        write_xyz(xyz,pos,spin,cell,vel)
        ref,_=execute(a,variant+'-reference',xyz,1,reference=True)
        for g in [1,2,4]:
            out,_=execute(a,f'{variant}-g{g}',xyz,g)
            record(f'{variant}-g{g}',out,ref,outputs)
        if variant=='bcc':
            for d in ['x','y','z']:
                out,_=execute(a,f'{variant}-{d}',xyz,4,direction=d)
                record(f'{variant}-{d}',out,ref,outputs)
    # Uniform translation across slab/PBC faces exceeds the cached-partition
    # displacement limit twice. Spin and force outputs must follow atom identity.
    pos,spin,cell,vel=structure((48,8,8));vel[:]=[.75,0,0]
    write_xyz(xyz,pos,spin,cell,vel)
    ref,_=execute(a,'migration-reference',xyz,1,'nve',40,.1,reference=True)
    for g in [1,2,4]:
        out,_=execute(a,f'migration-g{g}',xyz,g,'nve',40,.1)
        record(f'migration-g{g}',out,ref,outputs+['pos','vel','spin'])
        delta=out['pos']-pos;delta-=np.rint(delta/np.diag(cell))*np.diag(cell)
        assert delta[:,0].min()>2.5, 'Migration probe must exceed the partition rebuild limit'
    report['passed']=True;save()

if __name__=='__main__':main()
