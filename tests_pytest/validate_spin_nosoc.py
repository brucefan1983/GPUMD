#!/usr/bin/env python3
"""Non-SOC v2 oracle and native integration checks.

The independent FP64 oracle uses the Legendre addition theorem, not the CUDA
spherical-harmonic/CG tables or layout metadata. Derivatives use central finite
differences of this scalar energy. NumPy is needed only to generate fixtures;
native validation consumes the frozen JSON with the Python standard library.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path


class Oracle:
    def __init__(self, channels, lmax, order, active=(True, False, False)):
        self.C, self.L, self.S = channels, lmax, order
        self.active = active
        self.cutoffs = np.array([6., 5., 4.])
        self.primitives = [(d, j) for d in range(order+1)
                           for j in range(d % 2, min(d, lmax)+1, 2)]
        self.sectors = list(itertools.product(range(lmax+1), range(min(lmax, order)+1)))
        self.paths = [(g, j, p, site) for g, j in self.sectors
                      for p, (d, rank) in enumerate(self.primitives) if rank == j
                      for site in range(2 if d else 1)]
        local = sum(d > 0 and j == 0 for d, j in self.primitives)
        self.dim = 30 + local + channels*(len(self.sectors)+1)
        self.H = 6
        rng = np.random.default_rng(3401+channels+10*lmax+100*order)
        f32 = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)
        self.mix = f32(rng.uniform(.08, .3, (5, channels)))
        self.weight = f32(rng.uniform(-.5, .5, (len(self.paths), channels)))
        self.center = f32(rng.uniform(-.3, .3, (len(self.primitives)-1, channels)))
        self.W = f32(rng.uniform(-.18, .18, (3, self.H, self.dim)))
        self.W[:, :, :30] = 0  # Isolate spin derivatives; shared lattice tested separately.
        self.b = f32(rng.uniform(-.07, .07, (3, self.H)))
        self.v = f32(rng.uniform(-.3, .3, (3, self.H)))
        self.b1 = float(f32(.021))
        self.scaler = f32(np.linspace(.45, 1.15, self.dim))
        self.coefficient = float(f32(.035))

    @staticmethod
    def legendre(degree, dot, norm_product=1.):
        # Solid-harmonic contraction: (|s||t|)^j P_j(s.t/(|s||t|)).
        # Polynomial recurrence is defined and differentiable at zero spin.
        a = np.ones_like(dot)
        if degree == 0:
            return a
        b = dot
        for j in range(1, degree):
            a, b = b, ((2*j+1)*dot*b-j*norm_product*a)/(j+1)
        return b

    def edges(self, case):
        r, cell = np.array(case['positions']), np.array(case['cell']).reshape(3, 3)
        extent = np.ceil(6*np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int)+1
        shifts = np.array(list(itertools.product(*(range(-n, n+1) for n in extent))))
        result = []
        for i in range(len(r)):
            neighbors, images = [], []
            for j in range(len(r)):
                vectors = r[j]-r[i]+shifts@cell
                keep = (np.linalg.norm(vectors, axis=1) < 6) & (np.linalg.norm(vectors, axis=1) > 1e-9)
                neighbors.extend([j]*int(keep.sum()))
                images.extend(shifts[keep])
            result.append((np.array(neighbors, dtype=int), np.array(images).reshape(-1, 3)))
        return result

    def evaluate(self, r, spin, cell, types, edges):
        spin = spin*np.array([self.active[t] for t in types])[:, None]
        s2 = (spin*spin).sum(axis=1)
        features = []
        for i, (neighbors, images) in enumerate(edges):
            dr = r[neighbors]-r[i]+images@cell
            distance = np.linalg.norm(dr, axis=1)
            cutoff = (self.cutoffs[types[i]]+self.cutoffs[types[neighbors]])*.5
            inside = distance < cutoff
            neighbors, dr, distance, cutoff = neighbors[inside], dr[inside], distance[inside], cutoff[inside]
            u = dr/distance[:, None]
            x = 2*(distance/cutoff-1)**2-1
            t0, t1 = np.ones_like(x), x
            total = (t0+1)*.5+(t1+1)*.5
            for k in range(2, 9):
                t0, t1 = t1, 2*x*t1-t0
                total += (t1+1)*.5
            radial = total*(1+np.cos(np.pi*distance/cutoff))*.5*self.coefficient
            radial = radial[:, None]*self.mix.sum(axis=0)[None, :]
            row = [s2[i]**(d//2) for d, j in self.primitives if d > 0 and j == 0]
            for g, j in self.sectors:
                amps, spins, directions = [], [], []
                for path, (gg, jj, p, site) in enumerate(self.paths):
                    if (gg, jj) != (g, j):
                        continue
                    d, _ = self.primitives[p]
                    source = spin[neighbors] if site else np.repeat(spin[i:i+1], len(neighbors), axis=0)
                    factor = (source*source).sum(axis=1)**((d-j)//2)
                    amps.extend(radial*self.weight[path][None, :]*factor[:, None])
                    spins.extend(source)
                    directions.extend(u)
                if g == 0:
                    for p, (d, rank) in enumerate(self.primitives):
                        if p and rank == j:
                            amps.append(self.center[p-1]*s2[i]**((d-j)//2))
                            spins.append(spin[i])
                            directions.append([0., 0., 1.])
                amp = np.array(amps).reshape(-1, self.C)
                sp = np.array(spins).reshape(-1, 3)
                direction = np.array(directions).reshape(-1, 3)
                if (g, j) == (0, 0):
                    row.extend(amp.sum(axis=0))
                norms = (sp*sp).sum(axis=1)
                gram = self.legendre(g, np.einsum('ai,bi->ab', direction, direction))*self.legendre(j, np.einsum('ai,bi->ab', sp, sp), norms[:, None]*norms[None, :])
                row.extend(np.einsum('ac,ab,bc->c', amp, gram, amp))
            features.append(np.r_[np.zeros(30), row])
        q = np.array(features)
        hidden = np.tanh(np.einsum('nhd,nd->nh', self.W[types], q*self.scaler)-self.b[types])
        potential = (hidden*self.v[types]).sum(axis=1)-self.b1
        return potential, q

    def case(self, name, r, spin, cell, step=2e-5):
        result = dict(name=name, types=['Fe', 'Fe', 'Bi', 'O'], positions=r.tolist(),
                      spins=spin.tolist(), cell=cell.flatten().tolist())
        edges = self.edges(result)
        types = np.array([0, 0, 1, 2])
        def energy(rr, ss, cc):
            return self.evaluate(rr, ss, cc, types, edges)[0].sum()
        potential, q = self.evaluate(r, spin, cell, types, edges)
        force, mforce, virial = np.zeros_like(r), np.zeros_like(spin), np.zeros((3, 3))
        for a in range(4):
            for d in range(3):
                change = np.zeros_like(r); change[a, d] = step
                force[a, d] = -(energy(r+change, spin, cell)-energy(r-change, spin, cell))/(2*step)
                mforce[a, d] = -(energy(r, spin+change, cell)-energy(r, spin-change, cell))/(2*step)
        for a in range(3):
            for b in range(3):
                strain = np.zeros((3, 3)); strain[a, b] = step
                plus, minus = np.eye(3)+strain, np.eye(3)-strain
                virial[a, b] = -(energy(r@plus, spin, cell@plus)-energy(r@minus, spin, cell@minus))/(2*step)
        result.update(energy=float(potential.sum()), potential=potential.tolist(),
                      descriptor=q[:, 30:].tolist(), force=force.tolist(), mforce=mforce.tolist(),
                      virial=virial.flatten().tolist())
        return result

    def model(self):
        parameters = []
        for t in range(3):
            parameters.extend(self.W[t].flatten()); parameters.extend(self.b[t]); parameters.extend(self.v[t])
        parameters += [self.b1]+[0.]*405+[self.coefficient]*405
        for array in (self.mix, self.weight, self.center):
            parameters.extend(array.flatten())
        text = f'''nep 3 Fe Bi O
format_version 1
model nep_spin
spin_version 1
cutoff 6 5 128 128
n_max 4 4
basis_size 8 8
l_max 4 2 0
ann {self.H} 0
spin_basis_size 8
spin_l_max {self.L}
spin_compress {self.C}
spin_order {self.S}
spin_soc 0
spin_scaler 1
spin_type {' '.join(t for t, active in zip(['Fe','Bi','O'],self.active) if active)}
spin_cutoff 6 5 4
energy_reference 0 0 0
@ parameters {len(parameters)}
'''
        text += ''.join(f'{x:.17g}\n' for x in parameters)
        text += f'@ q_scaler {self.dim}\n'+''.join(f'{x:.17g}\n' for x in self.scaler)
        return text


def generate(directory):
    global np
    import numpy as np
    np.seterr(invalid='raise', divide='raise', over='raise')
    directory.mkdir(parents=True, exist_ok=True)
    r = np.array([[.2,.3,.4],[2.1,.5,.7],[.8,2.4,.6],[1.8,1.9,2.2]])
    spins = np.array([[.8,-.4,1.1],[-.5,.9,.3],[.2,.8,-.4],[-.7,.1,.2]])
    cell = np.diag([24., 25., 26.])
    for C, L, S, active in [(3,2,2,(True,False,False)), (2,1,4,(True,False,False)),
                           (1,3,2,(True,False,False)), (1,4,4,(True,False,False)),
                           (1,0,2,(True,False,False)), (3,2,2,(True,True,False))]:
        name = f'c{C}l{L}s{S}'+('-Bi' if active[1] else '')
        oracle = Oracle(C,L,S,active)
        cases = [oracle.case('large',r,spins,cell), oracle.case('zero',r,np.zeros_like(spins),cell)]
        if (C,L,S)==(3,2,2):
            cases += [oracle.case('triclinic',r,spins,np.array([[4.4,0,0],[.7,4.8,0],[.2,.6,4.5]])),
                      oracle.case('isolated',r+np.arange(4)[:,None]*7,spins,np.diag([40.]*3))]
        # Two genuinely independent, non-axis-permuting rotations.
        def rotation(axis, angle):
            axis=np.array(axis);axis/=np.linalg.norm(axis)
            x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*skew
        space=rotation([2.,-1.,.7],.42);spin_rot=rotation([1.,2.,3.],.71)
        for label,R,Q in [('spin-rotation',np.eye(3),spin_rot),('space-rotation',space,np.eye(3)),('independent-rotations',space,spin_rot),('time-reversal',np.eye(3),-np.eye(3))]:
            case=copy.deepcopy(cases[0]);case['name']=label
            case['positions']=(r@R.T).tolist();case['cell']=(cell@R.T).flatten().tolist();case['spins']=(spins@Q.T).tolist()
            case['force']=(np.array(case['force'])@R.T).tolist();case['mforce']=(np.array(case['mforce'])@Q.T).tolist();case['virial']=(R@np.array(case['virial']).reshape(3,3)@R.T).flatten().tolist()
            got,q=oracle.evaluate(np.array(case['positions']),np.array(case['spins']),np.array(case['cell']).reshape(3,3),np.array([0,0,1,2]),oracle.edges(case))
            assert np.max(np.abs(got-np.array(case['potential'])))<1e-11
            assert np.max(np.abs(q[:,30:]-np.array(case['descriptor'])))<1e-10
            cases.append(case)
        if L>=1:
            changed=spins.copy();changed[0]=changed[0]@spin_rot.T
            response=oracle.case('single-spin-rotation',r,changed,cell)
            assert abs(response['energy']-cases[0]['energy'])>1e-6, name
            cases.append(response)
        refined=oracle.case('refined-fd',r,spins,cell,step=1e-5)
        fd_error=max(np.max(np.abs(np.array(refined[k])-np.array(cases[0][k]))) for k in ['force','mforce','virial'])
        assert fd_error<1e-6, (name,fd_error)
        model=oracle.model();(directory/(name+'.nep')).write_text(model)
        record=dict(C=C,L=L,S=S,neurons=oracle.H,dimension=oracle.dim,scaler=oracle.scaler.tolist(),active=list(active),cases=cases,
                    model_sha256=hashlib.sha256(model.encode()).hexdigest(),fd_step_check=float(fd_error))
        if (C,L,S) == (1,4,4):
            # High spin magnitudes in small sheared cells amplify descriptor rounding.
            record['wide_training_cases'] = []
            record['wide_training_cases'].append(oracle.case(
                'high-spin-33',
                np.array([[0.1037333, 0.2290752, 0.4014025],
                          [2.3180941, 2.149327, 2.0514403],
                          [1.0535344, 2.3821418, 3.9667498],
                          [2.1680347, 3.902569, 0.5884638]]),
                np.array([[-0.7106552, 3.0362012, 2.0539245],
                          [-1.2779339, 3.5429107, -2.76223],
                          [0.7855874, 3.6730429, -0.2681852],
                          [0.3585205, 2.9178912, 2.1735258]]),
                np.array([[3.384, 0.0, 0.0],
                          [0.564, 4.042, 0.0],
                          [0.47, 0.188, 4.512]]), step=1e-7))
            record['wide_training_cases'].append(oracle.case(
                'high-spin-53',
                np.array([[0.4081361, 0.1920992, 0.3661417],
                          [2.6938751, 2.5005866, 2.5114778],
                          [1.3622696, 2.7153532, 5.0320948],
                          [2.940769, 4.7056167, 0.8035769]]),
                np.array([[3.0464194, -2.2098789, 1.8175436],
                          [-3.7627352, 0.5329079, -0.4606473],
                          [-3.6242841, -0.315025, -1.398689],
                          [0.7481424, -0.9098108, -2.7873275]]),
                np.array([[3.96, 0.0, 0.0],
                          [0.66, 4.73, 0.0],
                          [0.55, 0.22, 5.28]]), step=1e-7))
        (directory/(name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(name,oracle.dim,len(cases),'FD consistency',fd_error,flush=True)


def check_radial_cancellation(workdir, ref, model, text):
    from spin_force_helpers import run_case
    from validate_spin_model import execute, training_xyz
    from validate_spin_training import columns

    # Route only the g=j=0 scalar to the ANN. The first three radial channels
    # contract to 2^-25; a projection of 2^25 gives a measurable force.
    lines = model.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('@ parameters')) + 1
    scale = next(i for i, line in enumerate(lines) if line.startswith('@ q_scaler'))
    params = [0.] * (scale - start)
    dim, hidden = ref['dimension'], ref['neurons']
    for t in range(3):
        offset = t * hidden * (dim + 2)
        params[offset + 32] = 1.
        params[offset + hidden * (dim + 1)] = 1.
    radial = 3 * hidden * (dim + 2) + 1
    angular = radial + 9 * 5 * 9
    for pair in range(9):
        for channel in range(3):
            params[angular + (pair * 5 + channel) * 9] = 1.
    mix = angular + 9 * 5 * 9
    params[mix:mix+3] = [1., 2.**-25, -1.]
    params[mix+5] = 2.**25
    lines[start:scale] = list(map(str, params))
    lines[scale+1:] = ['1'] * dim
    model = '\n'.join(lines) + '\n'
    case = ref['cases'][0]
    step = 1e-4
    frames = [copy.deepcopy(case) for _ in range(3)]
    for frame, delta in zip(frames, (0., step, -step)):
        frame['positions'][0][0] += delta
    def check(force, plus, minus):
        fd = -(plus - minus) / (2 * step)
        assert abs(fd) > .01 and abs(force - fd) < 2e-5, (force, fd)
        return abs(force - fd)
    md = [run_case(workdir, f'cancel-md-{i}', frame, model) for i, frame in enumerate(frames)]
    errors = {'md': check(md[0]['force'][0][0], sum(md[1]['potential']), sum(md[2]['potential']))}
    for jit in (0, 1):
        path = workdir / f'cancel-nep-{jit}'
        execute(path, text + f'prediction 1\nnep_compile {jit}\n',
                ''.join(training_xyz(frame) for frame in frames), model)
        if jit:
            assert 'Spin aggregate JIT ready' in (path / 'stdout.log').read_text()
        energies = [row[0] * len(case['types']) for row in columns(path / 'energy_train.out', 1)]
        errors[f'jit{jit}'] = check(columns(path / 'force_train.out', 3)[0][0], *energies[1:])
    return errors


def validate(fixtures, workdir):
    from validate_spin_model import execute, training_xyz
    from spin_force_helpers import run_case, max_error, flatten
    from validate_spin_training import columns, run, NEP
    workdir.mkdir(parents=True,exist_ok=True)
    report={}
    for fixture in sorted(fixtures.glob('*.json')):
        ref=json.loads(fixture.read_text());name=fixture.stem;model=fixture.with_suffix('.nep').read_text()
        assert hashlib.sha256(model.encode()).hexdigest()==ref['model_sha256']
        text=f'''model nep_spin
spin_version 1
type 3 Fe Bi O
cutoff 6 5
n_max 4 4
basis_size 8 8
l_max 4 2 0
neuron {ref['neurons']}
spin_compress {ref['C']}
spin_l_max {ref['L']}
spin_order {ref['S']}
spin_soc 0
spin_type {' '.join(t for t,a in zip(['Fe','Bi','O'],ref['active']) if a)}
spin_cutoff 6 5 4
lambda_m 1
population 10
batch 32
generation 2
output_interval 1
seed 3401
'''
        xyz=''.join(training_xyz(case) for case in ref['cases'])
        results={}
        for jit in (0,1):
            p=workdir/f'{name}-jit{jit}';execute(p,text+f'prediction 1\noutput_descriptor 2\nnep_compile {jit}\n',xyz,model)
            if jit:
                assert 'Spin aggregate JIT ready:' in (p/'stdout.log').read_text()
            energy=columns(p/'energy_train.out',1);force=columns(p/'force_train.out',3);mforce=columns(p/'mforce_train.out',3);virial=columns(p/'virial_train.out',6)
            q=columns(p/'descriptor.out',ref['dimension'])
            assert len(q)==len(force)==len(mforce)==4*len(ref['cases'])
            assert len(energy)==len(virial)==len(ref['cases'])
            errors={field:0. for field in ['energy','force','mforce','virial','descriptor','net_spin_torque']}
            for k,case in enumerate(ref['cases']):
                errors['energy']=max(errors['energy'],abs(4*energy[k][0]-case['energy']))
                for field,rows in [('force',force),('mforce',mforce)]:errors[field]=max(errors[field],max_error(rows[4*k:4*k+4],case[field]))
                errors['virial']=max(errors['virial'],max_error([4*x for x in virial[k]],[case['virial'][t] for t in (0,4,8,1,5,6)]))
                errors['descriptor']=max(errors['descriptor'],max_error([row[30:] for row in q[4*k:4*k+4]],[[x*scale for x,scale in zip(row,ref['scaler'][30:])] for row in case['descriptor']]))
                torque=[sum(s[(d+1)%3]*h[(d+2)%3]-s[(d+2)%3]*h[(d+1)%3] for s,h in zip(case['spins'],mforce[4*k:4*k+4])) for d in range(3)]
                errors['net_spin_torque']=max(errors['net_spin_torque'],max(abs(x) for x in torque))
                for atom,t in enumerate(case['types']):
                    if not ref['active'][['Fe','Bi','O'].index(t)]:assert mforce[4*k+atom]==[0.,0.,0.]
            assert all(math.isfinite(x) and x<1e-4 for x in errors.values()),(name,jit,errors)
            results[f'training_jit{jit}']=errors
        if (ref['C'], ref['L'], ref['S']) == (1, 4, 4):
            results['radial_cancellation_fd'] = check_radial_cancellation(workdir, ref, model, text)
            for case in ref.get('wide_training_cases', []):
                md = run_case(workdir, name+'-md-'+case['name'], case, model)
                for field in ('energy', 'force', 'mforce', 'virial'):
                    norm = len(case['types']) if field in ('energy', 'virial') else 1
                    actual, expected = flatten(md[field]), flatten(case[field])
                    assert len(actual) == len(expected)
                    assert all(abs(a-b)/norm <= 5e-5 + 5e-6*abs(b)/norm
                               for a,b in zip(actual, expected)), (case['name'], field)
                for jit in (0, 1):
                    p = workdir / f"{name}-{case['name']}-jit{jit}"
                    execute(p, text + f'prediction 1\nnep_compile {jit}\n', training_xyz(case), model)
                    for field in ('force', 'mforce'):
                        predicted = columns(p / f'{field}_train.out', 3)
                        assert len(predicted) == len(case[field])
                        for row, target in zip(predicted, case[field]):
                            assert all(abs(a-b) <= 5e-5 + 5e-6*abs(b) for a,b in zip(row,target)), (
                                case['name'], jit, field, row, target)
        # Independent MD path, including its force/virial accumulation and reader.
        for case in ref['cases']:
            got=run_case(workdir,name+'-md-'+case['name'],case,model)
            errors={field:max_error(got[field],case[field]) for field in ['energy','potential','force','mforce','virial']}
            assert all(math.isfinite(x) and x<1e-4 for x in errors.values()),(name,case['name'],errors)
            results['md-'+case['name']]=errors
        if name=='c3l2s2':
            p=workdir/'nosoc-training';execute(p,text+'nep_compile 1\n',xyz)
            trained=(p/'nep.txt').read_text();assert '\nspin_soc 0\n' in trained and '\nspin_version 1\n' in trained
            execute(workdir/'nosoc-trained-prediction',text+'prediction 1\n',xyz,trained)
            md=run_case(workdir,'nosoc-trained-md',ref['cases'][0],trained)
            pred=workdir/'nosoc-trained-prediction'
            assert max_error(md['energy']/4,columns(pred/'energy_train.out',1)[0][0])<1e-4
            for field in ['force','mforce']:assert max_error(md[field],columns(pred/(field+'_train.out'),3)[:4])<2e-4
            # Existing restart restores selector, parameter count, and scalers.
            import shutil
            resume=workdir/'nosoc-restart';resume.mkdir()
            for f in ['nep.in','train.xyz','nep.txt','nep.restart']:shutil.copy2(p/f,resume/f)
            process=run(NEP,resume);(resume/'stdout.log').write_text(process.stdout+process.stderr);assert process.returncode==0
            assert '\nspin_soc 0\n' in (resume/'nep.txt').read_text()
            for folder in [p,resume]:
                rows=[l for l in (folder/'loss.out').read_text().splitlines() if l.strip() and not l.startswith('#')]
                assert float(rows[-1].split()[0])==2 and all(math.isfinite(float(x)) for l in rows for x in l.split())
            # A SOC model cannot be reinterpreted as non-SOC by changing the flag.
            bad=workdir/'nosoc-reject-soc-checkpoint';bad.mkdir();(bad/'nep.in').write_text(text+'prediction 1\n');(bad/'train.xyz').write_text(xyz);(bad/'nep.txt').write_text(trained.replace('spin_soc 0','spin_soc 1'))
            process=run(NEP,bad);assert process.returncode!=0
        report[name]=results
        (workdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,'passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-fixtures',type=Path)
    parser.add_argument('--fixtures',type=Path)
    parser.add_argument('--workdir',type=Path)
    args=parser.parse_args()
    if args.generate_fixtures:generate(args.generate_fixtures)
    else:
        if args.fixtures is None or args.workdir is None:parser.error('--fixtures and --workdir are required')
        validate(args.fixtures,args.workdir)


if __name__=='__main__':main()
