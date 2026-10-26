#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Full executable GLSD lifecycle and 1/4 GPU checks (numpy, four allocated GPUs).

The analytic CUDA companion tests a confining harmonic Hamiltonian. This file
checks actual NEP/field plumbing; the bounded tanh ANN is NOT a thermal oracle.
"""
import argparse
import json
import math
import re
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from spin_fixtures import FIXTURE
from spin_analytic_model import analytic_model, structure as small_structure
from validate_spin_multigpu import frame, structure, write_xyz, sha


# Independent unit conversion in the test: retain GPUMD's established rounded
# hbar for coefficient checks. Analytic oracles below also use
# the independently tabulated hbar and tolerate its documented rounding error.
HBAR_PS = 6.465412e-2 * 1.018051e1 / 1000
DEFAULT_GAMMA = 2 / HBAR_PS


def damping(mobility, gamma=DEFAULT_GAMMA):
    return f'damping {mobility / (gamma * gamma * HBAR_PS):.17g}'


def main():
    p=argparse.ArgumentParser()
    for key in ('binary','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model', type=Path, default=FIXTURE/'nep4_spin3_o3c2_uniform.nep')
    a=p.parse_args()
    for key in ('binary','model','output'): setattr(a,key,getattr(a,key).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    assert len(devices)>=4,'four allocated GPUs required'
    report={'binary_sha256':sha(a.binary),'model_sha256':sha(a.model),'cases':{}}
    def record(name,data):
        report['cases'][name]=data
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(name,data,flush=True)
    pos,spins,cell,vel=structure((24,4,4))
    xyz=a.output/'bcc.xyz';write_xyz(xyz,pos,spins,cell,vel)
    toy=a.output/'onsite.nep';toy.write_text(analytic_model(True))
    zero=a.output/'zero.nep';zero.write_text(analytic_model(False))
    def run(name,g,ens,steps=20,dt=.1,model=None,xyz_in=None,field='',extra='',fail=False):
        case=a.output/name;case.mkdir()
        shutil.copyfile(model or a.model,case/'nep.txt')
        shutil.copyfile(xyz_in or xyz,case/'model.xyz')
        commands=f'ensemble {ens}\ntime_step {dt}\n'+field
        (case/'run.in').write_text('potential nep.txt x\n'+commands+
            'dump_xyz '+str(steps)+' state.xyz precision double velocity force potential virial spin mforce\n'+
            f'dump_restart {steps}\nrun {steps}\n'+extra)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices[:g]))
        with (case/'run.log').open('w') as log:
            result=subprocess.run([str(a.binary)],cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
        if fail:
            assert result.returncode!=0,case
            assert "Input Error" in (case/'run.log').read_text(),case
            return case
        assert result.returncode==0,(case,(case/'run.log').read_text()[-3000:])
        state=frame(case/'state.xyz')
        assert all(np.isfinite(v).all() for v in state.values()),case
        return case,state
    def difference(x,y): return {k:float(np.max(np.abs(x[k]-y[k]))) for k in ('pos','vel','spin','mforce','potential','force','virial')}
    field='add_mfield region 0 1 0 1 0 1 7 -3 4\n'
    # Check input conversion against the physical definition, without a private
    # pre-conversion binary. Analytic trajectories below independently check units.
    for name, ens in [('custom', f'nve_glsd {damping(2, 3)} gamma 3 stemp 300 seed 731'),
                      ('order', f'nve_glsd gamma 3 {damping(2, 3)} stemp 300 seed 731'),
                      ('default', 'nve_glsd damping .01 stemp -1 seed 731'),
                      ('zero', 'nve_glsd damping 0 stemp -1 seed 731')]:
        case, _ = run(name, 1, ens, field=field)
        words = ens.split()
        eta = float(words[words.index('damping') + 1])
        gamma = float(words[words.index('gamma') + 1]) if 'gamma' in words else DEFAULT_GAMMA
        expected = (HBAR_PS * gamma)**2 * (eta / HBAR_PS)
        converted = float(re.search(r'converted mobility = ([^ ]+)', (case/'run.log').read_text())[1])
        assert math.isclose(converted, expected, rel_tol=5e-15, abs_tol=0)
        record('damping-' + name, converted)
    assert difference(frame(a.output/'custom/state.xyz'), frame(a.output/'order/state.xyz'))['spin'] < 1e-12
    # Exact zero and constant-field solution through actual Force::compute.
    small=a.output/'small.xyz';small.write_text(small_structure([[0,0,0]]*4))
    for g in (1,4):
        _,z=run(f'zero-g{g}',g,f'nve_glsd {damping(2, 3)} gamma 3',100,10,zero,small)
        assert np.max(np.abs(z['spin']))==0
        _,z=run(f'zero-field-g{g}',g,f'nve_glsd {damping(2, 3)} gamma 3',100,10,zero,small,
                'add_mfield region 0 1 0 1 0 1 0 0 100\n')
        expected=2*5.7883818060e-3
        err=float(np.max(np.abs(z['spin']-[0,0,expected])))
        assert err<1e-10,err
        record(f'zero-field-g{g}',{'analytic_error':err})
    paper_initial=np.array([[1.,0.,0.]]*4)
    small.write_text(small_structure(paper_initial.tolist()))
    dt_errors=[]
    # Paper s=-S/2, gamma_s_prime=58.8 / (eV ps), hbar in eV ps.
    # E=-muB*B*S_z; exact paper-spin evolution transformed back to moments.
    hbar=6.582119569e-4
    field_energy=5.7883818060e-5*100
    duration=.02
    angle=2*field_energy*duration/hbar
    expected=np.array([math.cos(angle),math.sin(angle),4*58.8*field_energy*duration])
    for dt in (1.,.5,.25):
        _,s=run(f'paper-units-{dt}',1,f'nve_glsd {damping(235.2)}',round(1000*duration/dt),dt,
                zero,small,'add_mfield region 0 1 0 1 0 1 0 0 100\n')
        dt_errors.append(float(np.max(np.abs(s['spin']-expected))))
    # GPUMD uses its existing rounded hbar; retain its known constant mismatch.
    assert dt_errors[-1]<2e-6,dt_errors
    assert dt_errors[0]/dt_errors[1]>3 and dt_errors[1]/dt_errors[2]>3,dt_errors
    record('paper_moment_conversion',{'errors':dt_errors,'reference_hbar_eV_ps':hbar})
    # Actual ANN radial relaxation: E=tanh(.1 r^2). Independent high-accuracy
    # scalar RK4 oracle solves dr/dt=-.2 lambda r sech^2(.1 r^2).
    small.write_text(small_structure([[1,.2,0],[.4,-.3,.7],[0,0,0],[.6,.1,-.4]]))
    initial=frame(small)['spin'];r=np.linalg.norm(initial,axis=1)
    def f(x): return -.4*x/np.cosh(.1*x*x)**2
    h=.0001
    for _ in range(10000):
        k1=f(r);k2=f(r+h*k1/2);k3=f(r+h*k2/2);k4=f(r+h*k3)
        r+=h/6*(k1+2*k2+2*k3+k4)
    rotation_errors=[]
    for dt in (50,25,12.5):
        _,s=run(f'transverse-{dt}',1,f'nve_glsd {damping(2, 3)} gamma 3',int(1000/dt),dt,zero,small,
                'add_mfield region 0 1 0 1 0 1 0 0 10000\n')
        hfield=5.7883818060e-1;angle=3*hfield
        expected=initial.copy()
        expected[:,0]=initial[:,0]*math.cos(angle)-initial[:,1]*math.sin(angle)
        expected[:,1]=initial[:,0]*math.sin(angle)+initial[:,1]*math.cos(angle)
        expected[:,2]+=2*hfield
        rotation_errors.append(float(np.max(np.abs(s['spin']-expected))))
    assert all(x/y>3.5 for x,y in zip(rotation_errors,rotation_errors[1:])),rotation_errors
    record('transverse_cross_product',{'errors':rotation_errors})
    errors=[]
    for dt in (20,10,5):
        _,s=run(f'onsite-{dt}',1,f'nve_glsd {damping(2, 3)} gamma 3',int(1000/dt),dt,toy,small)
        errors.append(float(np.max(np.abs(np.linalg.norm(s['spin'],axis=1)-r))))
        assert np.sum(s['potential']) < np.sum(np.tanh(.1*np.sum(initial**2,axis=1)))
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3,errors
    record('onsite_convergence',{'errors':errors})
    # Bundled model: spin-only, coupled lattice and pressure endpoints.
    for name,ens,steps in [
        ('Fe0',f'nve_glsd {damping(10)} stemp -1 seed 731',200),
        ('FeT',f'nvt_glsd 300 300 100 lattice off {damping(10)} seed 731',200),
        ('coupled',f'nvt_glsd 300 300 100 lattice on {damping(10)} seed 731',100),
        ('npt',f'npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 {damping(10)} seed 731',20)]:
        states=[]
        for g in (1,4):
            case,s=run(f'{name}-g{g}',g,ens,steps,field=field)
            states.append(s)
            _,endpoint=run(f'{name}-endpoint-g{g}',g,'nve',1,0,
                           xyz_in=case/'restart.xyz',field=field)
            endpoint_error={k:float(np.max(np.abs(endpoint[k]-s[k])))
                            for k in ('potential','force','mforce','virial')}
            assert max(endpoint_error.values())<1e-4,endpoint_error
            record(f'{name}-endpoint-g{g}',endpoint_error)
            norms=np.linalg.norm(s['spin'],axis=1)
            assert norms.max()<5 and norms.min()>=0
            record(f'{name}-g{g}',{'norm_min':float(norms.min()),'norm_max':float(norms.max()),
                   'mean_spin':s['spin'].mean(axis=0).tolist(),'mean_norm':float(norms.mean()),
                   'norm_sd':float(norms.std()),'wrapped_position_difference':float(np.max(np.abs(s['pos']-pos)))})
        dif=difference(*states)
        assert dif['spin']<3e-5 and dif['mforce']<1e-5 and dif['force']<1e-4,dif
        record(name+'-g1-g4',dif)
    # Free Cartesian diffusion is an independent Gaussian transient oracle.
    # It is deliberately not described as an equilibrium for a zero potential.
    for g in (1,4):
        samples=[]
        for seed in (731,913,1171):
            _,s=run(f'diffusion-{seed}-g{g}',g,
                    f'nve_glsd {damping(2, 3)} gamma 3 stemp 300 seed {seed}',20,10,zero)
            samples.append(s['spin']-spins)
        values=np.vstack(samples);target=2*2*8.617343e-5*300*.2
        mean=values.mean(axis=0);var=values.var(axis=0)
        assert np.max(np.abs(mean)) < 6*np.sqrt(target/len(values)),mean
        assert np.max(np.abs(var/target-1)) < 6*np.sqrt(2/len(values)),var
        record(f'diffusion-statistics-g{g}',{'samples':len(values),'target_variance':target,
               'mean':mean.tolist(),'variance':var.tolist()})
    # Restart and consecutive run blocks: exact counter stream, unchanged order.
    for temp in (-1,300):
        ens=f'nve_glsd {damping(2, 3)} gamma 3 stemp {temp} seed 731'
        for g in (1,4):
            _,whole=run(f'whole-{temp}-g{g}',g,ens,40,1,toy)
            half,_=run(f'half-{temp}-g{g}',g,ens,20,1,toy)
            _,restart=run(f'restart-{temp}-g{g}',g,ens,20,1,toy,half/'restart.xyz')
            err=float(np.max(np.abs(whole['spin']-restart['spin'])))
            assert err<1e-12,err
            extra=f'ensemble {ens}\ntime_step 1\ndump_xyz 20 continued.xyz precision double spin\nrun 20\n'
            repeated,_=run(f'repeated-{temp}-g{g}',g,ens,20,1,toy,extra=extra)
            err2=float(np.max(np.abs(whole['spin']-frame(repeated/'continued.xyz')['spin'])))
            assert err2<1e-12,err2
            assert 'spin_vel:' not in (half/'restart.xyz').read_text().splitlines()[1]
            record(f'restart-{temp}-g{g}',{'restart_error':err,'run_block_error':err2})
    for g in (1,4):
        source=a.output/f'half-300-g{g}'/'restart.xyz'
        sib_case,_=run(f'switch-sib-g{g}',g,'nve_sib alpha 0',1,1,toy,source)
        assert 'spin_glsd_' not in (sib_case/'restart.xyz').read_text()
        glsd_case,_=run(f'switch-glsd-g{g}',g,f'nve_glsd {damping(2)} seed 913',1,1,toy,sib_case/'restart.xyz')
        metadata=(glsd_case/'restart.xyz').read_text().splitlines()[1]
        assert 'spin_glsd_step=1 ' in metadata and 'spin_glsd_seed=913 ' in metadata
        record(f'integrator-switch-g{g}',{'counter':1,'seed':913})
    invalid_ensembles = [
        'nve_sib lsf 0', 'nve_sib lsf 1', 'nve_sib damping 1',
        'nvt_sib 300 300 100 lsf 0',
        'npt_sib temp 300 300 iso 0 0 lsf 1',
        'nve_glsd', 'nve_glsd lsf 1 damping 1',
        'nve_glsd mobility 10', 'nve_glsd damping .01 mobility 10',
        'nvt_glsd 300 300 100 mobility 10',
        'npt_glsd temp 300 300 iso 0 0 mobility 10',
        'nve_glsd damping inf', 'nve_glsd damping 1e308',
        'nve_glsd damping 1 gamma 1e308', 'nve_glsd damping 1 gamma 1e-300',
        'nve_glsd damping -1', 'nve_glsd damping nan',
        'nve_glsd damping 1 alpha 0', 'nve_glsd damping 1 alpha .1',
        'nve_glsd damping 1 damping 2', 'nve_glsd damping',
        'nve_glsd damping 1 stemp 0', 'nve_glsd damping 1 gamma 0',
        'nve_glsd damping 1 seed 0', 'nve_glsd damping 1 lattice off',
        'nvt_glsd 300 300 100',
        'npt_glsd temp 300 300 iso 0 0',
    ]
    for i,ens in enumerate(invalid_ensembles):
        run(f'invalid-{i}',1,ens,1,model=toy,fail=True)
    for i,dt in enumerate((0,-1,float('nan'))):
        run(f'invalid-dt-{i}',1,'nve_glsd damping 1',1,dt,model=toy,fail=True)
    for i,metadata in enumerate(('spin_glsd_step=1', 'spin_glsd_step=-1 spin_glsd_seed=731',
         'spin_glsd_step=1 spin_glsd_seed=913',
         'spin_glsd_step=18446744073709551615 spin_glsd_seed=731')):
        malformed=a.output/f'malformed-{i}.xyz'
        lines=xyz.read_text().splitlines();lines[1]+=' '+metadata
        malformed.write_text('\n'.join(lines)+'\n')
        run(f'invalid-restart-{i}',1,'nve_glsd damping 1 seed 731',1,model=toy,xyz_in=malformed,fail=True)
    ens=f'nve_glsd {damping(2, 3)} gamma 3 seed 731'
    run('invalid-spin-velocity',1,ens,1,model=toy,fail=True,
        extra=f'ensemble {ens}\ndump_xyz 1 invalid.xyz spin_velocity\nrun 1\n')
    record('invalid_inputs',len(invalid_ensembles)+8)
    report['status']='passed';(a.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
