#!/usr/bin/env python3
"""NVT NHC endpoint restart acceptance on one and four allocated GPUs."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from validate_spin_fire_runtime import analytic_model
from validate_spin_multigpu import structure, write_xyz, sha
from validate_mttk_restart import frames


def metadata(path):
    header=path.read_text().splitlines()[1]
    state=re.search(r'nhc_state=([^\s]+)',header)
    clock=re.search(r'nhc_time=([^\s]+)',header)
    return (np.array(state[1].split(','),float) if state else None,
            float(clock[1]) if clock else None)


def main():
    p=argparse.ArgumentParser()
    for key in ('binary','reference','model','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for key in ('binary','reference','model','output'):setattr(a,key,getattr(a,key).resolve())
    a.output.mkdir(exist_ok=False,parents=True)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    assert len(devices)>=4
    report={'binary_sha256':sha(a.binary),'reference_sha256':sha(a.reference),
            'model_sha256':sha(a.model),'cases':{}}
    def record(name,value):
        report['cases'][name]=value
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(name,value,flush=True)
    pos,spins,cell,vel=structure((24,4,4))
    xyz=a.output/'bcc.xyz';write_xyz(xyz,pos,spins,cell,vel)
    zero=a.output/'zero.nep';zero.write_text(analytic_model(False))
    lj=a.output/'mechanical.lj';lj.write_text('lj 1 Fe\n.01 2.2 4.5\n')
    def run(name,g,ens,steps=80,interval=43,source=xyz,model=None,dt=.1,old=False,fail=False):
        case=a.output/name;case.mkdir()
        shutil.copyfile(source,case/'model.xyz');shutil.copyfile(model or a.model,case/'nep.txt')
        spin='glsd' in ens
        (case/'run.in').write_text('potential nep.txt'+(' x' if spin else '')+'\nensemble '+ens+f'\ntime_step {dt}\n'+
            ('add_mfield region 0 1 0 1 0 1 7 -3 4\n' if spin else '')+
            'dump_xyz 1 state.xyz precision double velocity force potential virial'+(' spin mforce' if spin else '')+'\n'+
            f'dump_restart {interval}\nrun {steps}\n')
        with (case/'run.log').open('w') as log:
            r=subprocess.run([str(a.reference if old else a.binary)],cwd=case,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices[:g]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT)
        log=(case/'run.log').read_text()
        if fail:
            assert r.returncode!=0 and 'Input Error' in log,(name,log[-1500:]);return case
        assert r.returncode==0,(name,log[-3000:])
        return case
    gates=dict(pos=2e-7,vel=2e-8,spin=3e-5,force=5e-5,mforce=5e-6,potential=5e-6,virial=5e-5,cell=2e-7,time=1e-7)
    def errors(left,right):
        assert len(left)==len(right) and len(left)>0
        return {k:max(float(np.max(np.abs(x[k]-y[k]))) for x,y in zip(left,right)) for k in gates if k in left[0]}
    def check(err,strict=False):
        assert all(v<(1e-9 if strict else gates[k]) for k,v in err.items()),err
    cases=[
        ('Fe0','nvt_glsd 300 300 100 damping .01 stemp -1 seed 731',None),
        ('FeT','nvt_glsd 300 300 100 damping .01 seed 731',None),
        ('ramp','nvt_glsd 250 450 100 damping .01 seed 731',None),
        ('separate-bath','nvt_glsd 250 450 100 damping .01 stemp 200 seed 731',None),
        ('zero-force','nvt_glsd 250 450 100 damping .01 seed 731',zero),
        ('plain','nvt_nhc 250 450 100',lj),
    ]
    for name,ens,model in cases:
        for g in (1,4):
            whole=run(f'{name}-whole-g{g}',g,ens,model=model)
            hs,ht=metadata(whole/'restart.xyz');assert hs[1]==43 and hs[2]==80 and len(hs)==28
            assert abs(ht-43*.1/10.18051)<1e-12
            resume=run(f'{name}-resume-g{g}',g,ens,37,37,whole/'restart.xyz',model)
            left=frames(whole/'state.xyz');right=frames(resume/'state.xyz')
            err=errors(left[43:],right);check(err,model==zero)
            rs,rt=metadata(resume/'restart.xyz');assert rs[1]==80 and rs[2]==80
            assert abs(rt-80*.1/10.18051)<1e-12
            assert 'Restored NVT NHC endpoint' in (resume/'run.log').read_text()
            if 'glsd' in ens:assert 'spin_glsd_step=80 ' in (resume/'restart.xyz').read_text().splitlines()[1]
            data={'trajectory_max_abs':err}
            if name in ('FeT','zero-force'):
                endpoint=run(f'{name}-endpoint-g{g}',g,ens,interval=80,model=model)
                es,et=metadata(endpoint/'restart.xyz')
                assert np.array_equal(es[:16],rs[:16]) and abs(et-rt)<1e-12
                data['chain_max_abs']=float(np.max(np.abs(es[16:]-rs[16:])))
                assert data['chain_max_abs']<(1e-9 if model==zero else 1e-6),data
            if name in ('FeT','ramp','plain'):
                old=run(f'{name}-old-g{g}',g,ens,model=model,old=True)
                data['forward_old_new']=errors(frames(old/'state.xyz'),left);check(data['forward_old_new'])
                if g==4:
                    repeat=run(f'{name}-old-repeat-g{g}',g,ens,model=model,old=True)
                    data['old_repeat']=errors(frames(old/'state.xyz'),frames(repeat/'state.xyz'));check(data['old_repeat'])
            if name=='FeT' and g==1:
                stripped=a.output/'without-nhc.xyz'
                stripped.write_text(re.sub(r'nhc_(?:state|time)=[^\s]+\s*','',(whole/'restart.xyz').read_text()))
                control=run('missing-state-control',1,ens,37,37,stripped)
                # The missing clock is irrelevant to this dynamical negative control.
                data['missing_state_control']=errors([left[-1]],[frames(control/'state.xyz')[-1]])
                assert data['missing_state_control']['vel']>1e-7 or data['missing_state_control']['pos']>1e-5
            # A second restart proves the restored protocol/counter remain serializable.
            if name=='zero-force':
                second=run(f'chained-g{g}',g,ens,20,20,whole/'restart.xyz',model)
                third=run(f'chained-second-g{g}',g,ens,17,17,second/'restart.xyz',model)
                data['chained_max_abs']=errors(left[63:],frames(third/'state.xyz'));check(data['chained_max_abs'],True)
                fs,ft=metadata(third/'restart.xyz');assert fs[1]==80 and fs[2]==80
                # Beyond the original ramp, both segmentations must hold the final T.
                extension=run(f'extend-g{g}',g,ens,10,10,resume/'restart.xyz',model)
                long=run(f'extend-long-g{g}',g,ens,47,47,whole/'restart.xyz',model)
                data['extension_max_abs']=errors(frames(long/'state.xyz')[37:],frames(extension/'state.xyz'))
                check(data['extension_max_abs'],True)
            record(name+f'-g{g}',data)
        cross=errors(frames(a.output/f'{name}-whole-g1'/'state.xyz'),frames(a.output/f'{name}-whole-g4'/'state.xyz'))
        check(cross,model==zero);record(name+'-g1-g4',cross)
    source=a.output/'FeT-whole-g1'/'restart.xyz';ens=cases[1][1]
    for label,change in [('timestep',{'dt':.2}),('coupling',{'ens':ens.replace('300 300 100','300 300 101')}),
                         ('temperature',{'ens':ens.replace('300 300','300 301')})]:
        args=dict(ens=ens,dt=.1);args.update(change)
        run('invalid-'+label,1,args['ens'],1,1,source,dt=args['dt'],fail=True)
    original=source.read_text()
    def payload(t,index,value):
        match=re.search(r'nhc_state=([^\s]+)',t);v=match[1].split(',');v[index]=str(value)
        return t[:match.start(1)]+','.join(v)+t[match.end(1):]
    modifications={
        'version':lambda t:payload(t,0,2),
        'truncated':lambda t:re.sub(r'nhc_state=[^\s]+','nhc_state=1,43,80',t),
        'nan':lambda t:payload(t,17,'nan'),
        'mass':lambda t:payload(t,16,0),
        'atom-count':lambda t:payload(t,4,769),
        'counter':lambda t:payload(t,1,1.5),
        'duplicate':lambda t:t.replace('nhc_time=','nhc_time=0 nhc_time=',1),
        'missing-time':lambda t:re.sub(r'nhc_time=[^\s]+\s*','',t),
        'trailing-comma':lambda t:re.sub(r'(nhc_state=[^\s]+)',r'\1,',t),
        'conflicting':lambda t:t.replace('nhc_state=','mttk_state=1 mttk_time=0 nhc_state=',1),
    }
    for label,fn in modifications.items():
        path=a.output/(label+'.xyz');path.write_text(fn(original))
        run('invalid-'+label,1,ens,1,1,path,fail=True)
    record('invalid_inputs',3+len(modifications))
    for g in (1,4):
        off=run(f'lattice-off-g{g}',g,ens+' lattice off',3,3,source)
        assert metadata(off/'restart.xyz')==(None,None)
        assert 'Discarding NVT NHC' in (off/'run.log').read_text()
        record(f'lattice-off-g{g}','passed')
    report['status']='passed';(a.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
