#!/usr/bin/env python3
"""Spin lattice MTTK checkpoint and ordinary-driver isolation acceptance: uninterrupted vs separate-process resume.

Requires four allocated GPUs. Uses the actual Fe potential, an independent
zero-force fixture, old-binary forward regression and a missing-state control.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from spin_analytic_model import analytic_model
from validate_spin_multigpu import frame, structure, write_xyz, sha


def metadata(path):
    header=path.read_text().splitlines()[1]
    cell=np.array(re.search(r'Lattice="([^"]+)"',header)[1].split(),float)
    payload=re.search(r'mttk_state=([^\s]+)',header)
    state=np.array(payload[1].split(','),float) if payload else None
    clock=re.search(r'mttk_time=([^\s]+)',header)
    return cell,state,float(clock[1]) if clock else None


def frames(path):
    lines=path.read_text().splitlines();n=int(lines[0]);result=[]
    # Keep all post-checkpoint steps, not just the endpoint.
    for start in range(0,len(lines),n+2):
        h=lines[start+1];props=re.search(r'Properties=([^\s]+)',h)[1].split(':')
        raw=[line.split() for line in lines[start+2:start+2+n]];offset=0;v={}
        for i in range(0,len(props),3):
            key,kind,width=props[i],props[i+1],int(props[i+2]);key={'forces':'force','energy_atom':'potential'}.get(key,key)
            if kind!='S':v[key]=np.array([r[offset:offset+width] for r in raw],float)
            offset+=width
        v['cell']=np.array(re.search(r'Lattice="([^"]+)"',h)[1].split(),float)
        v['time']=np.array([float(re.search(r'Time=([^\s]+)',h)[1])]);result.append(v)
    return result

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
    def record(name,result):
        report['cases'][name]=result
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(name,result,flush=True)
    pos,spins,cell,vel=structure((24,4,4))
    xyz=a.output/'bcc.xyz';write_xyz(xyz,pos,spins,cell,vel)
    tri=a.output/'tri.xyz';tc=cell.copy();tc[1,0]=.4;tc[2,1]=.2
    write_xyz(tri,(pos@np.linalg.inv(cell))@tc,spins,tc,vel)
    zero=a.output/'zero.nep';zero.write_text(analytic_model(False))
    # Ordinary MTTK ensembles deliberately reject NEP_Spin. This mechanical
    # fixture is only an integration compatibility check, not a model of Fe.
    lj=a.output/'mechanical.lj';lj.write_text('lj 1 Fe\n.01 2.2 4.5\n')
    field='add_mfield region 0 1 0 1 0 1 7 -3 4\n'
    def run(name,g,ens,steps,interval,source=xyz,model=None,dt=.1,old=False,fail=False):
        case=a.output/name;case.mkdir()
        shutil.copyfile(source,case/'model.xyz');shutil.copyfile(model or a.model,case/'nep.txt')
        spin='glsd' in ens
        (case/'run.in').write_text('potential nep.txt'+(' x' if spin else '')+'\nensemble '+ens+f'\ntime_step {dt}\n'+(field if spin else '')+
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
    def difference(x,y):
        return {k:float(np.max(np.abs(x[k]-y[k]))) for k in ('pos','vel','spin','force','mforce','potential','virial') if k in x}
    cases=[
        ('iso0','npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 damping .01 stemp -1 seed 731',80,43,xyz,None),
        ('isoT','npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 damping .01 seed 731',80,43,xyz,None),
        ('ramp','npt_glsd temp 250 450 iso 0 .2 tperiod 100 pperiod 1000 damping .01 seed 731',80,43,xyz,None),
        ('tri','npt_glsd temp 300 300 tri 0 0 tperiod 100 pperiod 1000 damping .01 seed 731',80,43,tri,None),
        ('deviatoric','npt_glsd temp 300 350 x 0 .1 y .1 .2 z .2 .3 xy .02 .03 tperiod 100 pperiod 1000 damping .01 seed 731',80,43,tri,None),
        ('plain','npt_mttk temp 300 300 iso 0 0 tperiod 100 pperiod 1000',80,43,xyz,lj),
        ('nvt-mttk','nvt_mttk temp 300 350 tperiod 100',80,43,xyz,lj),
        ('nph-mttk','nph_mttk iso 0 0 pperiod 1000',80,43,xyz,lj),
        ('zero-force','npt_glsd temp 300 350 iso 0 0 tperiod 100 pperiod 1000 damping .01 seed 731',80,43,xyz,zero),
        ('reference-reset','npt_glsd temp 300 300 x 0 .001 y 0 .002 z 0 .003 tperiod 100 pperiod 1000 damping .01 seed 731',1007,997,tri,zero),
    ]
    # A restart recomputes forces and may rebuild atomically ordered neighbor
    # lists. Use the pre-existing NEP regression error scale, plus old/old controls.
    gates=dict(pos=2e-7,vel=2e-8,spin=3e-5,force=5e-5,mforce=5e-6,potential=5e-6,virial=5e-5,cell=2e-7,time=1e-7)
    for name,ens,total,split,source,model in cases:
        for g in (1,4):
            if name=='reference-reset' and g==4:continue
            whole=run(f'{name}-whole-g{g}',g,ens,total,split,source,model)
            if 'glsd' not in ens:
                assert metadata(whole/'restart.xyz')[1:] == (None, None)
                old=run(f'{name}-old-g{g}',g,ens,total,split,source,model,old=True)
                data={'forward_old_new':difference(frame(old/'state.xyz'),frame(whole/'state.xyz'))}
                assert all(data['forward_old_new'][k]<gates[k] for k in data['forward_old_new']),data
                legacy=run(f'{name}-legacy-g{g}',g,ens,3,3,old/'restart.xyz',model)
                assert metadata(legacy/'restart.xyz')[1:] == (None, None)
                assert 'Restored MTTK' not in (legacy/'run.log').read_text()
                assert abs(frames(legacy/'state.xyz')[0]['time'][0]-.1) < 1e-7
                record(name+f'-g{g}',data)
                continue
            # total<2*split: restart.xyz is the actual middle checkpoint.
            hc,hs,ht=metadata(whole/'restart.xyz');assert hs is not None and hs[1]==split and hs[2]==total
            resume=run(f'{name}-resume-g{g}',g,ens,total-split,total-split,whole/'restart.xyz',model)
            left=frames(whole/'state.xyz')[split:];right=frames(resume/'state.xyz');assert len(left)==len(right)
            errors={k:max(float(np.max(np.abs(x[k]-y[k]))) for x,y in zip(left,right)) for k in gates if k in left[0]}
            local_gates={k:1e-9 for k in gates} if model==zero else gates
            assert all(errors[k]<local_gates[k] for k in errors),(name,g,errors)
            rc,rs,rt=metadata(resume/'restart.xyz');assert rs[1]==total and rs[2]==total
            assert abs(rt-total*.1/10.18051)<1e-12
            assert 'Restored MTTK endpoint' in (resume/'run.log').read_text()
            if 'glsd' in ens:
                header=(resume/'restart.xyz').read_text().splitlines()[1]
                assert f'spin_glsd_step={total} ' in header
            data={'trajectory_max_abs':errors,'restored_step':int(hs[1]),'final_step':int(rs[1])}
            if name in ('isoT','ramp','plain'):
                old=run(f'{name}-old-g{g}',g,ens,total,split,source,model,old=True)
                data['forward_old_new']=difference(frame(old/'state.xyz'),frame(whole/'state.xyz'))
                assert all(data['forward_old_new'][k]<gates[k] for k in data['forward_old_new']),data
                legacy=run(f'{name}-old-checkpoint-g{g}',g,ens,total-split,total-split,old/'restart.xyz',model)
                pairs=list(zip(frames(old/'state.xyz')[split:],frames(legacy/'state.xyz')))
                data['old_checkpoint_resume']={k:max(float(np.max(np.abs(x[k]-y[k]))) for x,y in pairs) for k in gates}
                assert all(data['old_checkpoint_resume'][k]<gates[k] for k in gates),data
                if g==4:
                    repeat=run(f'{name}-old-repeat-g{g}',g,ens,total,split,source,model,old=True)
                    data['old_repeat']=difference(frame(old/'state.xyz'),frame(repeat/'state.xyz'))
                    assert all(data['old_repeat'][k]<gates[k] for k in data['old_repeat']),data
            if name=='isoT' and g==1:
                stripped=a.output/'without-mttk.xyz'
                text=(whole/'restart.xyz').read_text()
                text=re.sub(r'mttk_state=[^\s]+\s*','',text);text=re.sub(r'mttk_time=[^\s]+\s*','',text)
                stripped.write_text(text)
                control=run('missing-state-control',1,ens,total-split,total-split,stripped)
                data['missing_state_control']=difference(frame(control/'state.xyz'),frame(whole/'state.xyz'))
                assert data['missing_state_control']['vel']>1e-7 or data['missing_state_control']['pos']>1e-5,data
            record(name+f'-g{g}',data)
    source=a.output/'isoT-whole-g1'/'restart.xyz';ens=cases[1][1]
    for label,change in [('timestep',{'dt':.2}),('tperiod',{'ens':ens.replace('tperiod 100','tperiod 101')}),
                         ('pressure',{'ens':ens.replace('iso 0 0','iso .1 .1')}),('coupling',{'ens':ens.replace('iso','aniso')})]:
        args=dict(ens=ens,dt=.1);args.update(change)
        run('invalid-'+label,1,args['ens'],1,1,source,dt=args['dt'],fail=True)
    original=source.read_text()
    modifications={
        'version':lambda t:t.replace('mttk_state=1,','mttk_state=2,',1),
        'truncated':lambda t:re.sub(r'mttk_state=[^\s]+','mttk_state=1,43,80',t),
        'nan':lambda t:t.replace('mttk_state=1,','mttk_state=nan,',1),
        'duplicate':lambda t:t.replace('mttk_time=','mttk_time=0 mttk_time=',1),
        'missing-time':lambda t:re.sub(r'mttk_time=[^\s]+\s*','',t),
        'trailing-comma':lambda t:re.sub(r'(mttk_state=[^\s]+)',r'\1,',t),
    }
    for label,fn in modifications.items():
        path=a.output/(label+'.xyz');path.write_text(fn(original))
        run('invalid-'+label,1,ens,1,1,path,fail=True)
    record('invalid_inputs',4+len(modifications))
    report['status']='passed';(a.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
