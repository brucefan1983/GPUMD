#!/usr/bin/env python3
"""Independent response file parser, isolation, batching and numerical checks."""
import json
import math
import os
from pathlib import Path
import re
import tempfile
import validate_main_nep_spin3_training as base

CONFIG = base.NEP_IN.replace('batch 3', 'batch 1') + 'lambda_spin_response 0.3\nnep_compile 0\n'

def parsed(stdout):
    result = {'fit': [], 'points': []}
    for line in stdout.splitlines():
        words = line.split()
        if not words: continue
        if words[0] in ('SCALER', 'BASELINE'):
            result[words[0]] = list(map(float,words[1:]))
        if words[0] == 'FIT': result['fit'].append(list(map(float,words[1:])))
        if words[0] == 'POINT': result['points'].append((words[1],float(words[2]),float(words[3])))
    for key in ('SCALER', 'BASELINE', 'fit'):
        base.finite_values(result[key])
    for _, prediction, target in result['points']:
        base.finite_values([prediction, target])
    return result

def main():
    probe = Path(os.environ['RESPONSE_FITNESS_PROBE']).resolve()
    with tempfile.TemporaryDirectory(prefix='gpumd-response-file-') as temp:
        root = Path(temp)
        def case(name, config=CONFIG, response=base.response_xyz(), train=base.TRAIN_XYZ, test=None, error=None):
            path=root/name
            base.write_case(path,config,train)
            if response is not None: (path/'response.xyz').write_text(response)
            if test is not None: (path/'test.xyz').write_text(test)
            run=base.run(probe,path)
            if error is not None:
                assert run.returncode and error in run.stdout+run.stderr, (name,run.stdout+run.stderr)
                return
            assert run.returncode==0, (name,run.stdout+run.stderr)
            return parsed(run.stdout)
        for value in ('-1', 'nan', 'inf', '1e100'):
            case('invalid-weight-'+value,config=CONFIG.replace('lambda_spin_response 0.3','lambda_spin_response '+value),error='finite and non-negative')
        case('missing',response=None,error='requires response.xyz')
        case('empty',response='',error='rotation response frames in response.xyz')
        case('ordinary-file',response=base.TRAIN_XYZ,error='rotation response metadata')
        rows=base.response_xyz().splitlines()
        case('partial',response='\n'.join(rows[:12])+'\n',error='at least three')
        case('duplicate-coordinate',response=base.response_xyz().replace('response_coordinate=1.0','response_coordinate=0.0'),error='distinct response_coordinate')
        changed=rows.copy();changed[8]=changed[8].replace('Fe 1.0','Fe 1.5')
        case('moving',response='\n'.join(changed)+'\n',error='atomic positions')
        case('group-leak',test=base.response_xyz(),error='response_group leakage')
        case('frame-leak',test=re.sub(r' response_probe=rotation response_group=scan-a response_coordinate=\S+','',base.response_xyz()),error='Response frame leakage')
        disabled=CONFIG.replace('lambda_spin_response 0.3','lambda_spin_response 0')
        off=case('off',disabled,response=None)
        ignored=case('ignored',disabled,response='not xyz at all')
        assert off==ignored
        on=case('on')
        assert on['SCALER']==off['SCALER'] and on['BASELINE']==off['BASELINE']
        for left,right in zip(on['fit'],off['fit']):
            assert left[:-1]==right[:-1], (left,right)
            assert right[-1]==0
        values=[row[-1] for row in on['fit']]
        assert min(values)>0 and max(values)-min(values)<1e-7  # same complete groups/candidates across minibatches
        groups={}
        for group,prediction,target in on['points']: groups.setdefault(group,[]).append((prediction,target))
        huber=lambda x: .5*x*x if abs(x)<=1 else abs(x)-.5
        loss=0
        for points in groups.values():
            scale=max(math.sqrt(sum(t*t for _,t in points)/len(points)),1e-6)
            mp=sum(p for p,t in points)/len(points);mt=sum(t for p,t in points)/len(points)
            loss+=sum(huber(((p-mp)-(t-mt))/scale) for p,t in points)/len(points)+.25*huber((mp-mt)/scale)
        expected=.3*loss/len(groups)
        assert abs(values[0]-expected)<2e-6,(values[0],expected)
        full=case('full',CONFIG.replace('batch 1','batch 3'))
        assert len(full['SCALER'])==len(on['SCALER'])
        assert all(math.isclose(a,b,rel_tol=2e-6,abs_tol=1e-7) for a,b in zip(full['SCALER'],on['SCALER']))
        assert abs(full['fit'][0][-1]-values[0])<2e-6
        duplicated=case('groups-equal',response=base.response_xyz()+base.response_xyz().replace('scan-a','scan-b'))
        assert abs(duplicated['fit'][0][-1]-values[0])<2e-6
        changed=case('unused-labels',response=base.response_xyz().replace('energy=-4.0','energy=40.0').replace(' 0 0 0 1.0',' 30 40 50 1.0'))
        assert changed['BASELINE']==on['BASELINE'] and changed['SCALER']==on['SCALER'] and changed['fit']==on['fit']
        case('response-needs-energy',response=re.sub(r'energy=\S+\s*','',base.response_xyz()),error="'energy' is missing")
        for label,start,end,error in [('force',4,7,"'force' or 'forces' is missing"),('mforce',10,13,'requires mforce:R:3')]:
            stripped=[]
            for i,line in enumerate(base.response_xyz().splitlines()):
                if i%6==1:
                    line=line.replace(':'+label+':R:3','')
                elif i%6>=2:
                    fields=line.split();line=' '.join(fields[:start]+fields[end:])
                stripped.append(line)
            case('response-needs-'+label,response='\n'.join(stripped)+'\n',error=error)
        no_virial=case('no-virial',response=re.sub(r'virial="[^"]*"\s*','',base.response_xyz()))
        assert no_virial==on
        case('train-still-needs-energy',train=re.sub(r'energy=\S+\s*','',base.TRAIN_XYZ),error="'energy' is missing")
        print(json.dumps({'parser_and_leakage':'passed','lambda_zero':'passed','ordinary_fitness_isolation':'passed',
                          'minibatch_fullbatch_response':'passed','group_equal_weight':'passed','response':values[0],'oracle':expected}))

if __name__=='__main__':main()
