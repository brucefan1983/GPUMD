#!/usr/bin/env python3
"""Fixed-parameter restart parity: OLD_PROBE, NEW_PROBE, OLD_NEP, BFO_CASE."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import validate_main_nep_spin3_training as base


def execute(binary, path, mode='live', error=None):
    result = subprocess.run([str(binary), mode], cwd=path, text=True, capture_output=True)
    (path/'probe.log').write_text(result.stdout+result.stderr)
    if error:
        assert result.returncode != 0 and error in result.stdout+result.stderr, result.stdout+result.stderr
        return
    assert result.returncode == 0, result.stdout+result.stderr


def values(path):
    return [float(x) for x in path.read_text().split()]


def prediction(path):
    result = {}
    for line in path.read_text().splitlines():
        name, device, *numbers = line.split()
        result.setdefault(name+':'+device, []).extend(map(float, numbers))
    return result


def compare(left, right):
    assert left.keys() == right.keys()
    errors={}
    for key in left:
        assert len(left[key]) == len(right[key])
        errors[key]=max(abs(a-b) for a,b in zip(left[key],right[key]))
        assert errors[key] <= 1e-6, (key,errors[key])
    return errors


def main():
    old=Path(os.environ['OLD_PROBE']);new=Path(os.environ['NEW_PROBE'])
    source=Path(os.environ['BFO_CASE'])
    root=Path(os.environ.get('RESTART_TEST_ROOT','restart-tests')).resolve();root.mkdir(exist_ok=True)
    config=(source/'nep.in').read_text()
    config=re.sub(r'^generation .*$', 'generation 0', config, flags=re.M)
    config=re.sub(r'^batch .*$', 'batch 2', config, flags=re.M)
    config=re.sub(r'^lambda_spin_response .*$', 'lambda_spin_response 0',config,flags=re.M)
    config=re.sub(r'^nep_compile .*\n?', '',config,flags=re.M)+'\nnep_compile 0\n'
    # Keep original model/search files; only reduce the evaluation structures.
    lines=(source/'train.xyz').read_text().splitlines();frames=[];i=0
    while i<len(lines):
        count=int(lines[i]);frames.append('\n'.join(lines[i:i+count+2])+'\n');i+=count+2
    chosen=sorted(set([0,1,len(frames)//2,len(frames)//2+1,len(frames)-2,len(frames)-1]))
    train=''.join(frames[i] for i in chosen if 0<=i<len(frames))
    def case(name, cfg=config, model=True, restart=True, xyz=train):
        path=root/name;path.mkdir()
        (path/'nep.in').write_text(cfg);(path/'train.xyz').write_text(xyz)
        if model:shutil.copyfile(source/'nep.txt',path/'nep.txt')
        if restart:shutil.copyfile(source/'nep.restart',path/'nep.restart')
        return path
    ref=case('bfo-reference');execute(old,ref,'reference')
    broken=case('bfo-old-live');execute(old,broken)
    fixed=case('bfo-new-live');execute(new,fixed)
    imported=case('bfo-new-import',config+'import_q_scaler 1\n');execute(new,imported)
    report={}
    for name in ('mu','elite'):
        report[name]=compare(prediction(ref/(name+'.pred')),prediction(fixed/(name+'.pred')))
        compare(prediction(ref/(name+'.pred')),prediction(imported/(name+'.pred')))
    assert values(fixed/'restored.restart')==values(source/'nep.restart')
    assert values(imported/'restored.restart')==values(source/'nep.restart')
    assert (ref/'baseline.txt').read_text()==(fixed/'baseline.txt').read_text()
    scales=values(ref/'scaler.txt');assert values(fixed/'scaler.txt')==scales
    assert values(imported/'scaler.txt')==scales
    report['old_scaler_changed']=sum(a!=b for a,b in zip(scales,values(broken/'scaler.txt')))
    assert report['old_scaler_changed']==0  # accepted pre-refactor baseline already restores scaling
    report['old_mu_prediction_error']={k:max(abs(a-b) for a,b in zip(v,prediction(broken/'mu.pred')[k])) for k,v in prediction(ref/'mu.pred').items()}
    report['mu_elite_difference']=max(abs(a-b) for a,b in zip(prediction(ref/'mu.pred')['E:0'],prediction(ref/'elite.pred')['E:0']))
    assert report['mu_elite_difference']>0
    # Imported scaling is also available for a new search (it does not import parameters).
    fresh_import=case('fresh-import',config+'import_q_scaler 1\n',restart=False)
    execute(new,fresh_import);assert values(fresh_import/'scaler.txt')==scales
    fresh_a=case('fresh-a',config+'seed 1\n',model=False,restart=False)
    fresh_b=case('fresh-b',config+'seed 2\n',model=False,restart=False)
    execute(new,fresh_a);execute(new,fresh_b)
    assert values(fresh_a/'scaler.txt')==values(fresh_b/'scaler.txt')
    assert values(fresh_a/'restored.restart')!=values(fresh_b/'restored.restart')
    missing=case('missing-model',model=False);execute(new,missing,error='nep.txt')
    for name,value in [('zero','0'),('nan','nan'),('truncated',None)]:
        path=case('bad-scaler-'+name);rows=(path/'nep.txt').read_text().splitlines()
        if value is None: rows=rows[:-1]
        else:rows[-1]=value
        (path/'nep.txt').write_text('\n'.join(rows)+'\n')
        execute(new,path,error='nan' if name=='nan' else 'q_scaler')
    reordered=case('reordered',re.sub(r'^type (\d+) (\S+) (\S+)', r'type \1 \3 \2', config, flags=re.M))
    execute(new,reordered,error='inconsistent')
    # Normal structural NEP: same old model/search vector, same scaler and outputs.
    structural='\n'.join(line for line in base.NEP_IN.splitlines() if not line.startswith(('spin_','lambda_m','lambda_tau')))+'\nnep_compile 0\n'
    normal=root/'normal-train';base.write_case(normal,structural,base.TRAIN_XYZ)
    execute(Path(os.environ['OLD_NEP']),normal)
    normal_cases=[]
    for name,binary in [('normal-old',old),('normal-new',new)]:
        path=root/name;shutil.copytree(normal,path)
        execute(binary,path,'reference');normal_cases.append(path)
    a,b=normal_cases
    assert values(a/'scaler.txt')==values(b/'scaler.txt')
    assert values(a/'restored.restart')==values(b/'restored.restart')
    report['structural_mu']=compare(prediction(a/'mu.pred'),prediction(b/'mu.pred'))
    report['structural_elite']=compare(prediction(a/'elite.pred'),prediction(b/'elite.pred'))
    report.update(scaler_count=len(scales),sigma_and_mu='unchanged',baseline='unchanged',fresh_scaler='deterministic',parser='passed')
    (root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
