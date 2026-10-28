#!/usr/bin/env python3
"""Response input and data-isolation regressions through the real nep executable."""
import json
from pathlib import Path
import re
import tempfile
import validate_main_nep_spin3_training as base
from validate_spin_training_lifecycle import scalers, metadata

# The first generation initializes scalers; subsequent generations evaluate loss.
CONFIG = base.NEP_IN.replace('batch 3', 'batch 1').replace('generation 1', 'generation 3') + 'lambda_spin_response 0.3\nnep_compile 0\nseed 20261021\n'

def main():
    with tempfile.TemporaryDirectory(prefix='gpumd-response-file-') as temp:
        root = Path(temp)
        def case(name, config=CONFIG, response=base.response_xyz(), train=base.TRAIN_XYZ, test=None, error=None):
            path=root/name
            base.write_case(path,config,train)
            if response is not None: (path/'response.xyz').write_text(response)
            if test is not None: (path/'test.xyz').write_text(test)
            run=base.run(base.NEP,path)
            if error is not None:
                assert run.returncode and error in run.stdout+run.stderr, (name,run.stdout+run.stderr)
                return
            assert run.returncode==0, (name,run.stdout+run.stderr)
            model = (path / 'nep.txt').read_text()
            loss = [[float(x) for x in line.split()] for line in
                    (path / 'loss.out').read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
            base.finite_values(loss)
            scale = base.finite_values(scalers(model))
            return {'scaler': scale, 'metadata': metadata(model), 'loss': loss}
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
        assert on['scaler'] == off['scaler'] and on['metadata'] == off['metadata']
        assert abs(on['loss'][-1][1] - off['loss'][-1][1]) > 1e-6
        # Repeating a complete group under a new name must preserve its weight.
        duplicated=case('groups-equal',response=base.response_xyz()+base.response_xyz().replace('scan-a','scan-b'))
        assert duplicated['scaler'] == on['scaler']
        assert base.maximum_error(duplicated['loss'], on['loss']) < 2e-5
        changed=case('unused-labels',response=base.response_xyz().replace('energy=-4.0','energy=40.0').replace(' 0 0 0 1.0',' 30 40 50 1.0'))
        assert changed == on
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
        print(json.dumps({'parser_and_leakage': 'passed', 'lambda_zero': 'passed',
                          'scaler_and_reference_isolation': 'passed',
                          'response_affects_training': 'passed', 'group_equal_weight': 'passed',
                          'unused_labels_ignored': 'passed'}))


if __name__=='__main__':main()
