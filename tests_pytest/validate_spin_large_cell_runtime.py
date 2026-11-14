#!/usr/bin/env python3
"""Translation invariance across a 3000 A periodic boundary, including force paths."""
import argparse
import json
from pathlib import Path
import tempfile
from validate_spin_fire_runtime import FIXTURE, read_frame, run_case
from spin_analytic_model import structure


def validate(root):
    potential = (FIXTURE / 'c3l2s2_fege.nep').read_text()
    report = {}
    length = 3009.95893159187
    for triclinic,zbl in ((False,False),(True,False),(False,True),(True,True)):
        states = []
        for shift in (-1., length/2):
            lines = structure().splitlines()
            cell = f'{length} 0 0 {2 if triclinic else 0} 20 0 0 {1 if triclinic else 0} 20'
            lines[1] = lines[1].replace('20 0 0 0 20 0 0 0 20',
                                        cell)
            for i in range(2,len(lines)):
                row = lines[i].split()
                row[1] = str((float(row[1])+shift) % length)
                lines[i] = ' '.join(row)
            name = f'cell_{int(triclinic)}_zbl_{int(zbl)}_{len(states)}'
            model = potential.replace('model nep_spin\n','model nep_spin\nzbl 1.0 4.0\n') if zbl else potential
            case,run = run_case(root,name,model,'\n'.join(lines)+'\n','')
            assert run.returncode == 0, (name,run.stderr)
            states.append(read_frame(case/'state.xyz'))
        errors = {key:max(abs(a-b) for left,right in zip(states[0][key],states[1][key])
                          for a,b in zip(left,right))
                  for key in ('forces','mforce','energy_atom','virial')}
        report[f'triclinic={triclinic},zbl={zbl}'] = errors
    print(json.dumps(report,indent=2),flush=True)
    for shape,errors in report.items():
        assert all(value < 5e-8 for value in errors.values()), (shape,errors)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='spin-large-cell-') as temporary:
        root = args.output or Path(temporary)
        root.mkdir(parents=True,exist_ok=True)
        validate(root)
