"""Measure the zero-damping error of the gpumd binary against the stored simple-dftd3 references
and write zero_baseline.json (median error per case: energy, force, virial) used by tolerance() in
test_dftd3_zero.py. NOT part of the pytest run. Run from anywhere with the py312-gpumd env:

    python measure_zero_baseline.py [repeats=5] [--all-functionals]   # writes zero_baseline.json

The baselines are the errors of THE IMPLEMENTATION ITSELF (same code path as the test: d3_part), so the
tests detect changes of the implementation and bound the error by the caps; they are not an
independent proof of accuracy beyond the caps. Also records max over repeats, binary md5, GPU, CUDA, date.
"""
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import test_dftd3_zero as T  # noqa: E402


def measure_reference(repeats):
    out = {}
    for key in T.CASES:
        ref = T.REFERENCE[key]
        atoms = read(T.structure_path(ref))
        line = f"dftd3 {ref['functional']} {ref['potential_cutoff']} {ref['cn_cutoff']} zero"
        errs = []
        for _ in range(repeats):
            with tempfile.TemporaryDirectory() as d:
                e, f, v = T.d3_part(Path(d), atoms, line)
            errs.append((abs(e - ref['energy']),
                         float(np.abs(f - np.array(ref['forces'])).max()),
                         float(np.abs(v - np.array(ref['virial'])).max())))
        a = np.array(errs)
        out[key] = dict(case=ref['case'], median=np.median(a, axis=0).tolist(),
                        max=a.max(axis=0).tolist())
        print(key, out[key], flush=True)
    return out


def measure_all_functionals(repeats):
    out = {}
    atoms = read(T.ZERO_DIR / 'hco_mix.xyz')
    prov = T.ALL_FUNCTIONALS['_provenance']
    for name in T.ALL_FUNCTIONAL_NAMES:
        ref = T.ALL_FUNCTIONALS[name]
        line = f"dftd3 {name} {prov['potential_cutoff']} {prov['cn_cutoff']} zero"
        errs = []
        for _ in range(repeats):
            with tempfile.TemporaryDirectory() as d:
                _, (e, f, _) = T.run_gpumd(Path(d), atoms, line, model=T.ZERO_NEP)
            errs.append((abs(e - ref['energy']), float(np.abs(f - np.array(ref['forces'])).max())))
        a = np.array(errs)
        out[name] = dict(energy_abs_ref=abs(ref['energy']), median=np.median(a, axis=0).tolist(), max=a.max(axis=0).tolist())
        print(name, out[name], flush=True)
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    repeats = int(args[0]) if args else 5
    meta = dict(binary="src/gpumd", binary_md5=hashlib.md5(open(T.EXE, 'rb').read()).hexdigest(),
                gpu=subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   capture_output=True, text=True).stdout.strip(),
                cuda=[l for l in subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout.splitlines()
                      if 'release' in l],
                date=datetime.date.today().isoformat(), repeats=repeats,
                note='baselines are the measured errors of this implementation against simple-dftd3 (median over '
                     'repeats; units eV, eV/A, eV); they detect changes of the implementation, not independent accuracy')
    if '--all-functionals' in sys.argv:
        res = dict(_meta=meta, functionals=measure_all_functionals(repeats))
        json.dump(res, sys.stdout, indent=1)
    else:
        res = dict(_meta=meta, cases=measure_reference(repeats))
        json.dump(res, open(HERE / 'zero_baseline.json', 'w'), indent=1)
