"""Reference D3(0) values for ALL zero-damping functionals of src/force/dftd3/dftd3zero_table.cuh
(test_dftd3_zero.py::test_dftd3_zero_all_functionals). Writes hco_mix.xyz and all_functionals.json.

Oracle: simple-dftd3 (python `dftd3`), `ZeroDampingParam`, two-body only (s9 = 0), cutoffs equal to
those given to GPUMD (pair 12 A, coordination number 6 A). The functional parameters (s6, sr6, s8,
sr8; alpha6 = 14) are the ones of the GPUMD table (parsed from the .cuh), passed explicitly. The
functional-name mapping GPUMD name -> simple-dftd3 name is derived here: the GPUMD name itself, then
the names obtained by inserting up to two '-' into it, are tried with `ZeroDampingParam(method=...)`;
a candidate is accepted only if simple-dftd3's own parameters give the same energy (relative 1e-10,
CODATA2018 constants on both sides) as the explicit parameters of the GPUMD table. A functional that
simple-dftd3 does not know, or whose parameters differ, gets NO reference entry and is listed under
`_unmapped` with the reason. Length constant: as in generate_reference.py (GPUMD `Bohr`; sr6 and sr8
rescaled by the CODATA2018 ratio so that r/R0 is the ratio of Angstrom quantities).

    python -I generate_all_functionals.py       # env with numpy and dftd3 (no ase needed)
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np
from dftd3 import __version__ as DFTD3_VERSION
from dftd3.interface import DispersionModel, ZeroDampingParam

HA = 27.211386245988
BOHR = 0.5291772575069165           # GPUMD src/force/dftd3/dftd3para.cuh `Bohr` (float constant 0.5291772575069165f)
BOHR_CODATA2018 = 0.529177210903    # simple-dftd3 internal length constant
CUTOFFS = (12, 6)
HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[2] / 'src' / 'force' / 'dftd3' / 'dftd3zero_table.cuh'
CELL = 11.0
ATOMS = [  # one H2O, one H2CO, one CH4 in a periodic 11 A cube (small-box path)
    ('O', 2.00, 2.00, 2.00), ('H', 2.96, 2.00, 2.00), ('H', 1.76, 2.93, 2.00),
    ('C', 6.00, 2.00, 2.00), ('O', 7.20, 2.00, 2.00), ('H', 5.40, 2.93, 2.00), ('H', 5.40, 1.07, 2.00),
    ('C', 2.00, 7.00, 6.00), ('H', 3.09, 7.00, 6.00), ('H', 1.64, 8.03, 6.00), ('H', 1.64, 6.48, 6.90),
    ('H', 1.64, 6.48, 5.10)]
Z = {'H': 1, 'C': 6, 'O': 8}


def table_parameters():
    text = TABLE.read_text()
    block = text.split('D3ZERO_PARAMS[D3ZERO_NUM_PARAMS] = {')[1].split('};')[0]
    rows = re.findall(r'\{"([a-z0-9]+)",\s*([-0-9.]+)f,\s*([-0-9.]+)f,\s*([-0-9.]+)f,\s*([-0-9.]+)f\}', block)
    assert len(rows) == int(re.search(r'D3ZERO_NUM_PARAMS = (\d+)', text).group(1))
    return {n: tuple(float(x) for x in v) for n, *v in rows}     # name -> (s6, sr6, s8, sr8)


def candidates(name):
    yield name
    for k in (1, 2):
        for pos in itertools.combinations(range(1, len(name)), k):
            s = name
            for p in reversed(pos):
                s = s[:p] + '-' + s[p:]
            yield s


def energy(numbers, positions_ang, lib_name=None, params=None, codata=False):
    c = BOHR_CODATA2018 if codata else BOHR
    cell = np.diag([CELL] * 3) / c
    model = DispersionModel(numbers, positions_ang / c, cell, np.array([True] * 3))
    model.set_realspace_cutoff(CUTOFFS[0] / c, CUTOFFS[0] / c, CUTOFFS[1] / c)
    if lib_name is not None:
        param = ZeroDampingParam(method=lib_name)
    else:
        s6, sr6, s8, sr8 = params
        scale = 1.0 if codata else BOHR_CODATA2018 / BOHR
        param = ZeroDampingParam(s6=s6, s8=s8, sr6=sr6 * scale, sr8=sr8 * scale, alpha6=14.0, s9=0.0)
    return model.get_dispersion(param, grad=True), c


def main():
    numbers = np.array([Z[a[0]] for a in ATOMS])
    pos = np.array([a[1:] for a in ATOMS], float)
    lines = [str(len(ATOMS)), f'Lattice="{CELL} 0 0 0 {CELL} 0 0 0 {CELL}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    lines += [f'{a[0]:2s} {a[1]:10.6f} {a[2]:10.6f} {a[3]:10.6f}' for a in ATOMS]
    (HERE / 'hco_mix.xyz').write_text('\n'.join(lines) + '\n')

    table = table_parameters()
    out = {'_provenance': {
        'reference': 'simple-dftd3 via dftd3-python', 'dftd3_python_version': DFTD3_VERSION,
        'damping': 'zero (D3(0)), two-body only (s9 = 0); explicit parameters of dftd3zero_table.cuh, alpha6 = 14',
        'structure': 'hco_mix.xyz (H2O + H2CO + CH4, periodic 11 A cube)', 'potential_cutoff': CUTOFFS[0],
        'cn_cutoff': CUTOFFS[1], 'bohr_angstrom': BOHR,
        'bohr_note': 'GPUMD Bohr constant; sr6/sr8 rescaled by BOHR_CODATA2018/Bohr, as in generate_reference.py',
        'units': 'eV, eV/A; virial = -sigma'}, '_unmapped': {}, '_name_map': {}}
    for name, params in table.items():
        lib = None
        for cand in candidates(name):
            try:
                ZeroDampingParam(method=cand)
            except RuntimeError:
                continue
            lib = cand
            break
        if lib is None:
            out['_unmapped'][name] = 'not known to simple-dftd3 (no candidate name accepted)'
            continue
        a = energy(numbers, pos, lib_name=lib, codata=True)[0]['energy']
        b = energy(numbers, pos, params=params, codata=True)[0]['energy']
        if abs(a - b) > 1e-10 * max(abs(a), abs(b), 1e-12):
            out['_unmapped'][name] = f'simple-dftd3 "{lib}" has other parameters (E {a:.12e} vs table {b:.12e} Ha)'
            continue
        r, c = energy(numbers, pos, params=params)
        out['_name_map'][name] = lib
        out[name] = dict(functional=name, simple_dftd3_name=lib, params=list(params),
                         energy=float(r['energy'] * HA), forces=np.round(-r['gradient'] * HA / c, 12).tolist(),
                         virial=np.round(-r['virial'] * HA, 12).tolist())
    json.dump(out, open(HERE / 'all_functionals.json', 'w'), indent=1)
    print(len(out['_name_map']), 'mapped of', len(table))
    for k, v in out['_unmapped'].items():
        print('UNMAPPED', k, v)
    print({k: v for k, v in out['_name_map'].items() if k != v})


if __name__ == '__main__':
    main()
