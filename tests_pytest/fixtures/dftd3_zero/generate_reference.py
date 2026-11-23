"""Generate the independent D3(0) (zero damping) reference values for test_dftd3_zero.py.

Reference: simple-dftd3 (`dftd3` python bindings, version recorded in reference.json) with zero
damping, two-body only (s9 = 0, no ATM term), double precision, real-space cutoffs equal to the
values GPUMD is given (pair cutoff = `dftd3` potential_cutoff, CN cutoff = coordination number
cutoff). The structures are the *.xyz files next to this script. This script is NOT part of the pytest run; run it only to regenerate
reference.json, in an environment with `ase`, `numpy` and `dftd3`:

    python -I generate_reference.py         # writes reference.json next to this file

Functional parameters (D3(0), def2-QZVP set of Grimme et al., J. Chem. Phys. 132, 154104
(2010), as in dftd3-lib `setfuncpar` version 3 and simple-dftd3): s6 = 1, alpha6 = 14,
alpha8 = 16, sr8 = 1, and (sr6, s8):
    pbe 1.217 0.722 | revpbe 0.923 1.010 | blyp 1.094 1.682 | b3lyp 1.261 1.703 | tpss 1.166 1.105
The script asserts that these equal simple-dftd3's built-in parameters of the same functional
(simple-dftd3 names: b-lyp, b3-lyp; the GPUMD keyword names are blyp, b3lyp, like the BJ table).

Length constant. GPUMD (src/force/dftd3/dftd3para.cuh) converts with Bohr = 0.5291772575069165 A, not
with CODATA2018 (0.529177210903 A) as simple-dftd3 does. Positions, cutoffs, gradients and virials
are therefore converted with the GPUMD value (BOHR below). simple-dftd3's pair radii R0 (and
covalent radii) are internally tabulated in bohr via CODATA2018, so the damping radius parameters
sr6 and sr8 are multiplied by BOHR_CODATA2018 / BOHR to keep r / R0 equal to the ratio of the
Angstrom quantities (R0 table in Angstrom divided by the same constant as the geometry). With
alpha = 14/16 an inconsistent constant would change the energy by up to 1.5e-6 (relative). Energy unit: Hartree = 27.211386245988 eV (as in the BJ reference).

Stored values use GPUMD conventions: energy in eV, forces in eV/A, virial = -sigma (3x3, eV).
Do not edit the numbers by hand and do not regenerate them to make a failing test pass.
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read
from dftd3 import __version__ as DFTD3_VERSION
from dftd3.interface import DispersionModel, ZeroDampingParam

HA = 27.211386245988
BOHR = 0.5291772575069165           # GPUMD src/force/dftd3/dftd3para.cuh `Bohr`
BOHR_CODATA2018 = 0.529177210903    # simple-dftd3 internal length constant

HERE = Path(__file__).resolve().parent
STRUCTURES = HERE

# GPUMD functional name -> (simple-dftd3 name, sr6, s8); s6 = 1, sr8 = 1, alpha6 = 14.
FUNCTIONALS = {
    'pbe': ('pbe', 1.217, 0.722),
    'revpbe': ('revpbe', 0.923, 1.010),
    'blyp': ('b-lyp', 1.094, 1.682),
    'b3lyp': ('b3-lyp', 1.261, 1.703),
    'tpss': ('tpss', 1.166, 1.105),
}
# case name -> (structure file, (potential cutoff, CN cutoff), note); cutoffs as in the BJ examples
CASES = {
    'benzene_dimer': ('benzene_dimer.xyz', (12, 6), 'non-periodic (large-box path)'),
    'graphite': ('graphite.xyz', (12, 6), 'small-box path, hexagonal cell'),
    'water27': ('water27.xyz', (12, 6), 'small-box path, cubic cell'),
    'water27_rc25': ('water27_rc25.xyz', (25, 15), 'small-box path, large cutoffs'),
    'water27_tri': ('water27_tri.xyz', (12, 6), 'small-box path, triclinic cell'),
    'water_large': ('water_large.xyz', (8, 4), 'large-box path'),
    'cu_large': ('cu_large.xyz', (8, 4), 'large-box path, Cu'),
}
ALL_CASES = list(CASES)
SELECTION = {'pbe': ALL_CASES, 'revpbe': ['benzene_dimer', 'graphite', 'water27'],
             'blyp': ['benzene_dimer', 'graphite', 'water27'],
             'b3lyp': ['benzene_dimer', 'graphite', 'water27'],
             'tpss': ['benzene_dimer', 'graphite', 'water27']}


def heavy_cluster():
    """Periodic cluster with heavy elements (Z up to 83) in a 10 A box: small-box path, many R0 pairs."""
    from ase import Atoms
    symbols = ['La', 'Hg', 'Tl', 'Bi', 'Pt', 'Os', 'W', 'Cs', 'Xe', 'I']
    rng = np.random.default_rng(7)
    grid = np.array([[i, j, k] for i in range(3) for j in range(2) for k in range(2)], float)[:10]
    pos = grid * 3.3 + rng.uniform(-0.3, 0.3, (10, 3)) + 1.0
    return Atoms(symbols, positions=pos, cell=[10.0, 10.0, 10.0], pbc=True)


def dispersion(atoms, cutoffs, functional):
    """Two-body D3(0) energy (eV), forces (eV/A), virial (eV) with the GPUMD length constant."""
    _, sr6, s8 = FUNCTIONALS[functional]
    scale = BOHR_CODATA2018 / BOHR
    pbc = atoms.pbc.any()
    model = DispersionModel(atoms.numbers, atoms.positions / BOHR,
                            atoms.cell.array / BOHR if pbc else None, atoms.pbc if pbc else None)
    model.set_realspace_cutoff(cutoffs[0] / BOHR, cutoffs[0] / BOHR, cutoffs[1] / BOHR)
    param = ZeroDampingParam(s6=1.0, s8=s8, sr6=sr6 * scale, sr8=1.0 * scale, alpha6=14.0, s9=0.0)
    r = model.get_dispersion(param, grad=True)
    return (float(r['energy'] * HA), -r['gradient'] * HA / BOHR, -r['virial'] * HA)


def check_parameters():
    """Explicit parameters == simple-dftd3 built-in ones (same geometry, B2018 constant, dimer)."""
    from ase import Atoms
    dimer = Atoms('OHH', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    for name, (lib_name, sr6, s8) in FUNCTIONALS.items():
        m = DispersionModel(dimer.numbers, dimer.positions / BOHR_CODATA2018)
        a = m.get_dispersion(ZeroDampingParam(method=lib_name), grad=False)['energy']
        b = m.get_dispersion(ZeroDampingParam(s6=1.0, s8=s8, sr6=sr6, sr8=1.0, alpha6=14.0, s9=0.0),
                             grad=False)['energy']
        assert abs(a - b) <= 1e-12 * abs(a), (name, a, b)


if __name__ == '__main__':
    check_parameters()
    out = {'_provenance': {
        'reference': 'simple-dftd3 via dftd3-python', 'dftd3_python_version': DFTD3_VERSION,
        'damping': 'zero (D3(0)), two-body only (s9 = 0, no ATM); s6 = 1, alpha6 = 14, alpha8 = 16, sr8 = 1',
        'functional_parameters': {k: {'simple_dftd3_name': v[0], 'sr6': v[1], 's8': v[2]}
                                  for k, v in FUNCTIONALS.items()},
        'bohr_angstrom': BOHR, 'bohr_angstrom_note': 'GPUMD dftd3para.cuh Bohr; sr6/sr8 rescaled by '
        f'{BOHR_CODATA2018}/Bohr so that r/R0 matches the Angstrom ratio',
        'hartree_eV': HA, 'units': 'eV, eV/A; virial = -sigma (GPUMD convention)'}}
    for functional in FUNCTIONALS:
        for case in SELECTION[functional]:
            fname, cut, note = CASES[case]
            atoms = read(STRUCTURES / fname)
            e, f, v = dispersion(atoms, cut, functional)
            out[f'{functional}:{case}'] = dict(
                functional=functional, case=case, structure=fname, natoms=len(atoms),
                potential_cutoff=cut[0], cn_cutoff=cut[1], note=note, energy=e,
                forces=np.round(f, 12).tolist(), virial=np.round(v, 12).tolist())
            print(functional, case, len(atoms), cut, e)
    atoms = heavy_cluster()
    atoms.write(HERE / 'heavy_cluster.xyz')
    e, f, v = dispersion(atoms, (12, 6), 'pbe')
    out['pbe:heavy_cluster'] = dict(
        functional='pbe', case='heavy_cluster', structure='heavy_cluster.xyz', natoms=len(atoms),
        potential_cutoff=12, cn_cutoff=6, note='periodic 10 A box, heavy elements (Z <= 83), small-box path',
        energy=e, forces=np.round(f, 12).tolist(), virial=np.round(v, 12).tolist())
    print('pbe heavy_cluster', len(atoms), e)
    json.dump(out, open(HERE / 'reference.json', 'w'))
