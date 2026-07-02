"""Shared fixtures, tolerances, and CLI options for the GPUMD pytest suite.

GPUMD's `gpumd`/`nep` binaries are GPU-only. CPUNEP (calorine's independent in-memory CPU
reference evaluator -- a separate codebase, not a CPU build of GPUMD) is used here only to
cross-check GPUNEP, never as a GPU-free substitute, so every test in this suite still requires a
GPU; see `gpumd_pytest_suite_spec.md` at the repo root for the full design.
"""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'
MODELS_DIR = FIXTURES_DIR / 'models'
STRUCTURES_DIR = FIXTURES_DIR / 'structures'
GOLDEN_DIR = FIXTURES_DIR / 'golden'
SANITIZER_INPUTS_DIR = FIXTURES_DIR / 'sanitizer_inputs'

GPUMD_EXECUTABLE = REPO_ROOT / 'gpumd'
NEP_EXECUTABLE = REPO_ROOT / 'nep'

# Tolerances are set with fp32 accumulation error and non-associative reduction order in mind:
# GPUMD sums per-atom contributions across GPU threads/blocks in an order that is not fixed
# across kernel launches, so even two GPUNEP runs of the identical input never agree bit for
# bit, and GPUNEP vs. CPUNEP (a wholly separate evaluator) agree even less exactly. These
# rtol/atol values are loose enough to absorb that noise while still catching real regressions
# (missing terms, wrong sign, unit errors, broken kernels).
TOLERANCES = {
    'energy': dict(rtol=1e-5, atol=1e-8),  # eV
    'force': dict(rtol=1e-4, atol=1e-6),   # eV/Angstrom
    'virial': dict(rtol=1e-4, atol=1e-6),
}


def approx_tol(value, tol):
    """pytest.approx() using an explicit rtol/atol dict (the numpy.allclose convention, used
    directly for array comparisons elsewhere in this suite) rather than pytest.approx's own
    rel/abs parameter names, hence this small adaptor."""
    return pytest.approx(value, rel=tol['rtol'], abs=tol['atol'])


def approx(value, kind):
    """approx_tol() using this suite's centralized TOLERANCES, keyed by 'energy'/'force'/
    'virial'."""
    return approx_tol(value, TOLERANCES[kind])


# Translating/rotating/permuting a structure and re-evaluating it on the 'gpu' calculator runs a
# second, independent gpumd subprocess on genuinely different floating-point input (different
# absolute coordinates, different neighbor-list/atom ordering), not just a second read of the
# same configuration -- a larger, but still expected, source of fp32/reduction-order noise than
# TOLERANCES was calibrated for (same-configuration comparisons). This is most visible on small
# systems, where the total energy itself is small enough that a fixed absolute noise floor is a
# much larger relative fraction than on a large cell -- hence atol carrying more weight here than
# in TOLERANCES. qNEP models route through Ewald for these checks (see make_gpunep) specifically
# so that the reciprocal-space method itself isn't an additional variable on top of this ordinary
# reduction-order noise; what's left afterward is the same kind of noise plain NEP models show
# under the same transforms, just calibrated with enough margin to cover the smallest systems.
# Used only by test_invariances.py's translation/rotation/permutation checks on the 'gpu'
# calculator param.
GPU_TRANSFORM_ENERGY_TOLERANCE = dict(rtol=1e-4, atol=1e-5)
GPU_TRANSFORM_FORCE_TOLERANCE = dict(rtol=1e-4, atol=3e-5)


def transform_tolerance(kind, calculator_kind):
    """Tolerance for translation/rotation/permutation invariance checks; see the comment above
    GPU_TRANSFORM_ENERGY_TOLERANCE for why the 'gpu' calculator needs a looser bound than
    TOLERANCES here specifically."""
    if calculator_kind == 'gpu':
        if kind == 'energy':
            return GPU_TRANSFORM_ENERGY_TOLERANCE
        return GPU_TRANSFORM_FORCE_TOLERANCE
    return TOLERANCES[kind]


# test_force_energy_consistency.py checks GPUNEP's own analytic forces against a central
# finite-difference of GPUNEP's own energy -- deliberately GPUNEP only, since the point is to
# validate GPUMD's (this repo's) force/energy self-consistency, not calorine's separate nep_cpu
# reference implementation. GPU-computed energies have a precision floor that a smaller
# finite-difference step does not shrink below (unlike the usual truncation-error argument for
# picking a small step), so DISPLACEMENT is chosen large enough that the resulting energy
# difference sits comfortably above that floor, and the tolerance below is set accordingly rather
# than to the tightest value truncation error alone would justify.
GPU_FINITE_DIFFERENCE_FORCE_TOLERANCE = dict(rtol=1e-2, atol=4e-3)


# test_cross_check.py compares GPUNEP against CPUNEP -- two wholly separate codebases, so some
# fp32/implementation-detail divergence is expected even where both are correct, and small
# systems (where the total energy/force magnitude is itself small) make atol carry more weight
# than rtol alone would. qNEP models route through Ewald on the GPUNEP side for this comparison
# (see make_gpunep) so the reciprocal-space method matches calorine's Ewald-only implementation;
# what remains is the same class of cross-implementation noise seen for plain NEP models.
CROSS_CHECK_ENERGY_TOLERANCE = dict(rtol=1e-3, atol=2e-4)
CROSS_CHECK_FORCE_TOLERANCE = dict(rtol=1e-3, atol=5e-5)
CROSS_CHECK_BEC_TOLERANCE = dict(rtol=1e-3, atol=1e-5)


def pytest_addoption(parser):
    parser.addoption(
        '--update-golden',
        action='store_true',
        default=False,
        help='Overwrite golden reference files in fixtures/golden instead of asserting.',
    )
    parser.addoption(
        '--dump-fixtures',
        action='store_true',
        default=False,
        help='Write the small test structures/models used by run_sanitizer_checks.sh into '
             'fixtures/sanitizer_inputs/ and exit without running any tests.',
    )


def pytest_configure(config):
    if config.getoption('--dump-fixtures'):
        _write_sanitizer_fixtures()
        pytest.exit('Wrote sanitizer fixtures to fixtures/sanitizer_inputs/', returncode=0)


@pytest.fixture
def update_golden(request):
    return request.config.getoption('--update-golden')


def compare_or_update_golden(name, values, tolerances, update_golden):
    """Shared compare-or-update logic for golden-file regression tests. `values` is a dict of
    {key: array-like} to freeze/check; `tolerances` is either a single {'atol', 'rtol'} dict
    applied to every key, or a {key: {'atol', 'rtol'}} dict for per-key tolerances.

    GPUMD's own output is the ground truth for NEP-family architectures here, not an
    independently maintained reference implementation (calorine's CPUNEP is a separate codebase
    not guaranteed to support the same features) -- so correctness is judged by comparing against
    this repo's own previously-generated output, not by cross-checking against calorine. A real
    discrepancy against golden data is a signal to investigate (regression, or an intentional
    change needing --update-golden), never something to reconcile against a different evaluator.
    """
    path = GOLDEN_DIR / f'{name}.npz'
    if update_golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
        pytest.skip(f'Updated golden file {path}; rerun without --update-golden to verify it.')
    if not path.exists():
        pytest.fail(f'Golden file {path} does not exist; run with --update-golden to create it.')
    golden = np.load(path)
    for key, value in values.items():
        tol = tolerances[key] if key in tolerances else tolerances
        assert np.allclose(value, golden[key], **tol), f'{name}: {key} mismatch vs golden file'


@pytest.fixture
def gpumd_command():
    return str(GPUMD_EXECUTABLE)


@pytest.fixture
def nep_command():
    return str(NEP_EXECUTABLE)


def make_bulk_C():
    """16-atom diamond-C cell, pre-rattled (stdev=0.01, seed=42) with reference NEP energy,
    forces, and stress embedded as extended-xyz arrays. Diamond is high-symmetry with zero
    forces at the ideal positions, so we read an already-rattled structure from file rather
    than rattling the ideal cell in-fixture."""
    return read(STRUCTURES_DIR / 'C-nat16-rattled.xyz')


def make_bulk_perovskite():
    """40-atom (2x2x2) cubic BaTiO3 cell, a=4.009 Angstrom, pre-rattled (stdev=0.01, seed=42)
    with reference NEP energy, forces, and stress embedded. Same high-symmetry rationale as
    bulk_C above."""
    return read(STRUCTURES_DIR / 'BaTiO3-nat40-rattled.xyz')


def make_bulk_water():
    """63-atom (21 H2O) small bulk (liquid) water box taken from an MD snapshot -- not a gas-phase
    molecule, despite the single-molecule naming this fixture used earlier in this suite's
    development. Already periodic (pbc="T T T"), satisfying qNEP models' PBC requirement (gpumd
    raises 'Cannot use non-periodic boundaries for qNEP models' otherwise, since the charge
    contribution uses Ewald/PPPM summation, which assumes periodicity) without needing manual
    vacuum padding. No embedded energy/forces/stress in this file (unlike bulk_C/bulk_perovskite
    above) -- it carries only species and positions."""
    return read(STRUCTURES_DIR / 'water-nat63-from-md.xyz')


def make_bulk_bazro3():
    """40-atom cubic BaZrO3 cell, pre-rattled. Used by test_regression.py and
    test_io_tnep_commands.py, not part of _STRUCTURE_BUILDERS/the structure x model_type matrix
    used elsewhere, since both consumers pair it with one specific model rather than sweeping
    it across every model_type. No embedded energy/forces/stress -- species and positions
    only."""
    return read(STRUCTURES_DIR / 'BaZrO3-nat40-rattled.xyz')


_STRUCTURE_BUILDERS = {
    'bulk_C': make_bulk_C,
    'bulk_perovskite': make_bulk_perovskite,
    'bulk_water': make_bulk_water,
}

# (structure_name, model_type) -> file in fixtures/models/. Missing combinations (currently
# bulk_C + either qnep mode, since no qNEP model was supplied for carbon) are skipped
# explicitly in model_path below rather than silently omitted from the params list.
_MODEL_FILES = {
    ('bulk_C', 'nep'): 'nep_C.txt',
    ('bulk_perovskite', 'nep'): 'nep_BaTiO3.txt',
    ('bulk_perovskite', 'qnep_mode1'): 'qnep_mode1_BaTiO3.txt',
    ('bulk_perovskite', 'qnep_mode2'): 'qnep_mode2_BaTiO3.txt',
    ('bulk_water', 'nep'): 'nep_water.txt',
    ('bulk_water', 'qnep_mode1'): 'qnep_mode1_water.txt',
    ('bulk_water', 'qnep_mode2'): 'qnep_mode2_water.txt',
}


@pytest.fixture(params=list(_STRUCTURE_BUILDERS))
def structure_name(request):
    return request.param


@pytest.fixture
def structure(structure_name):
    """Returns an ase.Atoms object for the current structure_name param."""
    return _STRUCTURE_BUILDERS[structure_name]()


@pytest.fixture(params=['nep', 'qnep_mode1', 'qnep_mode2'])
def model_type(request):
    return request.param


@pytest.fixture
def model_path(structure_name, model_type):
    filename = _MODEL_FILES.get((structure_name, model_type))
    if filename is None:
        pytest.skip(f'No {model_type} model available for structure {structure_name!r}.')
    return MODELS_DIR / filename


def make_gpunep(model_path, gpumd_command, model_type):
    """Builds a GPUNEP calculator. For qNEP (charge) models, explicitly requests the Ewald
    reciprocal-space method rather than GPUMD's default PPPM (see
    doc/gpumd/input_parameters/kspace.rst), because:
    - calorine's CPUNEP only implements Ewald (src/nepy/ewald_nep.cpp) -- comparing against it
      while GPUNEP uses PPPM would be comparing two different approximations to the
      reciprocal-space sum, not validating agreement between two implementations of the same
      method.
    - PPPM's FFT mesh has a fixed orientation relative to the simulation box, so rotating the
      box (unlike relabeling/permuting atoms) changes the mesh's relationship to the atoms and
      introduces a real difference unrelated to whatever invariance is actually being checked.

    GPUMD's own PPPM-vs-Ewald agreement is validated directly in test_kspace_consistency.py,
    not routed around here -- this helper exists so that *other* comparisons (against CPUNEP, or
    against a transformed copy of the same structure) aren't also, incidentally, comparisons
    between two different reciprocal-space methods.
    """
    calc = GPUNEP(str(model_path), command=gpumd_command)
    if model_type != 'nep':
        calc.single_point_parameters = calc.single_point_parameters + [('kspace', 'ewald')]
    return calc


@pytest.fixture(params=['gpu', 'cpu_reference'])
def calculator_kind(request):
    return request.param


@pytest.fixture
def calculator(calculator_kind, model_path, model_type, gpumd_command):
    if calculator_kind == 'gpu':
        return make_gpunep(model_path, gpumd_command, model_type)
    return CPUNEP(str(model_path))


def _write_sanitizer_fixtures():
    """Writes a couple of representative (structure, model) run.in/model.xyz pairs into
    fixtures/sanitizer_inputs/, reusing the same structure builders and model files as the
    pytest fixtures above so run_sanitizer_checks.sh doesn't duplicate structure-generation
    logic. Covers a plain-NEP case and a qNEP case (to exercise the charge/BEC/dpdt kernels)."""
    SANITIZER_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            'nep_bulk_C',
            make_bulk_C(),
            MODELS_DIR / 'nep_C.txt',
            [
                ('velocity', 300),
                ('ensemble', ['nve']),
                ('time_step', 1),
                ('dump_thermo', 1),
                ('dump_force', 1),
                ('run', 5),
            ],
        ),
        (
            'qnep_mode1_bulk_perovskite',
            make_bulk_perovskite(),
            MODELS_DIR / 'qnep_mode1_BaTiO3.txt',
            [
                ('velocity', 300),
                ('ensemble', ['nve']),
                ('time_step', 1),
                ('dump_thermo', 1),
                ('compute_dpdt', 1),
                ('run', 5),
            ],
        ),
    ]

    for name, atoms, model_path_, params in cases:
        case_dir = SANITIZER_INPUTS_DIR / name
        case_dir.mkdir(parents=True, exist_ok=True)
        calc = GPUNEP(str(model_path_), directory=str(case_dir))
        atoms.calc = calc
        calc.run_custom_md(params, only_prepare=True)
