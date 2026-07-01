"""Command IO smoke tests (Tier 1): simple compute_* commands.

Confirms each keyword runs without error (exit code 0) and produces a parseable output file of
the expected shape -- not physical correctness. See io_helpers.py's module docstring.

GPUMD enforces a stricter constraint than plain minimum-image convention for the RDF-family
commands (compute_rdf/adf/angular_rdf): box thickness must be >= 2.5x the cutoff in every
periodic direction ("The box has a thickness < 2.5 RDF radial cutoffs in a periodic direction",
confirmed against the real binary). bulk_C's box is the thinnest of this suite's structures, so
rather than shrinking the cutoff to fit it (which would only ever sample the first coordination
shell, not a representative RDF), these three cases request a 2x2x2 repeat of whatever structure
they run against (CommandIOCase.repeat) to get enough headroom for a cutoff that spans a few
coordination shells uniformly across all structures. compute_orientorder shares the same nominal
cutoff for simplicity but has no such thickness constraint, so it runs unrepeated.

compute_elastic runs multiple internal energy minimizations per strain component, but measured
in isolation this stays within a few tens of seconds per case for this suite's small toy
structures -- comfortably within `fast`. (An earlier measurement of several minutes turned out to
be an artifact of running a second gpumd process concurrently on the same GPU during that
investigation, not the command's actual cost -- not reproduced with a single gpumd process at a
time.) Only applies to crystalline structures (bulk_C, bulk_perovskite) -- see the
compute_elastic/bulk_water skip below for why bulk_water is excluded.

compute_phonon is deferred, not merely skipped, from Tier 1 entirely (see test_compute_phonon
below for the full reasoning): the toy NEP models in this suite have large cutoffs (6-7
Angstrom), and doc/gpumd/input_parameters/compute_phonon.rst requires the box to be at least
2x(2x cutoff) per direction for many-body potentials -- a supercell of thousands of atoms.
Measured directly: with a 3456-atom supercell (bulk_C replicated 6x6x6, matching the scale used
in the working tests/gpumd/silicon_dispersion/ example, which uses a much cheaper two-body
Tersoff potential instead of NEP), compute_phonon did not finish within 6+ minutes. That's not a
hang (CPU usage stayed pegged the whole time) -- it's genuinely too expensive for any of this
suite's toy models to fit a "smoke test" tier at all, let alone `fast`.
"""
import numpy as np
import pytest
from calorine.gpumd import read_msd

from io_helpers import CommandIOCase, run_and_check
from test_parsing import read_dpdt_out

pytestmark = pytest.mark.fast

SAFE_CUTOFF = 1.6  # Angstrom; small, universally-safe cutoff for cutoff-based commands that
# aren't subject to the RDF-family thickness constraint (currently just compute_orientorder).
RDF_CUTOFF = 3.0  # Angstrom; see module docstring -- paired with CommandIOCase.repeat=(2, 2, 2)
# on the RDF-family cases below to give every structure enough box thickness for it.


def _check_columns(path, ncols, exact=True):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if exact:
        assert data.shape[1] == ncols
    else:
        assert data.shape[1] >= ncols
    assert data.shape[0] > 0


def _check_msd_format(path):
    df = read_msd(str(path))
    assert 'msd_x' in df.columns
    assert len(df) > 0


def _check_dpdt_format(path):
    data = read_dpdt_out(path)
    assert data.shape[1] == 7


def _check_elastic_format(path):
    # elastic.out has no dedicated doc/gpumd/output_files/ entry; format observed directly
    # against the real binary: a '# Elastic Constants Matrix (GPa):' comment line followed by a
    # 6x6 matrix (Voigt notation).
    data = np.loadtxt(path, comments='#')
    assert data.shape == (6, 6)


COMPUTE_COMMAND_CASES = [
    CommandIOCase(
        name='compute_rdf', repeat=(2, 2, 2),
        run_in_lines=[('compute_rdf', [RDF_CUTOFF, 50, 10])],
        expected_output_files=['rdf.out'],
        # >=2, not ==2: rdf.out also has one column per distinct atom-type pair (only 1 type
        # here, so 3 total; more with multiple species) on top of [radius, whole-system RDF].
        parse_check=lambda p: _check_columns(p, ncols=2, exact=False)),
    CommandIOCase(
        # interval must be small enough to actually sample within io_helpers.BASE_N_STEPS, or
        # adf.out is silently written empty rather than erroring (confirmed against the real
        # binary).
        name='compute_adf', repeat=(2, 2, 2),
        run_in_lines=[('compute_adf', [1, 30, 0.0, RDF_CUTOFF])],
        expected_output_files=['adf.out'],
        parse_check=lambda p: _check_columns(p, ncols=2, exact=False)),
    CommandIOCase(
        name='compute_angular_rdf', repeat=(2, 2, 2),
        # both bin counts must be > 20 (GPUMD rejects <= 20 with "A larger n(theta)bins is
        # recommended", confirmed against the real binary).
        run_in_lines=[('compute_angular_rdf', [RDF_CUTOFF, 21, 21, 10])],
        expected_output_files=['angular_rdf.out'],
        parse_check=lambda p: _check_columns(p, ncols=3)),
    CommandIOCase(
        name='compute_msd', run_in_lines=[('compute_msd', [1, 5])],
        expected_output_files=['msd.out'], parse_check=_check_msd_format),
    CommandIOCase(
        name='compute_sdc', run_in_lines=[('compute_sdc', [1, 5])],
        expected_output_files=['sdc.out'], parse_check=lambda p: _check_columns(p, ncols=7)),
    CommandIOCase(
        name='compute_dos', run_in_lines=[('compute_dos', [1, 5, 400.0])],
        expected_output_files=['mvac.out', 'dos.out'],
        parse_check=lambda p: _check_columns(p, ncols=4)),
    CommandIOCase(
        name='compute_orientorder', n_groups=0,
        run_in_lines=[('compute_orientorder', [10, 'cutoff', SAFE_CUTOFF, 1, 4, 0, 0, 0])],
        expected_output_files=['orientorder.out']),
    CommandIOCase(
        name='compute', n_groups=1, run_in_lines=[('compute', [0, 1, 1, 'temperature'])],
        expected_output_files=['compute.out'], parse_check=lambda p: _check_columns(p, ncols=3)),
    CommandIOCase(
        name='compute_chunk',
        run_in_lines=[('compute_chunk', [5, 5, 'bin/1d', 'z', 'lower', 1.0, 'temperature'])],
        expected_output_files=['compute_chunk.out']),
    CommandIOCase(
        name='compute_cohesive', skip_base_block=True,
        run_in_lines=[('compute_cohesive', [0.99, 1.01, 6])],
        expected_output_files=['cohesive.out'], parse_check=lambda p: _check_columns(p, ncols=2)),
]

COMPUTE_ELASTIC_CASE = CommandIOCase(
    name='compute_elastic', skip_base_block=True, run_in_lines=[('compute_elastic', 0.01)],
    expected_output_files=['elastic.out'], parse_check=_check_elastic_format)


@pytest.mark.parametrize(
    'case', [*COMPUTE_COMMAND_CASES, COMPUTE_ELASTIC_CASE], ids=lambda c: c.name)
def test_command_io(tmp_path, structure, structure_name, model_path, model_type, gpumd_command,
                     case):
    if case.name == 'compute_elastic' and structure_name == 'bulk_water':
        pytest.skip(
            'compute_elastic derives elastic constants from the stress response to small '
            'strains around a relaxed equilibrium structure -- a notion that presupposes a '
            'stable crystalline reference structure. bulk_water is a disordered/molecular '
            'liquid-like system with no such well-defined equilibrium lattice, so applying '
            'compute_elastic to it is not physically meaningful; bulk_C and bulk_perovskite '
            '(both crystalline) are the appropriate structures for this command.')
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)


def test_compute_dpdt(tmp_path, structure, model_type, model_path, gpumd_command):
    if model_type == 'nep':
        pytest.skip('compute_dpdt is only meaningful for qNEP models trained with BEC targets.')
    case = CommandIOCase(
        name='compute_dpdt', run_in_lines=[('compute_dpdt', 1)],
        expected_output_files=['dpdt.out'], parse_check=_check_dpdt_format)
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)


def test_compute_phonon():
    pytest.skip(
        'deferred, not merely skipped: compute_phonon requires a box at least 2x(2x potential '
        'cutoff) per direction for many-body potentials (doc/gpumd/input_parameters/'
        'compute_phonon.rst) -- with this suite\'s 6-7 Angstrom NEP cutoffs that means a '
        'supercell of thousands of atoms. Measured directly on a 3456-atom bulk_C supercell '
        '(replicate 6 6 6, matching the scale of the working tests/gpumd/silicon_dispersion/ '
        'example, which uses a far cheaper two-body Tersoff potential instead of NEP): did not '
        'finish within 6+ minutes of pegged CPU usage (not a hang, genuinely that expensive). '
        'Revisit with a smaller-cutoff toy model if compute_phonon coverage is wanted, rather '
        'than accepting this cost in a nominally "smoke test" tier.')
