"""Shared infrastructure for the Tier 1 command IO smoke tests (test_io_basic_setup.py,
test_io_dump_commands.py, test_io_compute_commands.py).

These tests confirm each gpumd run.in keyword runs (exit code 0) and produces a parseable
output file of the expected shape -- not physical correctness, which is out of scope here and
already covered elsewhere (test_cross_check.py, test_md_conservation.py) for the handful of
quantities those files touch.

Not a test_*.py module itself (pytest won't collect it), just shared helpers.
"""
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from calorine.gpumd import write_xyz

from conftest import make_gpunep

BASE_TIME_STEP = 1.0
BASE_N_STEPS = 5


@dataclass
class CommandIOCase:
    name: str
    run_in_lines: List[Tuple] = field(default_factory=list)
    expected_output_files: List[str] = field(default_factory=list)
    parse_check: Callable = lambda path: None
    n_groups: int = 0  # 0 = no groups; 1 = everyone in a single group (method 0, group 0),
    # enough for a smoke test of most group-consuming keywords (fix/add_force/add_spring); 2 =
    # split roughly in half into groups 0/1 of method 0, needed for keywords that require two
    # distinct groups (move, which GPUMD refuses to run without a separate fixed group). calorine's
    # GPUNEP has no built-in way to attach groupings through run_custom_md, so this rewrites
    # model.xyz after GPUNEP's automatic write via calorine.gpumd.write_xyz's groupings param.
    ensemble: str = 'nve'
    ensemble_params: tuple = ()
    prelude_lines: List[Tuple] = field(default_factory=list)  # lines that must precede
    # `potential` (only `replicate` needs this); an explicit ('potential', ...) entry must be
    # included in run_in_lines/prelude ordering for this to take effect, since GPUNEP otherwise
    # always auto-prepends `potential` as the very first line.
    skip_base_block: bool = False  # True for standalone actions not composable with an
    # ensemble+run block (minimize, compute_cohesive, compute_elastic); their own run_in_lines
    # must define a complete, self-sufficient run.in.
    expect_thermo: bool = None  # None = default to `not skip_base_block` (every base-block case
    # includes dump_thermo; standalone cases don't produce thermo.out unless their own
    # run_in_lines add a trailing ensemble+dump_thermo+run block, as `minimize`'s case does).
    # Override explicitly for a standalone case that does add such a trailing block.
    repeat: tuple = None  # Optional (nx, ny, nz) passed to Atoms.repeat() before writing
    # model.xyz -- lets a case get a bigger/thicker box than the shared structure fixtures
    # provide (e.g. so a cutoff-based command can use a more representative cutoff than the
    # smallest fixture's box would otherwise allow) without a dedicated larger fixture file.

    def __post_init__(self):
        if self.expect_thermo is None:
            self.expect_thermo = not self.skip_base_block


def run_command_io_case(tmp_path, atoms, model_path, model_type, gpumd_command, case):
    """Writes run.in (+ model.xyz with groupings if needed) for one CommandIOCase and executes
    gpumd directly (mirroring the pattern in tests/gpumd/dump_dipole/test_dump_dipole.py),
    rather than run_custom_md(..., only_prepare=False), since several cases need model.xyz
    rewritten with group information after GPUNEP's automatic (group-less) write.

    Returns the completed subprocess (caller checks returncode and expected_output_files).
    """
    if case.repeat is not None:
        atoms = atoms.repeat(case.repeat)

    calc = make_gpunep(model_path, gpumd_command, model_type)
    calc.set_directory(str(tmp_path))
    calc.set_atoms(atoms)

    if case.skip_base_block:
        params = list(case.prelude_lines) + list(case.run_in_lines)
    else:
        ensemble_value = [case.ensemble, *case.ensemble_params] if case.ensemble_params \
            else case.ensemble
        case_keywords = [keyval[0] for keyval in case.run_in_lines]
        base_dump_thermo = [] if 'dump_thermo' in case_keywords else [('dump_thermo', 1)]
        params = [
            *case.prelude_lines,
            ('velocity', [300, 'seed', 42]),
            ('time_step', BASE_TIME_STEP),
            ('ensemble', ensemble_value),
            *base_dump_thermo,
            *case.run_in_lines,
            ('run', BASE_N_STEPS),
        ]

    calc.run_custom_md(params, only_prepare=True)

    if case.n_groups == 1:
        write_xyz(str(tmp_path / 'model.xyz'), atoms, groupings=[[list(range(len(atoms)))]])
    elif case.n_groups == 2:
        half = len(atoms) // 2
        indices = list(range(len(atoms)))
        write_xyz(str(tmp_path / 'model.xyz'), atoms,
                  groupings=[[indices[:half], indices[half:]]])

    return subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False)


def run_and_check(tmp_path, atoms, model_path, model_type, gpumd_command, case):
    """run_command_io_case() + the standard exit-code/output-file/parse assertions shared by
    every Tier 1 test function."""
    result = run_command_io_case(tmp_path, atoms, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, (
        f'{case.name}: gpumd exited {result.returncode}\nstdout:\n{result.stdout}\n'
        f'stderr:\n{result.stderr}')

    if case.expect_thermo:
        thermo_path = tmp_path / 'thermo.out'
        assert thermo_path.exists(), f'{case.name}: thermo.out was not produced'

    for filename in case.expected_output_files:
        output_path = tmp_path / filename
        assert output_path.exists(), f'{case.name}: {filename} was not produced'
        case.parse_check(output_path)
