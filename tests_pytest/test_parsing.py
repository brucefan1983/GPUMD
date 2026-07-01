"""I/O round-trip and malformed-input tests for GPUMD/calorine parsers.

No calculator invocation here, and no GPU required — the one file in this suite that could in
principle run without one (see gpumd_pytest_suite_spec.md), though the overall suite makes no
such distinction elsewhere. thermo.out/dpdt.out round-trips against files actually produced by
`gpumd` are exercised indirectly in test_md_conservation.py; here we only need representative
well-formed content to validate the readers' own contract.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from calorine.nep.model import read_model
from hypothesis import given, settings, strategies as st

from conftest import MODELS_DIR

THERMO_OUT_NCOLS = 18
DPDT_OUT_NCOLS = 7

# calorine.nep.model.read_model is a hand-rolled parser. As of calorine's
# bug/fix-read_model-behavior-for-missing-version-line fix (merged to master, commit f02fad6),
# every file-content validation check in the read path (missing header fields, bad version,
# malformed cutoff/basis_size/n_max/l_max, non-finite cutoff, unknown field, wrong token count)
# raises ValueError or OSError -- the remaining `assert` statements in read_model are reserved
# for internal invariants ("please submit a bug report" style) that malformed *input* should
# never be able to trigger. So AssertionError is deliberately excluded from this envelope: if
# hypothesis ever finds a malformed nep.txt that raises AssertionError, that is itself the bug
# report (a calorine internal-invariant check firing on bad input, not a suite issue to paper
# over by widening this tuple). Anything else outside (KeyError, IndexError, TypeError,
# UnicodeDecodeError, ...) is likewise a real parser bug to report.
EXPECTED_MODEL_PARSE_EXCEPTIONS = (ValueError, OSError)

ALL_MODEL_FILES = sorted(MODELS_DIR.glob('*.txt'))
REFERENCE_HEADER_LINES = (MODELS_DIR / 'nep_C.txt').read_text().splitlines()


def _read_fixed_width_output(path, ncols, label):
    """Reads a whitespace-separated, fixed-column-count GPUMD output file (thermo.out,
    dpdt.out) with no header line, raising ValueError on any row with the wrong column count
    or non-numeric data rather than a bare numpy exception."""
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != ncols:
                raise ValueError(
                    f'{path}:{lineno}: {label} expects {ncols} columns, got {len(fields)}')
            try:
                rows.append([float(x) for x in fields])
            except ValueError as e:
                raise ValueError(f'{path}:{lineno}: non-numeric field in {label}') from e
    if not rows:
        raise ValueError(f'{path}: no data rows found in {label}')
    return np.array(rows)


def read_thermo_out(path):
    """Reads a thermo.out file; see doc/gpumd/output_files/thermo_out.rst for the 18-column
    format (T, K, U, Pxx, Pyy, Pzz, Pyz, Pxz, Pxy, ax, ay, az, bx, by, bz, cx, cy, cz)."""
    return _read_fixed_width_output(path, THERMO_OUT_NCOLS, 'thermo.out')


def read_dpdt_out(path):
    """Reads a dpdt.out file; see doc/gpumd/output_files/dpdt_out.rst for the 7-column format
    (time, dPx/dt, dPy/dt, dPz/dt, Px, Py, Pz)."""
    return _read_fixed_width_output(path, DPDT_OUT_NCOLS, 'dpdt.out')


@pytest.mark.parametrize('model_file', ALL_MODEL_FILES, ids=lambda p: p.name)
def test_read_model_round_trip(model_file):
    """Every toy model file supplied for this suite should parse without error and report
    metadata consistent with a real nep.txt file."""
    model = read_model(str(model_file))
    assert model.types
    assert model.version in (3, 4, 5)


def test_read_thermo_out_round_trip(tmp_path):
    path = tmp_path / 'thermo.out'
    row = ' '.join(f'{1.0 + i:.6e}' for i in range(THERMO_OUT_NCOLS))
    path.write_text(row + '\n' + row + '\n')
    data = read_thermo_out(path)
    assert data.shape == (2, THERMO_OUT_NCOLS)


def test_read_thermo_out_rejects_wrong_column_count(tmp_path):
    path = tmp_path / 'thermo.out'
    path.write_text(' '.join(['1.0'] * (THERMO_OUT_NCOLS - 1)) + '\n')
    with pytest.raises(ValueError):
        read_thermo_out(path)


def test_read_dpdt_out_round_trip(tmp_path):
    path = tmp_path / 'dpdt.out'
    row = ' '.join(f'{1.0 + i:.6e}' for i in range(DPDT_OUT_NCOLS))
    path.write_text(row + '\n')
    data = read_dpdt_out(path)
    assert data.shape == (1, DPDT_OUT_NCOLS)


def test_read_dpdt_out_rejects_non_numeric_field(tmp_path):
    path = tmp_path / 'dpdt.out'
    fields = ['1.0'] * DPDT_OUT_NCOLS
    fields[3] = 'nan_but_not_really'
    path.write_text(' '.join(fields) + '\n')
    with pytest.raises(ValueError):
        read_dpdt_out(path)


@given(text=st.text())
@settings(max_examples=200)
def test_read_model_never_crashes_on_arbitrary_text(text):
    """Parser should raise one of its documented exception types on malformed input, never an
    unhandled exception (KeyError, IndexError, etc.) or silent wrong-value return.

    Uses its own temporary directory per example rather than the tmp_path fixture, since
    tmp_path is function-scoped and hypothesis warns (FailedHealthCheck) when a function-scoped
    fixture isn't reset between generated examples.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'nep.txt'
        path.write_text(text)
        try:
            read_model(str(path))
        except EXPECTED_MODEL_PARSE_EXCEPTIONS:
            pass


@st.composite
def corrupted_nep_header(draw):
    """Mutates a real nep.txt's lines: truncate, drop a line, blank a line, or garble one
    token — more likely to trip real parsing bugs than pure random text once the trivial crash
    cases are covered."""
    lines = list(REFERENCE_HEADER_LINES)
    corruption = draw(st.sampled_from(['truncate', 'drop_line', 'blank_line', 'garble_token']))
    if corruption == 'truncate':
        n = draw(st.integers(min_value=0, max_value=len(lines)))
        lines = lines[:n]
    elif corruption == 'drop_line' and len(lines) > 1:
        idx = draw(st.integers(min_value=0, max_value=len(lines) - 1))
        del lines[idx]
    elif corruption == 'blank_line' and lines:
        idx = draw(st.integers(min_value=0, max_value=len(lines) - 1))
        lines[idx] = ''
    elif corruption == 'garble_token' and lines:
        idx = draw(st.integers(min_value=0, max_value=min(len(lines), 6) - 1))
        tokens = lines[idx].split()
        if tokens:
            pos = draw(st.integers(min_value=0, max_value=len(tokens) - 1))
            tokens[pos] = draw(st.text(min_size=1, max_size=6))
            lines[idx] = ' '.join(tokens)
    return '\n'.join(lines)


@given(text=corrupted_nep_header())
@settings(max_examples=150)
def test_read_model_never_crashes_on_corrupted_header(text):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'nep.txt'
        path.write_text(text)
        try:
            read_model(str(path))
        except EXPECTED_MODEL_PARSE_EXCEPTIONS:
            pass


@st.composite
def malformed_fixed_width_rows(draw, ncols):
    n_good_rows = draw(st.integers(min_value=0, max_value=5))
    good_rows = []
    for _ in range(n_good_rows):
        values = draw(st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=ncols, max_size=ncols,
        ))
        good_rows.append(' '.join(f'{v:.6e}' for v in values))
    corruption = draw(st.sampled_from(['none', 'wrong_ncols', 'non_numeric', 'blank_row']))
    if corruption == 'wrong_ncols':
        bad_ncols = draw(st.integers(min_value=0, max_value=ncols * 2).filter(
            lambda n: n != ncols))
        good_rows.append(' '.join(str(i) for i in range(bad_ncols)))
    elif corruption == 'non_numeric':
        good_rows.append(' '.join(['not_a_number'] * ncols))
    elif corruption == 'blank_row':
        good_rows.append('')
    return '\n'.join(good_rows)


@given(text=malformed_fixed_width_rows(THERMO_OUT_NCOLS))
@settings(max_examples=100)
def test_read_thermo_out_never_crashes(text):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'thermo.out'
        path.write_text(text)
        try:
            read_thermo_out(path)
        except ValueError:
            pass


@given(text=malformed_fixed_width_rows(DPDT_OUT_NCOLS))
@settings(max_examples=100)
def test_read_dpdt_out_never_crashes(text):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'dpdt.out'
        path.write_text(text)
        try:
            read_dpdt_out(path)
        except ValueError:
            pass
