"""CPU guards for the numerical validation machinery and response geometry."""
import shutil
import subprocess
from pathlib import Path

import pytest

from validate_main_nep_spin3_training import maximum_error, parity_tolerance


@pytest.mark.parametrize("comparison", [maximum_error, parity_tolerance])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_comparison_rejects_nonfinite(comparison, invalid, side):
    values = [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
    values[side][0][1] = invalid
    with pytest.raises(AssertionError, match="nonfinite"):
        comparison(*values)


def test_comparison_finite_values_and_shape():
    assert maximum_error([[0.0, -2.0]], [[0.0, -1.5]]) == 0.5
    assert parity_tolerance([2.0], [-1.0]) == 5e-5 + 5e-6 * 2
    with pytest.raises(AssertionError, match="shape mismatch"):
        maximum_error([0.0], [0.0, 0.0])
    with pytest.raises(AssertionError, match="empty"):
        maximum_error([], [])


def test_rotation_response(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "rotation_response"
    subprocess.run([
        compiler, "-std=c++14", "-pedantic-errors", "-I" + str(root / "src"),
        str(root / "tests_pytest/rotation_response.cpp"), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("line", ["SCALER 1 nan 1", "BASELINE 0 inf", "FIT 0 1 nan", "POINT a nan 0"])
def test_response_probe_rejects_nonfinite(line):
    from validate_response_file_runtime import parsed

    output = "SCALER 1 1 1\nBASELINE 0 0\nFIT 0 0 1\nPOINT a 1 0\n" + line
    with pytest.raises(AssertionError, match="nonfinite"):
        parsed(output)
