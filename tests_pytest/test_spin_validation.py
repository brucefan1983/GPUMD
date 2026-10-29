"""CPU guards for the numerical validation machinery."""
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
