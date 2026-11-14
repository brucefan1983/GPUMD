from validate_spin_fire_runtime import GPUMD
from validate_spin_large_cell_runtime import validate
import pytest


def test_spin_large_cell(tmp_path):
    if not GPUMD.exists():
        pytest.skip('GPUMD executable not built')
    validate(tmp_path)
