# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime regression for the fixed-lattice spin minimizer."""
import pytest
from validate_spin_fire_runtime import GPUMD, validate


def test_spin_fire(tmp_path):
    if not GPUMD.is_file():
        pytest.skip('GPUMD executable is not built')
    validate(tmp_path)
