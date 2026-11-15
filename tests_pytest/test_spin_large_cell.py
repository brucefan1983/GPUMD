# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.

from validate_spin_fire_runtime import GPUMD
from validate_spin_large_cell_runtime import validate
import pytest


def test_spin_large_cell(tmp_path):
    if not GPUMD.exists():
        pytest.skip('GPUMD executable not built')
    validate(tmp_path)
