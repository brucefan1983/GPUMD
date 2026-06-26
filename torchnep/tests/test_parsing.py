# Copyright 2025 Yongchao Wu and the GPUMD development team
# This file is part of GPUMD (Torchnep project).
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

"""Regression tests for legacy nep.in / nep.txt parsing.

Pins down acceptance of the historical 2-field ``l_max <L> <l_max_4b>``
form (the second field used to be "max L for the 4-body invariant" but had
boolean semantics everywhere), and the new layout that matches GPUMD's
parser after PR #1519 removed q_1122:

    l_max <L_3b> <has_q_222> <has_q_1111> <has_q_112>
          [<has_q_123> <has_q_233>]

Anything past index 0 is treated as a boolean flag.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator


def _minimal_config(l_max_list):
    return {
        "num_types": 1, "type_names": ["X"],
        "cutoff_radial": 6.0, "cutoff_angular": 5.0,
        "n_max_radial": 4, "n_max_angular": 4,
        "basis_size_radial": 8, "basis_size_angular": 8,
        "l_max": l_max_list, "neuron": 10,
    }


@pytest.mark.parametrize("l_max,expected", [
    # 3-field legacy form (used by every nep.in / nep.txt before the
    # mixed-body extension; '2' and '1' were "max L used in 4b/5b" but
    # had boolean semantics everywhere they entered the code).
    ([4, 2, 1], (4, 1, 1, 0)),
    ([4, 2, 0], (4, 1, 0, 0)),
    ([4, 0, 0], (4, 0, 0, 0)),
    # 4-field form: full GPUMD-core flags
    ([4, 1, 1, 1],    (4, 1, 1, 1)),
    ([4, 1, 0, 1],    (4, 1, 0, 1)),
    ([4, 1, 0, 0],    (4, 1, 0, 0)),
    # 1-field (3-body only; future-compat — every trailing flag defaults to 0)
    ([4],             (4, 0, 0, 0)),
])
def test_l_max_field_normalisation(l_max, expected):
    """Trailing fields normalise to 0/1 regardless of input width."""
    m = NEPModel(_minimal_config(l_max))
    got = (m.l_max_3b, m.has_q_222, m.has_q_1111, m.has_q_112)
    assert got == expected


def test_legacy_3field_nep_txt_loads():
    """Existing nep_CrCoNi.txt with ``l_max 4 2 1`` still loads correctly.

    Catches regressions where parsing of older nep.txt files breaks after
    a refactor of the header parser.
    """
    nep_txt = Path(__file__).resolve().parent / "data" / "nep_CrCoNi.txt"
    calc = NEPCalculator(str(nep_txt))
    assert calc.l_max_3b == 4
    assert calc.has_q_222 == 1
    assert calc.has_q_1111 == 1
    assert calc.has_q_112 == 0
    # Sanity on derived dim:
    # radial (n_max_r+1=9) + 3-body (9*4=36) + q_222 (9) + q_1111 (9) = 63
    assert calc.dim == 63
