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

"""pytest conftest — sets sys.path so ``torchnep`` is importable when running
``pytest`` from the repo root or from ``tests/`` itself, with no need for
``pip install -e .``.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

for p in (ROOT, TESTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Keep OpenMP / numpy thread fan-out predictable in CI.
os.environ.setdefault("TORCHNEP_PREPROC_WORKERS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
