import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
TRAIN_XYZ = (
    REPO_ROOT
    / "tests/nep/norm-loss/model-gpumd-fork-test/nepmodel_full/train.xyz"
)
NEP_BINARY = REPO_ROOT / "src/nep"

# Minimal fast architecture: small cutoff, few features, small ANN
BASE_NEP_IN = """\
version 4
type 3 Sr Ti O
cutoff 5 3
n_max 4 2
l_max 2 0
neuron 10
generation 1000
"""


@pytest.fixture(scope="session")
def nep_binary():
    if not NEP_BINARY.exists():
        pytest.skip(f"nep binary not found at {NEP_BINARY}; build with 'make' in src/")
    return NEP_BINARY


@pytest.fixture
def run_nep(nep_binary, tmp_path):
    """Returns a callable run(extra_lines) that trains nep in tmp_path.

    Copies train.xyz, writes nep.in = BASE_NEP_IN + extra_lines, runs nep,
    and returns the run directory.  Raises RuntimeError on non-zero exit.
    """

    def _run(extra_lines=""):
        shutil.copy(TRAIN_XYZ, tmp_path / "train.xyz")
        (tmp_path / "nep.in").write_text(BASE_NEP_IN + "\n" + extra_lines)
        result = subprocess.run(
            [str(nep_binary)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=600,
        )
        (tmp_path / "nep.stdout").write_text(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"nep exited with code {result.returncode}\n"
                f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
                f"--- stderr ---\n{result.stderr[-2000:]}"
            )
        return tmp_path

    return _run


# ── Synthetic single-element dataset for manually-verifiable loss tests ────────
#
# 3-structure Cu dataset in a 10 Å box.  The reference energies are small and
# positive (~0.3–0.4 eV/atom) so energy_train.out (%g format, 6 sig figs) has
# no catastrophic-cancellation problem and the manual formula check can use a
# tight tolerance.
#
# Atoms are placed 2.8–3.2 Å apart (within the 4 Å cutoff) so each atom has
# meaningful neighbors and the NEP descriptor is non-trivial.
# Forces are chosen to sum to zero within each structure.
# Stresses are in eV/Å³; volume = 10³ = 1000 Å³.
SYNTHETIC_TRAIN_XYZ = """\
3
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=0.90 stress="-0.001 0.0 0.0 0.0 -0.002 0.0 0.0 0.0 -0.001" pbc="T T T"
Cu  1.0 1.0 1.0   0.50  0.30 -0.20
Cu  3.8 1.0 1.0  -0.40  0.10  0.10
Cu  1.0 3.8 1.0  -0.10 -0.40  0.10
3
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=1.20 stress="-0.002 0.0 0.0 0.0 -0.001 0.0 0.0 0.0 -0.003" pbc="T T T"
Cu  1.0 1.0 1.0   0.30 -0.20  0.40
Cu  4.2 1.0 1.0  -0.30  0.30 -0.20
Cu  1.0 4.2 1.0   0.00 -0.10 -0.20
4
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=1.60 stress="-0.001 0.0 0.0 0.0 -0.002 0.0 0.0 0.0 -0.002" pbc="T T T"
Cu  1.0 1.0 1.0   0.40  0.20 -0.10
Cu  3.8 1.0 1.0  -0.30  0.10  0.20
Cu  1.0 3.8 1.0  -0.20 -0.10  0.00
Cu  3.8 3.8 1.0   0.10 -0.20 -0.10
"""

SYNTHETIC_NEP_HEADER = """\
version 4
type 1 Cu
cutoff 4.0 3.5
n_max 4 2
l_max 2 0
neuron 10
generation 1000
"""


@pytest.fixture
def run_nep_xyz(nep_binary, tmp_path):
    """Like run_nep but uses the synthetic Cu dataset and SYNTHETIC_NEP_HEADER."""

    def _run(extra_lines=""):
        (tmp_path / "train.xyz").write_text(SYNTHETIC_TRAIN_XYZ)
        (tmp_path / "nep.in").write_text(SYNTHETIC_NEP_HEADER + "\n" + extra_lines)
        result = subprocess.run(
            [str(nep_binary)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=600,
        )
        (tmp_path / "nep.stdout").write_text(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"nep exited with code {result.returncode}\n"
                f"--- stdout ---\n{result.stdout[-2000:]}\n"
                f"--- stderr ---\n{result.stderr[-2000:]}"
            )
        return tmp_path

    return _run
