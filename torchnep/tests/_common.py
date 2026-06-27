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

"""Shared helpers for the torchnep test suite.

Only depends on the standard library + numpy + torch + torchnep itself.
No third-party packages (mdapy / polars / etc.) — predictions are checked
against frozen GPUMD-baked fixtures in ``data/*.gpumd.npz``.
"""
import os
from pathlib import Path
from typing import List

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent / "data"

DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64}
NP_DTYPE_MAP = {"float32": np.float32, "float64": np.float64}


# --- environment-driven selection (run a subset for CI / debugging) ----------

def devices() -> List[str]:
    """Devices to test on. Honors $TEST_DEVICE (e.g. ``cpu``, ``cuda``)."""
    env = os.environ.get("TEST_DEVICE")
    if env:
        return [env]
    ds = ["cpu"]
    if torch.cuda.is_available():
        ds.append("cuda")
    return ds


def dtypes() -> List[str]:
    """Dtypes to test on. Honors $TEST_DTYPE (``float32`` / ``float64``)."""
    env = os.environ.get("TEST_DTYPE")
    if env:
        return [env]
    return ["float32", "float64"]


# --- fixture catalogue --------------------------------------------------------

# Each fixture: (nep.txt, xyz, npz) and the architectural feature it exercises.
# When adding a fixture, also rebake the .gpumd.npz via ``bake_fixtures.py``.
FIXTURES = [
    {
        "name":  "CrCoNi",
        "nep":   DATA_DIR / "nep_CrCoNi.txt",
        "xyz":   DATA_DIR / "CrCoNi.xyz",
        "ref":   DATA_DIR / "CrCoNi.gpumd.npz",
        "note":  "nep4_zbl, typewise ZBL, l_max 4 2 1 (legacy 3-field). "
                 "Multi-frame: original (NN ~1.5 A) plus compressed/rattled "
                 "frames down to NN ~1.18 A that drive the ZBL repulsion "
                 "to/below its inner cutoff (1.25 A).",
    },
]


def load_reference(npz_path: Path) -> dict:
    """Load a frozen GPUMD reference fixture.

    Returns a dict with keys:
      ``E_per_atom``  : (Nframes,)            eV/atom
      ``F``           : (Natoms, 3)           eV/A
      ``V_per_atom``  : (Nframes, 6)          eV, GPUMD-ordered (xx,yy,zz,xy,yz,zx)
      ``D_per_atom``  : (Natoms, dim) or None scaled descriptor (q * q_scaler);
                                              ``None`` if the fixture pre-dates
                                              descriptor baking.
    """
    z = np.load(npz_path)
    return {
        "E_per_atom": z["E_per_atom"],
        "F":          z["F"],
        "V_per_atom": z["V_per_atom"],
        "D_per_atom": z["D_per_atom"] if "D_per_atom" in z.files else None,
    }


def write_gpumd_xyz(frames, dst_path: Path) -> None:
    """Write a list of read_xyz()-style frames to a GPUMD-compatible xyz.

    GPUMD prediction-mode requires ``energy=`` and a ``force:R:3`` column;
    we inject zero-filled defaults when the source frame doesn't carry them.
    """
    lines = []
    for fr in frames:
        natoms = fr["natoms"]
        cell = np.asarray(fr["cell"]).reshape(-1)
        cell_s = " ".join(f"{x:.10f}" for x in cell)
        energy = fr.get("energy", 0.0)
        if energy is None:
            energy = 0.0
        forces = fr.get("forces")
        if forces is None:
            forces = np.zeros((natoms, 3), dtype=float)
        lines.append(f"{natoms}")
        lines.append(
            f'pbc="T T T" Lattice="{cell_s}" energy={energy:.10f} '
            f'Properties=species:S:1:pos:R:3:force:R:3'
        )
        positions = np.asarray(fr["positions"])
        species = list(fr["species"])
        for i in range(natoms):
            x, y, z = positions[i]
            fx, fy, fz = forces[i]
            lines.append(
                f"{species[i]} {x:.10f} {y:.10f} {z:.10f} "
                f"{fx:.10f} {fy:.10f} {fz:.10f}")
    dst_path.write_text("\n".join(lines) + "\n")


# Mixed-virial fixture: first three CrCoNi frames, with an explicit virial tag
# on frames 0 and 2 and NONE on frame 1. Exercises predict_dataset's output
# parity with GPUMD for both the present-virial path (reference scaled to
# stress) and the missing-virial path (reference written as the -1e6 sentinel
# in both virial_train.out and stress_train.out). The baker and the test both
# build the input from this single definition so they stay in lock-step.
VIRIAL_MIX_NAME = "virial_mix"
VIRIAL_MIX_NFRAMES = 3
VIRIAL_MIX = {            # frame index -> 9-component (row-major) virial string
    0: "12 1 2 1 25 3 2 3 31",
    2: "-8 0.5 0 0.5 -6 1 0 1 -7",
}


def write_virial_mix_xyz(frames, dst_path: Path) -> None:
    """Write the mixed-virial fixture (GPUMD-compatible: energy + force column,
    plus a virial tag on the frames named in ``VIRIAL_MIX``). Forces are zero —
    only the virial/stress columns are under test."""
    lines = []
    for k, fr in enumerate(frames[:VIRIAL_MIX_NFRAMES]):
        na = fr["natoms"]
        cell = " ".join(f"{x:.10f}" for x in np.asarray(fr["cell"]).reshape(-1))
        tag = f"energy={-3.0 * na:.6f} "
        if k in VIRIAL_MIX:
            tag += f'virial="{VIRIAL_MIX[k]}" '
        tag += (f'pbc="T T T" Lattice="{cell}" '
                f'Properties=species:S:1:pos:R:3:force:R:3')
        lines.append(str(na))
        lines.append(tag)
        pos = np.asarray(fr["positions"])
        sp = list(fr["species"])
        for i in range(na):
            x, y, z = pos[i]
            lines.append(f"{sp[i]} {x:.10f} {y:.10f} {z:.10f} 0.0 0.0 0.0")
    dst_path.write_text("\n".join(lines) + "\n")


def parse_nep_header(nep_path: Path) -> dict:
    """Extract just the architecture fields we need to reconstruct nep.in for
    GPUMD prediction mode (cutoff / n_max / basis_size / l_max / neuron /
    type / zbl)."""
    out = {}
    with open(nep_path) as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            try:
                float(parts[0])
                break               # numeric body starts -> header done
            except ValueError:
                pass
            key = parts[0]
            if key.startswith("nep"):
                # version-and-types line: "nep4[_zbl] N T1 T2 ..."
                out["version_tag"] = key
                out["num_types"] = int(parts[1])
                out["type_names"] = parts[2:2 + out["num_types"]]
            elif key == "zbl":
                # "zbl <rc_inner> <rc_outer>" or with typewise factor at end
                if len(parts) == 4:
                    out["zbl_outer"] = float(parts[2])
                    out["zbl_factor"] = float(parts[3])
                else:
                    out["zbl_outer"] = float(parts[2])
            elif key == "cutoff":
                out["rc_radial"]  = float(parts[1])
                out["rc_angular"] = float(parts[2])
            elif key == "n_max":
                out["n_max_radial"]  = int(parts[1])
                out["n_max_angular"] = int(parts[2])
            elif key == "basis_size":
                out["basis_size_radial"]  = int(parts[1])
                out["basis_size_angular"] = int(parts[2])
            elif key == "l_max":
                out["l_max"] = [int(x) for x in parts[1:]]
            elif key == "ANN":
                out["neuron"] = int(parts[1])
    return out


def write_nep_in(hdr: dict, dst: Path, output_descriptor: int = 0) -> None:
    """Write a nep.in that mirrors the architecture of a nep.txt for GPUMD
    prediction mode.

    ``output_descriptor`` in {0, 1, 2}: 0 disables descriptor.out; 1 writes
    one row per frame (averaged); 2 writes one row per atom.
    """
    lines = [
        f"type {hdr['num_types']} {' '.join(hdr['type_names'])}",
        "prediction 1",
        "batch 1",
        f"cutoff {int(hdr['rc_radial'])} {int(hdr['rc_angular'])}",
        f"n_max {hdr['n_max_radial']} {hdr['n_max_angular']}",
        f"basis_size {hdr['basis_size_radial']} {hdr['basis_size_angular']}",
        # Pad l_max to 5 fields with zeros (new GPUMD format accepts 1–5
        # but writes 5 by default; we feed it the form it expects).
        f"l_max {' '.join(str(x) for x in (hdr['l_max'] + [0] * 5)[:5])}",
        f"neuron {hdr['neuron']}",
    ]
    if "zbl_outer" in hdr:
        lines.append(f"zbl {hdr['zbl_outer']}")
        if "zbl_factor" in hdr:
            lines.append(f"use_typewise_cutoff_zbl {hdr['zbl_factor']}")
    if output_descriptor:
        lines.append(f"output_descriptor {output_descriptor}")
    dst.write_text("\n".join(lines) + "\n")
