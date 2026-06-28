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

"""Regenerate the frozen GPUMD reference fixtures in ``tests/data/*.gpumd.npz``.

Runs GPUMD's ``nep`` binary in prediction mode for every fixture in
``_common.FIXTURES`` and saves the resulting predicted E/F/V as a single
.npz file per fixture. Tests then read these arrays back instead of
invoking GPUMD or any third-party package.

Run only when the GPUMD output format or the fixture nep.txt changes:

    GPUMD_NEP=/path/to/nep python bake_fixtures.py

Default GPUMD path: /u/22/wuy33/unix/Study/GPUMD/src/nep
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from torchnep.data import read_xyz, parse_nep_in
from torchnep.model import NEPModel
from _common import (FIXTURES, DATA_DIR, parse_nep_header, write_gpumd_xyz,
                     write_nep_in, write_virial_mix_xyz, QSCALER_NEP_IN)


GPUMD_NEP = os.environ.get("GPUMD_NEP", "/u/22/wuy33/unix/Study/GPUMD/src/nep")


def _run_gpumd(workdir: Path) -> None:
    t0 = time.time()
    res = subprocess.run([GPUMD_NEP], cwd=str(workdir),
                         capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        sys.stderr.write("=== GPUMD stdout (tail) ===\n" + res.stdout[-2000:])
        sys.stderr.write("=== GPUMD stderr (tail) ===\n" + res.stderr[-2000:])
        raise RuntimeError(f"GPUMD nep exited with code {res.returncode}")
    print(f"    GPUMD took {time.time()-t0:.1f}s", flush=True)


def bake_one(fixture: dict) -> None:
    print(f"baking {fixture['name']}", flush=True)
    frames = read_xyz(str(fixture["xyz"]))
    hdr = parse_nep_header(fixture["nep"])

    workdir = Path(tempfile.mkdtemp(prefix=f"bake_{fixture['name']}_"))
    try:
        # nep.txt
        shutil.copy(fixture["nep"], workdir / "nep.txt")
        # train.xyz (inject dummy energy / forces if absent)
        write_gpumd_xyz(frames, workdir / "train.xyz")
        # nep.in (with descriptor mode 2 = per-atom row)
        write_nep_in(hdr, workdir / "nep.in", output_descriptor=2)

        _run_gpumd(workdir)

        # GPUMD output schemas:
        #   energy_train.out  shape (Nframes, 2)   col 0 = predicted E/atom
        #   force_train.out   shape (Natoms_total, 6)   col 0..2 = predicted F
        #   virial_train.out  shape (Nframes, 12)  col 0..5 = predicted V/atom
        #   descriptor.out    shape (Natoms_total, dim) — scaled q (mode 2)
        e = np.loadtxt(workdir / "energy_train.out")
        f = np.loadtxt(workdir / "force_train.out")
        v = np.loadtxt(workdir / "virial_train.out")
        d = np.loadtxt(workdir / "descriptor.out")

        if e.ndim == 1:
            e = e.reshape(1, -1)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if f.ndim == 1:
            f = f.reshape(1, -1)
        if d.ndim == 1:
            d = d.reshape(1, -1)

        E_pa = e[:, 0].astype(np.float64)
        F = f[:, :3].astype(np.float64)
        V_pa = v[:, :6].astype(np.float64)
        D_pa = d.astype(np.float64)

        np.savez(fixture["ref"],
                 E_per_atom=E_pa, F=F, V_per_atom=V_pa, D_per_atom=D_pa)
        print(f"    wrote {fixture['ref']}  "
              f"(E:{E_pa.shape} F:{F.shape} V:{V_pa.shape} D:{D_pa.shape})")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def bake_virial_mix() -> None:
    """Bake GPUMD's full virial_train.out / stress_train.out (both predicted
    and reference columns) for the mixed-virial fixture, so the output-parity
    test can check predict_dataset against GPUMD without a GPUMD build."""
    print("baking virial_mix", flush=True)
    nep = DATA_DIR / "nep_CrCoNi.txt"
    frames = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))
    hdr = parse_nep_header(nep)

    workdir = Path(tempfile.mkdtemp(prefix="bake_virial_mix_"))
    try:
        shutil.copy(nep, workdir / "nep.txt")
        write_virial_mix_xyz(frames, workdir / "train.xyz")
        write_nep_in(hdr, workdir / "nep.in", output_descriptor=0)
        _run_gpumd(workdir)

        # 12 columns each: 0..5 predicted (per-atom virial / stress in GPa),
        # 6..11 reference (-1e6 sentinel where the frame has no virial).
        v_out = np.loadtxt(workdir / "virial_train.out").reshape(-1, 12)
        s_out = np.loadtxt(workdir / "stress_train.out").reshape(-1, 12)
        ref = DATA_DIR / "virial_mix.gpumd.npz"
        np.savez(ref, virial_out=v_out, stress_out=s_out)
        print(f"    wrote {ref}  (virial:{v_out.shape} stress:{s_out.shape})")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def bake_qscaler() -> None:
    """Bake GPUMD's generation-0 q_scaler (descriptor coefficients = 1.0,
    1/(max-min) per dimension over the full CrCoNi set). Lets the q_scaler
    parity test check compute_q_scaler against GPUMD without a GPUMD build."""
    print("baking qscaler", flush=True)
    frames = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))
    workdir = Path(tempfile.mkdtemp(prefix="bake_qscaler_"))
    try:
        (workdir / "nep.in").write_text(QSCALER_NEP_IN)
        write_gpumd_xyz(frames, workdir / "train.xyz")
        _run_gpumd(workdir)
        # q_scaler is the trailing `dim` single-value lines of nep.txt.
        dim = NEPModel(parse_nep_in(str(workdir / "nep.in"))).dim
        vals = [float(p[0]) for p in
                (ln.split() for ln in (workdir / "nep.txt").read_text().splitlines())
                if len(p) == 1]
        q_scaler = np.array(vals[-dim:])
        ref = DATA_DIR / "qscaler_CrCoNi.gpumd.npz"
        np.savez(ref, q_scaler=q_scaler)
        print(f"    wrote {ref}  (q_scaler:{q_scaler.shape})")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    if not Path(GPUMD_NEP).is_file():
        sys.exit(f"GPUMD nep binary not found: {GPUMD_NEP}")
    print(f"GPUMD nep: {GPUMD_NEP}\n")

    for fx in FIXTURES:
        bake_one(fx)
    bake_virial_mix()
    bake_qscaler()

    print("\nAll fixtures regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
