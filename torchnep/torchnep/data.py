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

"""
Data loading utilities for NEP training and prediction.

Supports extended XYZ format (as used by GPUMD) and nep.in parameter files.
"""

import numpy as np
from typing import Dict, List


def _parse_properties_schema(comment: str):
    """Parse the extended-XYZ ``Properties=...`` field.

    Returns a dict mapping field name -> (type_code, token_offset, width).
    ``token_offset`` is the starting column in the FULL per-atom token list
    (so species and numeric fields share the same coordinate). This lets
    downstream parsing handle arbitrary field ordering, including the
    ``pos:R:3:species:S:1`` form produced by some exporters.
    """
    # Case-insensitive match: ASE uses "Properties=" while some exporters
    # use "properties=". Look in a lower-cased copy but slice from original
    # so we preserve field names (species, pos, force, ...).
    key = "properties="
    idx = comment.lower().find(key)
    if idx < 0:
        return None
    start = idx + len(key)
    end = start
    while end < len(comment) and comment[end] not in (" ", "\t"):
        end += 1
    spec = comment[start:end]
    toks = spec.split(":")
    if len(toks) % 3 != 0:
        return None
    schema = {}
    col = 0
    for i in range(0, len(toks), 3):
        name, tp, cnt = toks[i], toks[i + 1], int(toks[i + 2])
        schema[name] = (tp, col, cnt)
        col += cnt
    return schema


def _parse_frame_block(block, energy_key="energy"):
    """Parse one frame from a list of text lines (picklable for mp.Pool).

    Reads extended XYZ with a ``Properties=...`` schema. Only
    ``species:S:1``, ``pos:R:3``, and ``force:R:3`` / ``forces:R:3`` are
    consumed — any other columns are silently ignored. Energy / virial /
    stress / lattice come from the comment-line key=value tags; see
    ``_parse_comment`` for strict-mode validation rules.
    """
    natoms = int(block[0].strip())
    comment = block[1].strip()
    frame = _parse_comment(comment, natoms, energy_key=energy_key)

    schema = _parse_properties_schema(comment)
    if schema is None or "pos" not in schema:
        raise ValueError(
            "extended-xyz parser requires a Properties=... header with "
            "at least species and pos fields; got: " + comment[:120])

    # Find species column (only field with type 'S')
    species_key = next((k for k, v in schema.items() if v[0] == "S"), None)
    if species_key is None:
        raise ValueError("Properties schema missing species (type S) field")
    _, sp_col, _ = schema[species_key]

    atoms = block[2:2 + natoms]
    species = [None] * natoms
    numeric_rows = []
    for j, line in enumerate(atoms):
        toks = line.split()
        species[j] = toks[sp_col]
        numeric_rows.append([t for k, t in enumerate(toks) if k != sp_col])

    ncol_numeric = len(numeric_rows[0])
    flat = " ".join(" ".join(r) for r in numeric_rows)
    arr = np.fromstring(flat, sep=" ", dtype=np.float64)
    arr = arr.reshape(natoms, ncol_numeric)

    frame["natoms"] = natoms
    frame["species"] = species

    def _numeric_offset(field_col):
        return field_col if field_col < sp_col else field_col - 1

    _, pos_col, pos_w = schema["pos"]
    pos_off = _numeric_offset(pos_col)
    frame["positions"] = arr[:, pos_off:pos_off + pos_w].copy()

    force_key = "force" if "force" in schema else (
                "forces" if "forces" in schema else None)
    if force_key is not None:
        _, f_col, f_w = schema[force_key]
        f_off = _numeric_offset(f_col)
        frame["forces"] = arr[:, f_off:f_off + f_w].copy()
    return frame


def _split_frames(lines):
    """Split an XYZ text into per-frame line blocks."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        natoms = int(lines[i].strip())
        end = i + 2 + natoms
        blocks.append(lines[i:end])
        i = end
    return blocks


def read_xyz(filename: str, energy_key: str = "energy") -> List[Dict]:
    """Read extended XYZ file (GPUMD format).

    Parameters
    ----------
    filename : str
        Path to the .xyz file.
    energy_key : str
        Name of the per-frame energy tag to read. Defaults to ``"energy"``;
        set to e.g. ``"atomization_energy"`` to use atomization energies
        instead of total energies.
    """
    with open(filename) as f:
        lines = f.readlines()

    blocks = _split_frames(lines)
    del lines
    return [_parse_frame_block(b, energy_key=energy_key) for b in blocks]


def _find_quoted(comment: str, key: str):
    """Return the content inside key="..." or None if absent.

    Matches ``key`` case-insensitively (e.g. ``Virial="..."`` and
    ``virial="..."`` both work — GPUMD / ASE / pymatgen differ on casing).
    """
    low = comment.lower()
    needle = key.lower() + '="'
    i = low.find(needle)
    if i < 0:
        return None
    i += len(needle)
    j = comment.find('"', i)
    if j < 0:
        return None
    return comment[i:j]


def _find_scalar(comment: str, key: str):
    """Return the string value after ``key=`` (unquoted) or None if absent.

    Matches ``key`` case-insensitively. Only returns a match when ``key=``
    is preceded by whitespace or start of string — prevents
    ``atomization_energy=`` from matching ``energy=``.
    """
    low = comment.lower()
    needle = key.lower() + "="
    start = 0
    while True:
        i = low.find(needle, start)
        if i < 0:
            return None
        if i == 0 or low[i - 1] in (" ", "\t"):
            i += len(needle)
            j = i
            while j < len(comment) and comment[j] not in (" ", "\t", '"'):
                j += 1
            return comment[i:j]
        start = i + 1


def _parse_comment(comment: str, natoms: int, energy_key: str = "energy") -> Dict:
    """Parse extended XYZ comment line with strict-mode validation.

    Rules:
      - ``Lattice="ax ay az bx by bz cx cy cz"`` is mandatory (9 floats).
        Every frame is treated as fully periodic; the ``pbc=`` tag is
        ignored (isolated clusters / molecules must be wrapped in a large
        vacuum box by the user before loading).
      - Energy tag name is configurable via ``energy_key`` (default "energy");
        if missing, the frame simply has no energy (handled downstream).
      - ``virial="..."`` and ``stress="..."`` are optional but, when present,
        must have exactly 9 components. If both are given, virial wins.
        stress (eV/A**3) is converted to virial (eV) as
        ``virial = -stress * |det(lattice)|`` — opposite sign convention.
    """
    frame = {}

    lat_str = _find_quoted(comment, "Lattice")
    if lat_str is None:
        raise ValueError(
            "extended-xyz frame is missing mandatory Lattice=\"...\" tag; "
            "comment: " + comment[:160])
    lat_vals = [float(x) for x in lat_str.split()]
    if len(lat_vals) != 9:
        raise ValueError(
            f"Lattice must have exactly 9 components, got {len(lat_vals)}; "
            "comment: " + comment[:160])
    frame["cell"] = np.array(lat_vals).reshape(3, 3)

    e_val = _find_scalar(comment, energy_key)
    if e_val is not None:
        frame["energy"] = float(e_val)

    vir_str = _find_quoted(comment, "virial")
    if vir_str is not None:
        vir_vals = [float(x) for x in vir_str.split()]
        if len(vir_vals) != 9:
            raise ValueError(
                f"virial must have exactly 9 components, got {len(vir_vals)}; "
                "comment: " + comment[:160])
        frame["virial"] = np.array(vir_vals)
    else:
        stress_str = _find_quoted(comment, "stress")
        if stress_str is not None:
            stress_vals = [float(x) for x in stress_str.split()]
            if len(stress_vals) != 9:
                raise ValueError(
                    f"stress must have exactly 9 components, got "
                    f"{len(stress_vals)}; comment: " + comment[:160])
            volume = abs(float(np.linalg.det(frame["cell"])))
            frame["virial"] = -np.array(stress_vals) * volume

    return frame


def parse_nep_in(filename: str) -> Dict:
    """Parse nep.in parameter file.

    Parameters
    ----------
    filename : str
        Path to nep.in file.

    Returns
    -------
    dict
        Dictionary of NEP parameters.
    """
    params = {}

    with open(filename) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0].lower()

            if key == "type":
                params["num_types"] = int(parts[1])
                params["type_names"] = parts[2 : 2 + int(parts[1])]
            elif key == "version":
                v = int(parts[1])
                # torchnep only implements NEP4.
                if v != 4:
                    raise ValueError(
                        f"nep.in version {v} is not supported — torchnep "
                        f"only implements NEP4 (set 'version 4').")
                params["version"] = v
            elif key == "zbl":
                params["zbl"] = float(parts[1])
            elif key == "use_typewise_cutoff_zbl":
                params["typewise_cutoff_zbl_factor"] = float(parts[1])
            elif key == "cutoff":
                params["cutoff_radial"] = float(parts[1])
                params["cutoff_angular"] = float(parts[2])
            elif key == "n_max":
                params["n_max_radial"] = int(parts[1])
                params["n_max_angular"] = int(parts[2])
            elif key == "basis_size":
                params["basis_size_radial"] = int(parts[1])
                params["basis_size_angular"] = int(parts[2])
            elif key == "l_max":
                params["l_max"] = [int(x) for x in parts[1:]]
            elif key == "neuron":
                params["neuron"] = int(parts[1])
            elif key == "lambda_1":
                params["lambda_1"] = float(parts[1])
            elif key == "lambda_e":
                params["lambda_e"] = float(parts[1])
            elif key == "lambda_f":
                params["lambda_f"] = float(parts[1])
            elif key == "lambda_v":
                params["lambda_v"] = float(parts[1])
            elif key == "lambda_2":
                params["lambda_2"] = float(parts[1])
            elif key == "batch":
                params["batch_size"] = int(parts[1])
            elif key == "save_potential":
                params["save_interval"] = int(parts[1])
                if len(parts) > 2:
                    params["save_start"] = int(parts[2])
                if len(parts) > 3:
                    params["save_count"] = int(parts[3])
            # --- torchnep training parameters ---
            elif key == "epoch":
                params["num_epochs"] = int(parts[1])
            elif key == "lr":
                params["lr"] = float(parts[1])
            elif key == "scheduler_patience":
                params["scheduler_patience"] = int(parts[1])
            elif key == "scheduler_factor":
                params["scheduler_factor"] = float(parts[1])
            elif key == "stop_lr":
                params["stop_lr"] = float(parts[1])
            elif key == "lr_scheduler":
                # "plateau" (default, ReduceLROnPlateau) or "step" (StepLR)
                mode = parts[1].lower()
                if mode not in ("plateau", "step"):
                    raise ValueError(
                        f"lr_scheduler must be 'plateau' or 'step', got {parts[1]!r}")
                params["lr_scheduler"] = mode
            elif key == "max_grad_norm":
                params["max_grad_norm"] = float(parts[1])
            elif key == "stage2":
                params["stage2"] = int(parts[1]) != 0
            elif key == "start_stage2":
                params["start_stage2"] = int(parts[1])
            elif key == "stage2_lr":
                params["stage2_lr"] = float(parts[1])
            elif key == "stage2_lambda_e":
                params["stage2_pref_e"] = float(parts[1])
            elif key == "stage2_lambda_f":
                params["stage2_pref_f"] = float(parts[1])
            elif key == "stage2_lambda_v":
                params["stage2_pref_v"] = float(parts[1])
            elif key == "stage2_scheduler_patience":
                params["stage2_scheduler_patience"] = int(parts[1])
            elif key == "stage2_scheduler_factor":
                params["stage2_scheduler_factor"] = float(parts[1])

    # Snapshot the explicit (user-set) keys before applying defaults so the
    # trainer can report which values came from nep.in vs which fell back
    # to a default.
    explicit = set(params.keys())

    # Defaults — model architecture
    params.setdefault("version", 4)
    params.setdefault("cutoff_radial", 8.0)
    params.setdefault("cutoff_angular", 4.0)
    params.setdefault("n_max_radial", 6)
    params.setdefault("n_max_angular", 6)
    params.setdefault("basis_size_radial", 6)
    params.setdefault("basis_size_angular", 6)
    params.setdefault("l_max", [4, 1, 0])
    params.setdefault("neuron", 30)

    # Defaults — training hyperparameters (match train_nep / train_nep_sharded)
    params.setdefault("num_epochs", 600)
    params.setdefault("batch_size", 32)
    params.setdefault("lr", 0.01)
    params.setdefault("stop_lr", 1e-6)
    params.setdefault("scheduler_patience", 15)
    params.setdefault("scheduler_factor", 0.7)
    params.setdefault("lr_scheduler", "plateau")
    params.setdefault("max_grad_norm", 10.0)
    params.setdefault("lambda_e", 0.01)
    params.setdefault("lambda_f", 1.0)
    params.setdefault("lambda_v", 0.01)
    params.setdefault("lambda_1", 0.0)
    params.setdefault("lambda_2", 0.0)
    params.setdefault("stage2", False)

    # Defaults for optional stage-2 parameters (only used if stage2=1).
    params.setdefault("stage2_lr", 1e-3)
    params.setdefault("stage2_pref_e", 1.0)
    params.setdefault("stage2_pref_f", 0.05)
    params.setdefault("stage2_pref_v", 0.1)
    # start_stage2 defaults to 0.5 * num_epochs if not set — handled in trainer

    # Stash explicit-key set in the dict itself; consumers can read it (and
    # safely ignore it). Leading underscore so it can't collide with nep.in
    # tokens.
    params["_explicit"] = explicit
    return params


# ---------------------------------------------------------------------------
# Neighbor list construction (numpy, CPU — shared by training and prediction)
# ---------------------------------------------------------------------------

def build_neighbor_list_np(positions, cell, cutoff):
    """Build neighbor list using numpy (for preprocessing). Returns arrays.

    Cell is stored with lattice vectors as ROWS. The perpendicular distance
    between planes spanned by (b,c), (a,c), (a,b) is V/|b*c|, V/|a*c|,
    V/|a*b|; these are ``1/|inv_cell[:,i]|`` (columns of inv_cell are the
    reciprocal vectors). Using rows silently undercounts image replicas for
    heavily skewed triclinic cells and drops real neighbors — bug fixed 2025.

    Input positions may lie outside the primary cell. Under full PBC, physics
    is translation-invariant, so we wrap fractional coordinates into [0, 1)
    before computing ``n_rep``; otherwise atoms far outside the box would
    miss periodic images that the (inside-box) ``n_rep`` estimate doesn't
    cover.
    """
    N = positions.shape[0]
    inv_cell = np.linalg.inv(cell)

    frac = positions @ inv_cell
    frac -= np.floor(frac)
    positions = frac @ cell

    n_rep = [int(np.ceil(cutoff * np.linalg.norm(inv_cell[:, i]))) for i in range(3)]

    a_r = np.arange(-n_rep[0], n_rep[0] + 1)
    b_r = np.arange(-n_rep[1], n_rep[1] + 1)
    c_r = np.arange(-n_rep[2], n_rep[2] + 1)
    shifts_frac = np.stack(np.meshgrid(a_r, b_r, c_r, indexing="ij"), axis=-1)
    shifts_frac = shifts_frac.reshape(-1, 3).astype(positions.dtype)
    shifts_cart = shifts_frac @ cell
    S = shifts_cart.shape[0]

    if N * N * S < 8_000_000:
        disp = (positions[None, :, None, :] + shifts_cart[None, None, :, :]
                - positions[:, None, None, :])
        dist = np.linalg.norm(disp, axis=-1)
        zero_shift = np.all(shifts_frac == 0, axis=1)
        self_mask = np.eye(N, dtype=bool)[:, :, None] & zero_shift[None, None, :]
        valid = (dist < cutoff) & (dist > 1e-10) & ~self_mask
        idx_i, idx_j, idx_s = np.where(valid)
        return idx_i.astype(np.int64), idx_j.astype(np.int64), disp[idx_i, idx_j, idx_s]

    zero_shift = np.all(shifts_frac == 0, axis=1)
    all_i, all_j, all_rij = [], [], []
    for si in range(S):
        shifted = positions + shifts_cart[si]
        disp = shifted[None, :, :] - positions[:, None, :]
        dist = np.linalg.norm(disp, axis=-1)
        valid = (dist < cutoff) & (dist > 1e-10)
        if zero_shift[si]:
            np.fill_diagonal(valid, False)
        ii, jj = np.where(valid)
        if len(ii) > 0:
            all_i.append(ii)
            all_j.append(jj)
            all_rij.append(disp[ii, jj])
    if not all_i:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, 3), positions.dtype))
    return (np.concatenate(all_i).astype(np.int64),
            np.concatenate(all_j).astype(np.int64),
            np.concatenate(all_rij))
