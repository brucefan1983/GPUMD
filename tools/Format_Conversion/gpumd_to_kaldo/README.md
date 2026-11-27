# NEP force constants for kaldo

[gpumd_to_kaldo.py](gpumd_to_kaldo.py) evaluates NEP forces with calorine's `CPUNEP`, computes second- and third-order force constants with phono3py, and exports `gpumd_fc.npz` in kaldo's C-grid replica order. The calculation runs on the CPU. The optional acoustic sum rule correction applies to the second-order force constants.

## Dependencies

Python packages: `ase`, `calorine`, `phonopy`, `phono3py`, `numpy`, `scipy`, `sparse`, and `kaldo`. Normal export uses `kaldo.grid`; reading the output requires a kaldo version with the `format='gpumd'` reader. Use a NEP model supported by the installed calorine version.

## Usage

Run from this directory. For the built-in two-atom silicon primitive cell:

```bash
python3 gpumd_to_kaldo.py \
    --nep /path/to/nep.txt \
    --supercell 3 3 3 \
    --acoustic-sum \
    --out gpumd_fc.npz
```

The Si lattice constant is relaxed automatically, or set with `--a` in angstrom. To use a prepared primitive cell, add `--structure my_unitcell.vasp` (any ASE-readable format). This structure is used as provided, without relaxation, and `--a` is ignored.

`--supercell` defaults to `3 3 3`. `--threshold` drops third-order entries with absolute values at or below the specified value (default `0`, in eV/Å³). `--out` defaults to `gpumd_fc.npz` and overwrites an existing output file. See `python3 gpumd_to_kaldo.py --help` for the command-line options.

The included Si example uses the repository's Si NEP model:

```bash
cd example_silicon
bash run.sh
```

It writes `example_silicon/gpumd_fc.npz`.

## Output format

The compressed NumPy archive stores dense second-order force constants and sparse third-order force constants as separate coordinate, value, and shape arrays. NumPy can read all stored arrays; `sparse.COO` reconstructs the third-order tensor.

Here `n_uc` is the number of unit-cell atoms, and `n_rep2` and `n_rep3` are the products of the stored supercell dimensions.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `format_version` | int64 scalar | `1` |
| `atomic_numbers` | int64 `(n_uc,)` | Unit-cell species. |
| `positions` | float64 `(n_uc, 3)` | Cartesian positions, Å. |
| `cell` | float64 `(3, 3)` | Unit-cell vectors, Å. |
| `supercell` / `third_supercell` | int64 `(3,)` | Stored second- and third-order supercell dimensions. |
| `fc2` | float64 `(1, n_uc, 3, n_rep2, n_uc, 3)` | Second-order force constants, eV/Å². |
| `fc3_coords` | int32 `(3, nnz)` | Flattened COO coordinates. |
| `fc3_data` | float64 `(nnz,)` | Third-order force constants, eV/Å³. |
| `fc3_shape` | int64 `(3,)` | `(n_uc*3, n_rep3*n_uc*3, n_rep3*n_uc*3)`. |
| `units_fc2` / `units_fc3` | String scalars | `eV/angstrom^2` / `eV/angstrom^3`. |
| `grid_order` | String scalar | `C`. |
| `acoustic_sum_applied` | Boolean scalar | Whether the second-order sum rule was applied. |
| `nep_potential` / `generator` | String scalars | NEP path and SHA-256; tool name and version. |

Replica `r` follows `np.unravel_index(r, supercell, order='C')`, matching `kaldo.grid.Grid(supercell, 'C').grid(is_wrapping=False)[r]`. The entry `fc2[0, i, alpha, r, j, beta]` couples reference-cell atom `i` to atom `j` in replica `r`. These force constants are not mass-weighted.

Load the archive from its output directory:

```python
from kaldo.forceconstants import ForceConstants

fc = ForceConstants.from_folder('.', format='gpumd')
```

## Tests

With the runtime dependencies and `pytest` installed, run from this directory:

```bash
python3 -m pytest test_gpumd_to_kaldo.py -v
```

The tests use the repository's Si model by default; `SI_NEP` can select another path to that model. The independent comparison tests additionally require `hiphive` and skip when it is unavailable.
