# gpumd_to_kaldo — NEP force-constant exporter for kaldo

Export second- and third-order interatomic force constants (IFCs) derived from
a GPUMD NEP potential into kaldo's `format='gpumd'` on-disk layout
(`gpumd_fc.npz`).  The pipeline runs entirely on the CPU and requires neither
a GPU nor a running GPUMD simulation: calorine's `CPUNEP` evaluates forces on
phono3py-generated displaced supercells, and the resulting IFCs are remapped
into kaldo's canonical C-grid replica order, optionally corrected with the
acoustic sum rule, and written to a single compressed numpy archive.

---

## Dependencies

**Required (core pipeline):**

| Package | Purpose |
|---|---|
| `ase` | Atoms object, structure I/O |
| `calorine` | CPUNEP ASE calculator |
| `phonopy` | Harmonic (fc2) finite displacements |
| `phono3py` | Anharmonic (fc3) finite displacements |
| `numpy` | Array operations |
| `scipy` | Lattice-constant relaxation (Brent) |
| `sparse` | COO storage for fc3 |

**Required only for the cross-check tests** (`test_fc2_layout_matches_hiphive_oracle`,
`test_fc3_matches_hiphive_oracle`, and the kaldo round-trip / end-to-end tests):

| Package | Purpose |
|---|---|
| `kaldo` | kaldo `format='gpumd'` reader; grid utilities |
| `hiphive` | Independent fc2/fc3 oracle for element-wise comparison |

The pure-NEP tests (B.1/B.2 — structure helpers, force evaluation, phonopy fc2,
phono3py fc3) run without kaldo or hiphive installed.

---

## `gpumd_fc.npz` format

A single `np.savez_compressed` archive.  All arrays are numpy-native; no
additional libraries are needed to read it beyond numpy and sparse.

| key | dtype | shape | meaning |
|---|---|---|---|
| `format_version` | int | scalar | starts at `1` |
| `atomic_numbers` | int | `(n_uc,)` | unit-cell species |
| `positions` | float64 | `(n_uc, 3)` | unit-cell Cartesian positions (Å) |
| `cell` | float64 | `(3, 3)` | unit-cell vectors (Å) |
| `supercell` | int | `(3,)` | diagonal supercell for `fc2` |
| `third_supercell` | int | `(3,)` | diagonal supercell for `fc3` |
| `fc2` | float64 | `(1, n_uc, 3, n_rep2, n_uc, 3)` | second-order IFC, eV/Å², kaldo C-grid replica order, reference cell at axis 0 |
| `fc3_coords` | int32 | `(3, nnz)` | flattened COO coords for the `(n_uc·3, n_rep3·n_uc·3, n_rep3·n_uc·3)` array |
| `fc3_data` | float64 | `(nnz,)` | third-order IFC values, eV/Å³ |
| `fc3_shape` | int | `(3,)` | `(n_uc·3, n_rep3·n_uc·3, n_rep3·n_uc·3)` |
| `units_fc2` / `units_fc3` | str | scalar | `'eV/angstrom^2'` / `'eV/angstrom^3'` |
| `grid_order` | str | scalar | `'C'` |
| `acoustic_sum_applied` | bool | scalar | whether ASR was applied by the writer |
| `nep_potential` / `generator` | str | scalar | provenance: NEP filename + sha256; tool name + version |

**Index convention:** replica index `r ∈ [0, n_rep)` corresponds to cell-image
grid index `np.unravel_index(r, supercell, order='C')`, i.e. exactly
`kaldo.grid.Grid(supercell, 'C').grid(is_wrapping=False)[r]`.
`fc2[0, i, α, r, j, β]` = Φ²(ref-cell atom *i*, direction α; atom *j* in
image *r*, direction β), eV/Å² (bare IFCs, not mass-weighted).

---

## CLI usage

### Default: built-in silicon example

```bash
python gpumd_to_kaldo.py \
    --nep /path/to/nep.txt \
    --supercell 3 3 3 \
    --acoustic-sum \
    --out gpumd_fc.npz
```

The script relaxes the Si lattice constant automatically (`--a` overrides it).

### Any ASE-readable structure

Pass any POSCAR, extxyz, CIF, or other ASE-readable file with `--structure`:

```bash
python gpumd_to_kaldo.py \
    --nep /path/to/nep.txt \
    --structure my_unitcell.vasp \
    --supercell 3 3 3 \
    --acoustic-sum \
    --out gpumd_fc.npz
```

The structure is used **as provided** — no relaxation is performed.  Ensure
the cell is at (or close to) its equilibrium geometry for the given NEP so
that the force-constant expansion is well-conditioned.  The `--a` flag is
ignored when `--structure` is given.

### All options

```
--nep PATH            Path to the NEP potential file (required).
--out PATH            Output .npz path (default: gpumd_fc.npz).
--supercell NX NY NZ  Supercell for fc2 and fc3 (default: 3 3 3).
--third-supercell NX NY NZ
                      Separate supercell for fc3 (default: same as --supercell).
--structure PATH      Any ASE-readable unit cell; used as-is.
--a FLOAT             Si lattice constant in Å (default: auto-relaxed).
                      Ignored when --structure is given.
--threshold FLOAT     Drop fc3 entries with |value| <= threshold (default: 0).
--acoustic-sum        Apply the acoustic sum rule to fc2 before writing.
```

---

## Silicon example

```bash
cd example_silicon
bash run.sh
```

`run.sh` runs a (3, 3, 3) supercell on the in-repo Si NEP and prints the kaldo
load command.  The output `gpumd_fc.npz` lands in `example_silicon/`.

---

## How kaldo reads it

```python
from kaldo.forceconstants import ForceConstants

fc = ForceConstants.from_folder('.', format='gpumd')
# fc.second  — SecondOrder in kaldo C-grid layout
# fc.third   — ThirdOrder sparse COO
```

kaldo rebuilds the replicated supercell from the stored `supercell` vector (no
`replicated_atoms.xyz` needed) and loads fc3 directly into a `sparse.COO`
without any index remapping.

---

## Running the tests

```bash
cd tools/Format_Conversion/gpumd_to_kaldo
conda run -n gpumd-kaldo python -m pytest test_gpumd_to_kaldo.py -v
```

Tests that require kaldo or hiphive are automatically skipped if those packages
are not installed, so GPUMD's own CI (which need not install kaldo) can still
run the pure-NEP subset.
