# GPUMD trajectory rerun (experimental)

This feature is under development on `feature/rerun`. It reads existing GPUMD
Extended XYZ trajectories, evaluates optional potential energies and does not
perform time integration. **CUDA build and NEP numerical verification are still
outstanding. PLUMED is not supported yet.**

## Example: NEP evaluation

```text
potential nep.txt
time_step 1

rerun trajectory.xyz first 1000 last 100000 every 1000 skip 1 output energy_1.dat
rerun trajectory2.xyz output energy_2.dat
```

`rerun` is **not** required to be the last command: it restores the original
positions, simulation cell, image flags and available force/energy/virial buffers
on completion. It does not finalize the active force object. Subsequent
`run` or `rerun` commands are therefore allowed, subject to their individual
command requirements. Normal MD `run` still requires `ensemble` and `potential`.

## Example: trajectory-only pass (no potential)

```text
time_step 1
rerun trajectory.xyz output frames.dat
```

Without a potential, `rerun` emits only `timestep time_fs` metadata, not
energy. This is a foundation for future structure-only analysis; **it does not
yet run PLUMED or GPUMD measurement actions**. `model.xyz` is still required
to define the atom count, species, masses, and grouping. With no potential, its
species indices are inferred in first-occurrence order from `model.xyz`.

## Keywords

- `first N`: minimum timestep (default 0)
- `last N`: maximum timestep (default unlimited)
- `every N`: timesteps divisible by N, except the first in-range frame is
  eligible regardless of divisibility (default 0)
- `skip N`: retain every Nth frame after timestep filtering (default 1)
- `output FILE`: GPUMD-specific output path; default `rerun_energy.dat` if
  a potential is loaded and `rerun_frames.dat` otherwise

`first/last/every/skip` mirror the basic LAMMPS selection semantics.
Unlike LAMMPS this GPUMD implementation does not currently support
multi-file arguments, `start/stop/post`, `dump`, or LAMMPS dump parsing.
Output file names may differ from the input trajectory. Files are
overwritten at each rerun invocation, so specify different names to keep
outputs from successive reruns.

## Frame metadata and requirements

The reader requires `Time` (fs), `Lattice` (9 components), `pbc` (3 flags)
and `Properties` containing `species:S:1` and `pos:R:3`.
Optional `Step`, `step`, or `Timestep` integer metadata overrides
`Time/time_step` reconstruction. When absent, an integral step is reconstructed
from `Time` and the `time_step` value in `run.in`. For variable-timestep
trajectories, include an explicit integer step in the XYZ header.

Frames must have a fixed number of atoms, valid cells and strictly
increasing steps. When an `id:I:1` property is present, atom rows are mapped
to the GPUMD index using the **1-based IDs 1..N**. Duplicate, invalid and
out-of-range IDs are rejected. Without an ID property, atom rows must remain
in the exact same atom-index order as `model.xyz`; exchanging atoms of the same
species cannot be detected.

## Limitations and validation

- The existing `Measure` actions (including PLUMED) are rejected until an
  analysis-only lifecycle is implemented. Removing that guard alone would
  be unsafe.
- The potential evaluation path is written but is not yet validated by CUDA
  compilation, direct GPU NEP reference calculations, or numerical regression.
- A standalone CPU-only parser/selector test is in
  `tests_regression/rerun_reader_test.cpp`:

```sh
c++ -std=c++14 -Wall -Wextra -Werror -Isrc/main_gpumd \\
  tests_regression/rerun_reader_test.cpp -o /tmp/rerun_reader_test
/tmp/rerun_reader_test
```

Before production use, run GPU build and compare per-frame energy, forces and
virials with independent NEP single-point evaluations, then test consecutive
rerun and run commands for exact state restoration.
