# GPUMD trajectory rerun (initial implementation)

This development branch contains an **experimental**, energy-only rerun path.
It is not yet a replacement for LAMMPS `rerun` or PLUMED `driver`.

## Input

Place `model.xyz`, the NEP potential, `run.in`, and a previously written GPUMD
Extended XYZ trajectory in the working directory:

```text
potential nep.txt
time_step 1
rerun trajectory.xyz first 1000 last 100000 every 1000 skip 1
```

The command must be the last one in `run.in`. A potential is currently required.
No `ensemble`, `plumed`, `dump_xyz` or other analysis action is accepted in
this first stage. The command **does not integrate positions, velocities, or the
box**. The current output is `rerun_energy.dat`, with three columns: original
timestep, time in fs, and total potential energy in eV.

## Current keyword meanings

- `first N`: minimum original MD timestep (default 0)
- `last N`: maximum original MD timestep (default unlimited)
- `every N`: select steps divisible by N (default 0, all)
- `skip N`: retain every Nth frame from those remaining after timestep filters (default 1)

As with GPUMD's current `dump_xyz`, timestep is reconstructed from the
`Time` metadata in femtoseconds divided by `time_step`. Non-integral times
are rejected. Trajectory steps must be strictly increasing. This initial
version is limited to a single file with a fixed number and order of atoms,
matching `model.xyz`. It reads `Properties` to find species and Cartesian
positions, and handles variable triclinic `Lattice` and PBC per frame.

**Important:** do not use an input trajectory produced with variable MD
timesteps unless its times can be mapped unambiguously to integral steps.

## Remaining work before general use

- Integrate PLUMED with a read-only/analysis-only evaluation path, preserving
  real trajectory steps and excluding biasing effects.
- Support multiple trajectory files and selected GPUMD analysis actions.
- Add GPU-backed integration tests and compare energies and forces against
  direct per-frame NEP evaluation.
- Verify builds in CUDA and HIP CI. No compilation or runtime verification
  is claimed for this first commit.

## Parser regression test

The standalone C++ test does not need CUDA:

```sh
c++ -std=c++14 -Isrc/main_gpumd tests_regression/rerun_reader_test.cpp -o /tmp/rerun_reader_test
/tmp/rerun_reader_test
```
