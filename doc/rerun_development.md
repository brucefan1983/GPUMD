# GPUMD rerun: replay existing trajectories

**Implementation status:** rerun V1 has completed CUDA, PLUMED, numerical,
selection/scheduling, negative-input, repeated-command, path-collision, normal
MD, and `tests_pytest` validation on an H100. The validation report contains
the tested source commits, commands, status matrix, and numerical results.

The replay loop reads one GPUMD Extended XYZ trajectory and uses its
snapshots as the source of coordinates instead of calling the MD integrator.
Duplicate or out-of-order timesteps are skipped, and reading stops when a
snapshot exceeds `last`. Multiple trajectories are deliberately not supported
in this first version. Existing measurement
actions handle all outputs. No dedicated `rerun_energy.dat` or `output`
keyword is produced/supported.

## PLUMED structural analysis (potential required)

```text
potential nep.txt
time_step 1
plumed plumed.dat 1 0
rerun trajectory.xyz first 0 last 100000 every 1000
```

A valid `potential` command is mandatory for **all** `rerun` invocations,
including analysis-only PLUMED workflows. The potential must be loaded **before**
`rerun`. Every selected snapshot gets a fresh physical force, potential
energy and virial evaluation with `force.compute(box, atom, group)`.
There is no MD integration. An `ensemble` command is optional for ordinary potentials, but required when the potential needs simulation temperature.

This preserves existing `plumed <input> <interval> <restart>` syntax,
initialization and `PLUMED::calculate()`. PLUMED structure-based CVs are
evaluated on trajectory positions, and its COLVAR output remains controlled
by `plumed.dat`. PLUMED does not write its bias forces back into GPUMD during
replay. PLUMED gets its initialization temperature from the existing GPUMD
`ensemble` command, if one was provided. Without an ensemble, PLUMED's kBT
is zero; temperature-dependent PLUMED actions must not be used in that case.

## Temperature-dependent potential replay

```text
potential temperature_dependent_nep.txt
time_step 1
ensemble nvt_ber 950 950 100
plumed plumed.dat 1 0
rerun trajectory.xyz
```

Temperature is specified using the existing GPUMD `ensemble` command,
not a `rerun` keyword. The configured temperature is passed to GPUMD's
existing force-temperature initializer and PLUMED initialization. The
ensemble does **not** execute time integration during rerun; it is used
only to supply physical conditions. For the first version, use a constant
target temperature (no ramp). Temperature-dependent potentials require
an ensemble that supplies a positive temperature.

## Recalculate potential-dependent properties

```text
potential nep.txt
time_step 1
dump_xyz 1000 recalculated.xyz force potential virial
plumed plumed.dat 1 0
rerun trajectory.xyz first 0 last 100000 every 1000
```

All XYZ output attributes are selected with existing `dump_xyz` keywords,
not `rerun output`. `dump_xyz` writes Extended XYZ with the existing GPUMD
format and metadata. Frame output schedules use **original trajectory timestep**.

## Thermodynamic replay

```text
potential nep.txt
time_step 1
dump_thermo 1000
rerun trajectory-with-velocities.xyz
```

`dump_thermo` requires a valid potential **and `vel:R:3` for every selected
frame**. GPUMD's per-atom velocities in XYZ are in Angstrom/fs; the replay
reader converts them to GPUMD internal units. Thermodynamics are calculated
from this frame's actual positions, velocities, and recalculated potential.
Without velocities, the command errors rather than outputting synthetic
temperature or kinetic energy. Other output actions requiring velocities also
error if `vel:R:3` is absent.

When trajectory velocities are absent, XYZ still supports
the potential energy and per-atom virial fields, but temperature and
velocity-dependent stress header fields are reported as NaN rather than using
synthetic velocities.

## Execution semantics

`rerun` is not required to be the last `run.in` command. It restores
positions, velocities, simulation box, image flags, unwrapped positions,
force/potential/virial arrays, and thermo state before the next run/replay.
Both `run` and `rerun` require a loaded potential. An `ensemble` is optional
for ordinary replay but required when a temperature-dependent potential
needs a simulation temperature.

Supported selection keywords:
- `first N`: minimum timestep, default 0
- `last N`: maximum timestep, default unlimited
- `every N`: modulo-step filtering, with the first in-range snapshot eligible
- `skip N`: retain every Nth candidate frame, default 1

Frame metadata: `Time` (fs), `Lattice` (9 values), `pbc` (three flags)
and `Properties` with `species:S:1` and `pos:R:3`. Modern GPUMD `dump_xyz`
now also emits an exact cumulative integer `Step` (not reset by consecutive
`run` commands); during `rerun`, that Step is preserved in the output.
Optional `Step`/`step`/`Timestep` takes precedence over deriving a step
from `Time/time_step`, but when both are present their clocks must agree
for the configured fixed timestep.
Optional `id:I:1` maps 1-based atom indices. Optional `vel:R:3` supplies
per-frame velocity. Without atom IDs, XYZ line order must match `model.xyz`.

## Limitations

- Only existing `plumed`, `dump_xyz` and `dump_thermo` actions are currently
  dispatched during replay; other Measure actions are explicitly rejected.
- Dump of `unwrapped_position` is not yet supported because the reader does
  not restore image flags from the trajectory.
- `dump_xyz` output is checked against the input trajectory before opening.
  Normal output is checked during replay setup; per-frame (`*`) and split
  output use their actual generated filenames. Relative/absolute aliases,
  `..` components, symlinks, and hard links to the input are rejected. Split
  output continues to skip already existing chunk names.
- Missing velocity is never silently replaced with the initial random
  velocity when actual velocity-dependent outputs are requested.
- Time-dependent PLUMED quantities rely on `Step` and the configured
  `time_step`. `Step` and `Time` are checked for consistency; variable
  timestep trajectories are deliberately rejected until an explicit-time
  replay implementation is available.
- The first version supports exactly one Extended XYZ trajectory. Multiple
  input trajectories, LAMMPS dump-format selection, and arbitrary dump formats
  are not supported.
- LAMMPS options `start`, `stop`, `post`, and `dump` are not implemented.
  Replay temperature comes from the existing GPUMD `ensemble` command,
  never from an extra `rerun` argument.
- PLUMED `setStep` and existing GPUMD Measure interfaces use signed integer
  step numbers; large timesteps above INT_MAX are rejected.
- Initialization requires `model.xyz` and a valid potential to define
  the atom types, masses and grouping before replay.
- `potential` is mandatory and must appear before the `rerun` command.
- For temperature-dependent NEP potentials, specify a constant simulation
temperature using GPUMD's existing `ensemble` command. The same temperature
is supplied to PLUMED and Force. No MD integration is performed.
- Temperature ramps and changing target temperatures are not represented
by replay in this first version.

## Standalone reader/selector test

```sh
c++ -std=c++14 -Wall -Wextra -Werror -Isrc/main_gpumd \\
  tests_regression/rerun_reader_test.cpp -o /tmp/rerun_reader_test
/tmp/rerun_reader_test
```

## Input trajectory protection regression

The collision regression driver accepts an executable, trajectory, model, and
potential. It checks normal and per-frame (`*`) collision rejection, relative,
absolute, `..`, symlink, and hard-link aliases, split filename selection, and
successful noncolliding output. Add `--with-plumed` when using a PLUMED-enabled
executable to exercise a structural distance CV and check COLVAR output too:

```sh
python3 tests_regression/rerun_v1/rerun_dump_collision_test.py \
  --gpumd /path/to/gpumd \
  --trajectory /path/to/trajectory.xyz \
  --model tests_regression/fixtures/systems/cu_grouped.xyz \
  --potential potentials/eam/Cu_Zhou_2004.txt \
  --with-plumed
```

The complete V1 validation and the independent-review collision regression
results are recorded in `tests_regression/rerun_v1/REPORT.md`.
