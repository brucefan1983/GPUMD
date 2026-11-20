# PPPM spacing regression tests

Both executables use the same PPPM input, mesh and comparison contracts.
Use identical compiler, architecture and feature flags; both executables must
use `-DDEBUG`.

The `pppm` focused suite is a subset of `full`.

## Shared checks

- Both executables must accept valid explicit spacing and reject invalid input
  with the current diagnostic and exit status.
- Every case compares the generated files and normalized stdout/stderr between
  baseline and candidate. No case disables cross-version comparison.
- The per-case `pppm` declaration checks the initial mesh dimensions, even
  FFT-friendly sizes, target/actual spacing and, where declared, frame count
  and final cell independently for each executable.
- Exactly one initial mesh message is required when forces are evaluated.
  No-force construction/destruction cases must not create a mesh.
- Mesh lines remain in both stdout streams. The runner stores the independent
  results under `metrics.pppm.baseline` and `metrics.pppm.candidate`.
- Both PPPM default-equivalence relations run for both executables. They compare
  omitted `kspace`, `kspace pppm`, `kspace pppm 1.0`, and late `kspace pppm 1.0`.

## Numerical comparisons

The spacing, rounding, triclinic and grow/shrink comparisons use same-mesh
PPPM absolute bounds: `1e-7` for `thermo.out` and
`2e-5` for `state.xyz`, with zero relative tolerance. All numerical fields and
nonnumeric text are compared; nonfinite values are rejected. `neighbor.out`
remains byte-exact.

`qnep_pppm_future_bec` enables BEC and per-atom virial output in the second run,
checking initialization for consumers declared after the first run. Its
same-mesh bounds are `1e-5` for `bec.xyz`, `1e-7` for `thermo.out`, and `2e-6`
for `dpdt.out`. Calibration measured BEC
drift up to `6.07e-6`, thermodynamic drift up to `4.01e-8`, and DPDT drift up to
`1e-6`. These bounds account for unordered float atomics; they are not PPPM
accuracy tolerances or permission to compare different mesh algorithms.

When changing the build environment, validate these bounds with the same
executable on both sides on the target GPU. Inspect any numerical delta before
changing a bound.

## Box changes and validation scope

The grow/shrink cases expand the x thickness from 15.9 to 18.6 Angstrom,
then compress it to 15.9 Angstrom in a second run. The intended mesh sequence is
`16 x 16 x 16 -> 18 x 16 x 16 -> 20 x 16 x 16`; the x mesh stays at 20 during
compression. Both with- and without-per-atom-virial paths are exercised.
Same-mesh deformation, triclinic shear, short NPT, and two-run reuse cases are
retained. The final XYZ cell is checked independently, and the generated
outputs are compared between baseline and candidate.

Only the initial mesh is printed. These tests do not directly observe
intermediate mesh dimensions or FFT-plan rebuilds, and do not establish
PPPM/Ewald accuracy equivalence.

## Running the tests

1. Run manifest validation, `test_manifest.py`, and unittest discovery.
2. Check the setup by passing the same executable for both roles, as shown in
   [the regression overview](../README.md). Keep failing artifacts for numerical inspection.
3. Run `--suite full --keep-all` with the accepted and candidate GPUMD/NEP pairs.

The focused suite omits unrelated relation members, so skipped relations are
expected for a focused run. For a full run, all cases and relations must pass,
with no skipped relations.

The FFT radix policy builds on the idea in GPUMD PR #1800 / issue #1796.
