# PPPM spacing regression tests

Pinned old baseline: `87c1cf22401ac8c791adda0879e0af704dd5f981`.
This is a transition manifest: run the old executable against the patched
candidate, with identical compiler, architecture and feature flags. Both must
use `-DDEBUG`; `gpumd_replica` is outside the change.

The package has 253 cases and 25 relations. The `pppm` focused suite selects
37 cases (32 new and 5 existing). All new cases also belong to `full`.

## Declared differences

- Explicit spacing is rejected by the old baseline and accepted by the
  candidate. Invalid input has separate, exact old/new diagnostics.
- `pppm_round_exact`, `pppm_round_above`, `pppm_round_5_7`,
  `pppm_triclinic_mesh`, both `pppm_grow_shrink` cases, and the existing
  `qnep_pppm_future_bec` intentionally use different old/new meshes. Their
  outputs are validated separately, not compared with a relaxed tolerance.
  The existing BEC case changes from 32^3 to 28^3; its former atomic-noise
  tolerances must not be reused as mesh-approximation tolerances.
- All other pre-existing cases keep their original numerical comparison.
  The initial mesh line in `dftd3_kspace_one_each` is validated separately.
- Every case still runs both executables. No `candidate_only` cases were added.
  A no-force construction/destruction case may declare no generated files only
  with an explicit required stdout completion diagnostic.

## Candidate mesh checks

The per-case `pppm` declaration checks exactly one initial mesh message, its
expected dimensions, even FFT-friendly sizes, target spacing, actual spacing,
and (where declared) XYZ frame count and the final prescribed cell. Additional
PPPM mesh messages fail the check. Only the recognized initial mesh message
is removed from a case's stdout comparison after this check succeeds.
Other stdout/stderr comparisons are unchanged.

The two candidate-only `numeric_equal` relations check omitted `kspace`,
`kspace pppm`, `kspace pppm 1.0`, and a late `kspace pppm 1.0`. The comparator
preserves field order and nonnumeric text and rejects nonfinite values. The
new same-mesh comparisons use absolute bounds of 1e-7 for thermo output and
2e-5 for XYZ numerical fields, with zero relative tolerance. These are
float-atomic noise bounds, not a PPPM accuracy claim. They passed the reported
target-GPU validation; other platforms must still be checked. Inspect failures
before changing a bound.

## Box changes and validation scope

The grow/shrink cases expand the x thickness from 15.9 to 18.6 Angstrom,
then compress it to 15.9 Angstrom in a second run. The intended mesh sequence is:

```
16 x 16 x 16 -> 18 x 16 x 16 -> 20 x 16 x 16
```

The x mesh stays at 20 during compression. Both with- and without-per-atom-virial
paths are exercised. Same-mesh deformation, triclinic shear, short NPT, and
two-run reuse cases are retained. The final XYZ cell independently confirms
that a prescribed deformation ran, and same-mesh cases retain the old/new
numerical comparisons.

The initial implementation was validated on the target GPU using temporary
per-evaluation logs. Those checks required nondecreasing mesh dimensions,
the spacing bound, and a rebuild exactly when dimensions changed. The logs
and their test option were removed after validation. The retained automated
tests exercise these paths but do not directly observe intermediate mesh
dimensions or FFT plan rebuilds. Future changes to those internals require
targeted validation again.

Only the initial mesh is printed, including in a `-DDEBUG` build. No runtime
PPPM/Ewald accuracy comparison or automatic accuracy tuning exists.

## Running the tests

1. Run manifest validation, `test_manifest.py`, and unittest discovery.
2. Run `--suite pppm --keep-all` on the GPU.
3. Run `--suite full --keep-all`, supplying both DEBUG NEP
   executables as well as both GPUMD executables. Expected summary:
   `253 passed, 0 failed; relations: 25 passed, 0 failed, 0 skipped`.

The focused suite omits unrelated relation members, so its report normally
contains skipped relations. Only the full suite requires zero skips.
Each new invocation deletes the previous `.work` tree; archive a failed run
before starting another invocation.

When promoting this version to the accepted baseline, update the role-specific
transition expectations and restore same-version differential checks for the
formerly changed-mesh cases.
The cleanup does not promote the baseline or change any numerical tolerance.

The FFT radix policy builds on the idea in GPUMD PR #1800 / issue #1796.
