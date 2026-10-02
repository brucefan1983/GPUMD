# PPPM spacing V1 validation

Pinned old baseline: `87c1cf22401ac8c791adda0879e0af704dd5f981`.
This is a transition manifest: run the old executable against the patched
candidate, with identical compiler, architecture and feature flags. Both must
use `-DDEBUG`; `gpumd_replica` is outside the change.

The package has 253 cases and 25 relations. The `pppm` diagnostic suite selects
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
and (where declared) XYZ frame count and the final prescribed cell. Only
recognized mesh messages are removed from a case's stdout comparison after
this check succeeds. Other stdout/stderr comparisons are unchanged.

The two candidate-only `numeric_equal` relations check omitted `kspace`,
`kspace pppm`, `kspace pppm 1.0`, and a late `kspace pppm 1.0`. The comparator
preserves field order and nonnumeric text and rejects nonfinite values. The
new same-mesh comparisons use absolute bounds of 1e-7 for thermo output and
2e-5 for XYZ numerical fields, with zero relative tolerance. These are
provisional float-atomic noise bounds to confirm on the target GPU, not a
PPPM accuracy claim. Inspect failures before changing a bound.

## Temporary dynamic trace

For V1, compile with `-DGPUMD_PPPM_DIAGNOSTICS` in addition to `-DDEBUG`.
The separate macro emits mesh dimensions, current thicknesses and a rebuild
flag at each PPPM force evaluation. It has no effect on the old baseline.
Run the runner with `--pppm-diagnostics` to require the trace for the six
mesh-lifecycle cases. Merely adding that runner option without the compiled
macro must fail, rather than silently omit the dynamic check.

For the grow/shrink cases the compressed mesh sequence must be:

```
16 x 16 x 16 -> 18 x 16 x 16 -> 20 x 16 x 16
```

The second run compresses the x thickness from 18.6 back to 15.9 Angstrom,
while the mesh stays at 20. Both with- and without-per-atom-virial paths are
covered. Every trace record checks monotonic dimensions, the spacing bound,
and that the rebuild flag occurs exactly when dimensions change. Same-mesh
deformation, triclinic shear, short NPT, and two-run reuse are also checked.
The final XYZ cell independently confirms that the intended deformation ran.

A normal build does not print intermediate mesh information. A run without
`--pppm-diagnostics` still validates the initial mesh and result files; its
report explicitly records `dynamic_trace_checked: false` for lifecycle cases
without a trace. It is not a substitute for V1's diagnostic acceptance run.
No runtime PPPM/Ewald accuracy comparison or automatic accuracy tuning exists.

## Acceptance and cleanup

1. Run manifest validation, `test_manifest.py`, and unittest discovery.
2. Run `--suite pppm --pppm-diagnostics --keep-all` on the GPU.
3. Run `--suite full --pppm-diagnostics --keep-all`, supplying both DEBUG NEP
   executables as well as both GPUMD executables. Expected summary:
   `253 passed, 0 failed; relations: 25 passed, 0 failed, 0 skipped`.
4. Build the candidate without the diagnostic macro and run the PPPM suite
   without `--pppm-diagnostics` to verify the production output behavior.

The focused suite omits unrelated relation members, so its report normally
contains skipped relations. Only the full suite requires zero skips.
Each new invocation deletes the previous `.work` tree; archive a failed run
before starting another invocation.

After GPU validation, remove the temporary `#ifdef` block, diagnostic parser,
runner option, and diagnostic-specific self-tests together. Retain initial
mesh checks and all numerical/lifecycle cases. When promoting this version to
the accepted baseline, update the role-specific transition expectations and
restore same-version differential checks for the formerly changed-mesh cases.
The temporary diagnostic cleanup and baseline promotion must be reviewed
explicitly; do not silently remove checks or relax output tolerances.

The FFT radix policy builds on the idea in GPUMD PR #1800 / issue #1796.
