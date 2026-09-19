# GPUMD regression tests

This directory contains the long-term differential regression tests for the default
`gpumd` executable. It compares a previously accepted executable with a candidate
executable and is intended to be run after every source change.

The runner does not build GPUMD, infer revisions, or select a baseline. Supply
the repository root and both executables explicitly:

```bash
python3 tests/gpumd/regression/run_regression.py \
  --repo-root /path/to/GPUMD \
  --baseline /path/to/gpumd_baseline \
  --candidate /path/to/gpumd_candidate \
  --suite full
```

Paths prefixed with `repo:` in `manifest.json` are resolved below
`--repo-root`; paths prefixed with `package:` are resolved below this package.
This permits reuse of public GPUMD examples and potentials while keeping
regression-specific inputs self-contained.

## Acceptance contract

`full` is the only acceptance suite. It contains all 195 cases and evaluates
all 23 cross-case relations after the cases pass. Every case and relation runs
for both the baseline and candidate. A successful run ends with:

```text
Summary: 195 passed, 0 failed; relations: 23 passed, 0 failed, 0 skipped
```

Focused suites are diagnostic subsets only. A focused result is not a
substitute for a successful `full` run. The intended sequence after every
source change is:

1. validate the manifest;
2. build the baseline and candidate consistently;
3. run `--suite full`.

## Requirements and reproducible builds

- Python 3.9 or newer, the standard library, and NumPy;
- Linux with one available CUDA or HIP device;
- baseline and candidate built from the intended sources with identical
  compiler, backend, architecture, feature, and optimization options.

Use the last accepted executable as the baseline and the newly built
executable as the candidate. Clean the build when compiler options or optional
features change so that stale objects cannot mix configurations.

For strict comparison of stochastic paths, build both executables with the
same deterministic `-DDEBUG` convention. The package is designed for
deterministic single-GPU execution. By default the runner exposes the first
device already selected by `CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES`, or
device `0` when neither variable is set. Use `--device` to select another
single device. The runner also sets `OMP_NUM_THREADS=1` for each GPUMD process.

Reports record executable paths and SHA-256 digests, but the runner cannot
prove that two binaries used identical build options. That remains a build
requirement.

## Validate and inspect the package

Manifest validation does not start GPUMD or require executable arguments. It
does require the repository root so that every `repo:` source can be checked:

```bash
python3 tests/gpumd/regression/run_regression.py \
  --repo-root /path/to/GPUMD \
  --check-manifest
```

The package self-tests verify the manifest and runner contracts:

```bash
GPUMD_REPO_ROOT=/path/to/GPUMD \
  python3 tests/gpumd/regression/test_manifest.py

python3 -m unittest discover \
  -s tests/gpumd/regression -p 'test_*.py'
```

Use `--list` and optional shell-style case patterns to inspect a selection
without executing GPUMD:

```bash
python3 tests/gpumd/regression/run_regression.py \
  --repo-root /path/to/GPUMD \
  --suite full --case 'pimd_*' --list
```

## Suites

The manifest defines 21 suites. `full` is the acceptance gate; the other
suites exist only to isolate a failure:

| Suite | Diagnostic scope |
|---|---|
| `quick` | small representative subset |
| `standard` | conventional NVE, NVT, and NPT ensembles |
| `mttk` | MTTK and pressure-coupling variants |
| `adaptive_time_step` | displacement-limited adaptive stepping |
| `heat` | heat transport and two-temperature workflows |
| `pimd` | PIMD, RPMD, TRPMD, bead state, and bead output |
| `special` | MSST, NPHug, and moving-wall ensembles |
| `ti` | thermodynamic-integration variants |
| `lifecycle` | multiple runs and setting propagation or cleanup |
| `lifecycle_contract` | required ensemble/run lifecycle rules |
| `parser` | lexical parsing and top-level token handling |
| `parsing_validation` | option grammar, bounds, and interval validation |
| `invalid` | rejected input and public diagnostics |
| `actions` | Action parsing, ordering, state, and output |
| `potential` | potential setup, transitions, and consumers |
| `snapshot` | supported commands that inspect later input settings |
| `state` | cross-run state propagation and reset |
| `first_step_init` | initialization exercised by the first integration step |
| `static` | non-dynamics calculations |
| `transport` | transport measurements |
| `full` | all 195 cases and all 23 relations |

For example, a focused rerun may help diagnose a full-suite failure:

```bash
python3 tests/gpumd/regression/run_regression.py \
  --repo-root /path/to/GPUMD \
  --baseline /path/to/gpumd_baseline \
  --candidate /path/to/gpumd_candidate \
  --suite lifecycle --case '*two_runs*' \
  --keep-all
```

Relations whose members are outside a focused selection are reported as
skipped. This is expected for diagnostics; the acceptance run must report zero
skipped relations. See `CONTRACT_SUITES.md` for the durable relation oracles.

## Execution and comparison contract

Every selected case starts the baseline once and the candidate once in
separate fresh working directories. The runner does not repeatedly execute a
binary to establish self-repeatability. There are no candidate-only or
role-specific cases: both executables must meet the same declared contract.

Comparison is deliberately strict:

- the generated-file inventory must exactly match the manifest;
- ordinary generated files are compared byte for byte;
- `stdout` is compared after removing only declared timing and speed noise;
- `stderr` is compared after removing only internal source-file and source-line
  locations;
- missing, unexpected, empty, or unauthorized modified files fail the case.

A non-byte comparison must be declared for the individual output in the
case's `comparisons` mapping and include a narrow reason. There is no global
floating-point tolerance and no output sorting. Text mismatches produce a
unified diff; binary mismatches identify the first differing byte. UTF-8 text
outputs fail if they contain NaN, infinity, or an overflowing numeric token,
including under exact-byte mode.

The manifest contains narrowly scoped zero-relative-tolerance numerical
comparisons for qNEP outputs affected by unordered single-precision GPU
accumulation. The comparator still requires identical token counts, token
order, and nonnumeric text, and rejects non-finite or overflowing values. All
other outputs remain byte-exact unless their manifest entry explicitly says
otherwise.

Selected cases also declare NumPy-based semantic post-checks. These checks are
additional oracles applied independently to the baseline and candidate after
the direct outputs have satisfied the normal byte-exact comparison. Tolerances
are permitted only inside these post-processing checks, currently for quantities
that can be recomputed independently from generated data: active-learning
uncertainty and MSD. They do not relax baseline/candidate output comparison.

Inputs and staged fixtures are hashed before and after execution. A case that
intentionally rewrites an input must declare it in `mutable_inputs`.
Unexpected mutation or deletion fails the case. Declared mutable inputs are
also compared byte for byte between baseline and candidate after execution.

### Cross-case relations

The top-level `relations` list provides exact oracles involving multiple
cases:

- `equal` requires all referenced case/output members to be byte-identical;
- `concat` requires a result output to equal the ordered bytewise
  concatenation of its parts.

A relation runs only after all referenced cases pass. `full` includes every
relation member, so all 23 relations run for both executables. Passing work
directories are retained until applicable relations complete. A relation
failure retains all implicated case directories and exits unsuccessfully.

Captured `stdout` and `stderr` are stored in runner-private directories outside
the GPUMD working directory. Therefore an executable-created file literally
named `stdout.txt` or `stderr.txt` remains visible as an undeclared output.

## Input styles

Every manifest case declares one input style:

- `canonical` follows the recommended layout: comments and blank lines,
  optional `replicate` as the first effective command, a contiguous potential
  block, optional global initialization commands, and run blocks beginning
  with `ensemble` and ending with `run`;
- `compatibility` exercises a supported noncanonical order, such as `fix`,
  `move`, `deform`, or an Action before `ensemble`;
- `intentional_invalid` violates a declared input contract and must fail with
  the expected public diagnostic.

The style annotation validates test data; it does not alter parsing. An
`ensemble` is the recommended beginning of a run block but is not a collection
boundary. Pending non-immediate settings are consumed by the next `run`, and
compatibility cases preserve that supported behavior.

For canonical and compatibility inputs, `replicate`, when present, is the
first effective command; comments and blank lines may precede it. `dftd3` and
`kspace` may each occur at most once. Canonical inputs keep a contiguous
potential block before the first run. Compatibility inputs may append another
potential between completed runs. Velocity initialization remains before the
first run.

## Reports and retained work

Machine-readable reports are written below:

```text
tests/gpumd/regression/reports/
```

Each report records the selected cases, results, durations, executable
metadata, and staged/generated file hashes. Failed work directories are
retained below:

```text
tests/gpumd/regression/.work/<run-id>/
```

Passing directories are normally removed after their hashes and applicable
relations have been recorded. Add `--keep-all` to retain all baseline and
candidate directories for inspection. Inspect the report and retained files
before changing an expected output, tolerance, or normalization rule.

## Coverage and known gaps

The package covers every accepted `ensemble` keyword and representative NVE,
NVT, NPT, MTTK, adaptive-step, heat-transport, path-integral,
thermodynamic-integration, and moving-wall workflows. It also covers parser
validation, multiple-run lifecycle, immediate commands, Action ordering,
potential selection and transitions, output append/overwrite behavior,
deposition, minimization, phonons, and deterministic neighbor-search edge
cases. Transport and structural measurements include DOS, SDC, MSD, RDF, ADF,
HAC, HNEMD, SHC, and viscosity, including coexistence of multiple measurement
modules. Both fully periodic and common two-dimensional `T T F` boundary
conditions are exercised.

Potential coverage includes EAM, Tersoff, NEP89, multiple NEPs,
temperature-dependent NEP, qNEP Ewald/PPPM with charge and BEC consumers, and
three hybrid ILP paths. Additional cases exercise RDF/angular RDF, active learning (including interval,
threshold, output-field, and observer-independence contracts), observer
observe/average modes, dipole and polarizability response dumps, fixed/moving
atoms, and electric-field consumers. NumPy post-checks independently recompute
active uncertainty and MSD. Dipole, polarizability, observer, and liquid-TI
outputs are checked by the normal byte-exact baseline/candidate regression and
by exact cross-case side-effect relations where applicable; no historical
hardcoded response or free-energy values are used.
Coverage is recorded through generic `covers` tags rather than inferred from
case names.

This is a regression gate, not an exhaustive validation of every GPUMD build
configuration or potential family. Optional NetCDF and MDI paths are not
covered by executable-level cases. Other optional or normally disabled
potential implementations require separate coverage before making claims
about them.
