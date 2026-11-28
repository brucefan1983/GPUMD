# NeSK regression tests

Compare two NeSK executables from the GPUMD repository root:

```sh
python3 tests_regression/nesk/run_regression.py --old ./src_old/nesk --new ./src/nesk
```

The tests require Python 3.8+ and a GPU runtime compatible with the supplied
executables. Each case runs both executables in separate directories with
identical inputs, using the selected XYZ fixture as `train.xyz`.

This suite runs independently of the GPUMD/NEP runner in the parent directory.
Its 36 cases are separate from that runner's 255-case `full` suite. See the
[regression overview](../README.md) for the other test collections.

## Cases

The suite contains **32 training cases and 4 fixed-model prediction cases**.
Training covers four datasets (`diamond`, `graphene`,
`silicon`, `sige`), four orbital bases (`sp3`, `sp3sstar`, `sp3d5`, `sp3d5sstar`),
and dispersion weights 0 and 0.2.

Each training case uses two structures, two epochs, seed 17, and no band curriculum.
The learning rate is 0.003 with plateau patience 20. These short runs check
numerical regression; they do not assess convergence or fitting quality.

Prediction covers SiGe with dispersion weight 0.2 and all four orbital bases.
For each basis, both executables receive an identical byte-for-byte copy of
`nesk.txt` from the corresponding successful `--old` training run, then run with
`prediction 1`. Here, fixed means that both predictions use the same baseline
model without updating it. The four models are obtained anew from baseline
training in each suite run; they are not stored reference fixtures.

To select cases and retain their files:

```sh
python3 tests_regression/nesk/run_regression.py --old ./src_old/nesk --new ./src/nesk \
  --dataset graphene --basis sp3sstar --keep
```

Both `--dataset` and `--basis` are repeatable. Selecting `sige` includes one
prediction case for each selected basis; other datasets include training only.
For example, `--dataset sige --basis sp3` runs two training cases and one
prediction case, while the graphene selection above runs two training cases.
`--timeout` sets the time limit per executable invocation, in seconds
(default: 300).

## Checks and tolerances

Training checks completed epochs, model metadata, and scalers. Both modes check
output dimensions, finite values, structure/k-point indices, band column labels,
and reference bands.
The following outputs are compared between the two executables:

| File | Compared content |
| --- | --- |
| `loss.out` | Training: all columns except elapsed time |
| `band_train.out` | Training and prediction: structure/k-point indices, coordinates, predicted and reference bands |
| `nesk.txt` | Training: model metadata, parameters, and descriptor scalers |

Prediction checks the copied model's SHA-256 before and after each run, rejects
`loss.out` or `nesk.restart`, and applies the same band structure, column-label,
reference-value, and numerical checks as training.

Floating-point values must satisfy:

```text
abs(new - old) <= atol + rtol * abs(old)
```

The defaults are `--atol 2e-5 --rtol 2e-4`. Model parameters in
`train_graphene_sp3sstar_dispersion_0.2` use `5 * atol` (default: `1e-4`) to
allow for GPU run-to-run variation. All other fields and cases use the standard
tolerances. Use `--atol 0 --rtol 0` to require exact numerical equality.

Counts, indices, and categorical metadata must agree exactly. Whitespace,
decimal formatting, terminal messages, and elapsed time do not affect the
numerical comparison. A nonzero process exit status or a timeout fails the case.

## Results

The runner prints PASS, FAIL, or SKIP for each case and returns 0 only when all
selected cases pass. A complete successful run ends with:

```text
36 passed, 0 failed, 0 skipped
```

Prediction runs whenever baseline training succeeds and passes its output
checks, even if candidate training or the training comparison fails. If
baseline training fails, its dependent prediction is marked SKIP. Any failure
or skipped prediction makes the suite return 1.

Failed or skipped suites retain their inputs, outputs, `run.log` files, and `summary.json`.
Use `--keep` or `--workdir /absolute/new/directory` to retain successful suites
as well. The report records executable paths and SHA-256 hashes, case results,
effective parameter tolerances, maximum per-file differences for successful
comparisons, and pass/fail/skip counts. Prediction records also identify the
source training case, model path, and model SHA-256. The work directory is
printed at startup.

## Fixture data

Each file in `fixtures/` contains two structures with five and three k-points,
respectively. The fixtures preserve the source cells, atom order, coordinates,
and reference band energies. `band_range` selects the reference columns used
by all four orbital bases.

Source: [NESK datasets, commit 772024b](https://github.com/brucefan1983/NESK/tree/772024b1af92b2e7d10f4cbd549b20c629f34528/datasets).
The source manifests and sampling indices are listed below. Structure and
k-point indices are zero-based.

| Fixture | Source manifest | K-points in structure 1 / structure 7 | `band_range` |
| --- | --- | --- | --- |
| `diamond.xyz` | `Diamond_C8_101/data_diamond/dataset.in` | 17,34,51,67,84 / 26,51,75 | `0 32` |
| `graphene.xyz` | `Graphene_C18_101/gra/dataset.in` | 15,30,45,59,75 / 23,45,68 | `0 72` |
| `silicon.xyz` | `deeptb-Si-10structures/dataset.in` | 60,118,177,235,294 / 88,177,266 | `0 28` |
| `sige.xyz` | `SiGe/sige/dataset.in` | 47,93,140,186,232 / 69,140,210 | `0 28` |
