# TB regression tests

Compare two TB executables from the GPUMD repository root:

```sh
python3 tests_tb/run_regression.py --old /path/to/tb_old --new ./src/tb
```

The tests require Python 3.8+ and a GPU runtime compatible with the supplied
executables. Each case runs both executables in separate directories with
identical inputs, using the selected XYZ fixture as `train.xyz`.

## Cases

The suite contains **32 training cases**: four datasets (`diamond`, `graphene`,
`silicon`, `sige`), four orbital bases (`sp3`, `sp3sstar`, `sp3d5`, `sp3d5sstar`),
and dispersion weights 0 and 0.2.

Each case uses two structures, two epochs, seed 17, and no band curriculum.
The learning rate is 0.003 with plateau patience 20. These short runs check
numerical regression; they do not assess convergence or fitting quality.

To select cases and retain their files:

```sh
python3 tests_tb/run_regression.py --old /path/to/tb_old --new ./src/tb \
  --dataset graphene --basis sp3sstar --keep
```

Both `--dataset` and `--basis` are repeatable. `--timeout` sets the time limit
per executable invocation, in seconds (default: 300).

## Checks and tolerances

Each run checks completed epochs, output dimensions, finite values,
structure/k-point indices, reference bands, and model metadata and scalers.
The following outputs are compared between the two executables:

| File | Compared content |
| --- | --- |
| `loss.out` | All columns except elapsed time |
| `band_train.out` | Structure/k-point indices, coordinates, predicted and reference bands |
| `tb.txt` | Model metadata, parameters, and descriptor scalers |

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

The runner prints PASS/FAIL for each case and returns 0 when all selected cases
pass, or 1 when any case fails.

Failed suites retain their inputs, outputs, `run.log` files, and `summary.json`.
Use `--keep` or `--workdir /absolute/new/directory` to retain successful suites
as well. The report records executable paths and SHA-256 hashes, case results,
effective parameter tolerances, and maximum per-file differences for successful
comparisons. The work directory is printed at startup.

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
