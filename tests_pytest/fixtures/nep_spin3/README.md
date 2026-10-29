# Frozen Spin reference data

These three oracles contain eight cases each: energy, atomic force, atomic
virial, magnetic force, torque and descriptors evaluated in FP64. The reference
uses PyTorch autograd and CPU tensor operations, independently of GPUMD's CUDA
kernels, training model parser and force implementation.

Fixed sources:

- [TorchNEP `3e2a5b3b31803f5cd8733d1fe67180cc7b144ad5`](https://github.com/MagTheoryLab/torchnep/tree/3e2a5b3b31803f5cd8733d1fe67180cc7b144ad5):
  `torchnep/nep.py`, `magnetic_polynomial_v3.py` and their model/descriptor dependencies.
- [NEPAdapters `0ab9f452330e43a21ba76c9969d5a6859a6de31f`](https://github.com/MagTheoryLab/NEPAdapters/tree/0ab9f452330e43a21ba76c9969d5a6859a6de31f):
  `tests/generate_spin2_oc_oracle.py` and
  `tests/fixtures/nep4_spin3_{o3c2_uniform,o3c2_typewise,o3c3_uniform}.nep`.
  Generator SHA256: `5be4782db6df7d25d630866c3d666be29aea0f5bed9936f7e65dfb76b1abad59`.

The fixed model parameters/scalers originate in that generator (seed 20260812).
The bundled GPUMD models serialize those values in the current block format.
The typewise variant derives its `spin_cutoff 4.0 6.0` header from the C2 model.
Each oracle's SHA256 binds it to the corresponding current-format model text.

## Reproduce

Use a CPU Python environment with TorchNEP's dependencies installed (PyTorch,
NumPy and the dependencies declared at the fixed TorchNEP revision). Clone the
repositories into scratch directories, or use existing clones containing the
commits above. From the GPUMD repository root:

```sh
git clone https://github.com/MagTheoryLab/torchnep.git "$RESULTS/torchnep"
git clone https://github.com/MagTheoryLab/NEPAdapters.git "$RESULTS/adapters"
git -C "$RESULTS/torchnep" fetch origin 3e2a5b3b31803f5cd8733d1fe67180cc7b144ad5
git -C "$RESULTS/adapters" fetch origin 0ab9f452330e43a21ba76c9969d5a6859a6de31f
python tests_pytest/reproduce_spin_oracles.py \
  --torchnep "$RESULTS/torchnep" --adapters "$RESULTS/adapters" \
  --output "$RESULTS/fp64-oracles"
```

The wrapper exports the exact committed reference sources into a temporary
directory, so local edits or the clones' current branches do not affect the
calculation. It reads only fixed case inputs from the frozen files, recomputes
all numerical outputs, writes new files outside this directory, and checks them
against the bundled values with a `1e-10` absolute tolerance. It never calls the
candidate NEP executable. `report.json` records versions and maximum errors.

## Why the spin_type change updated references

The model selects Fe. Ge now supplies geometric environment information with
zero effective spin. Ge's input spins were therefore set to zero and all outputs
were recomputed with the independent reference; parameters and scalers stayed
fixed. This changes energies and their derivatives where the previous nonzero
Ge spins contributed. The original full-spin references were reproduced before
this update. Both-species nonzero-spin coverage is provided separately by the
training native/JIT layout comparisons with `spin_type Fe Ge`.

Do not overwrite expected values from GPUMD output. A physics or input-contract
change requires reviewing the input/model change, recomputing with the fixed
independent reference and retaining its report outside the repository.
