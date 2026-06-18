**Summary**

This branch adds a two-stage thermodynamic integration workflow for calculating free energies of superionic systems in GPUMD. The new workflow supports hybrid reference states in which selected species are treated as Einstein-crystal atoms and selected mobile species are treated as Uhlenbeck-Ford (UF) fluids.

The implementation exposes two ensemble commands:

```text
ensemble ti_superionic_stage1 ...
ensemble ti_superionic_stage2 ...
```

Stage 1 integrates from the reference state to the auxiliary state:

```text
H1(lambda) = H_ref + lambda * U_UF_cross
dH1/dlambda = U_UF_cross
```

Stage 2 integrates from the auxiliary state to the target state:

```text
H2(lambda) = (1 - lambda) * H_aux + lambda * H_target
dH2/dlambda = U_target - U_aux
```

The final Helmholtz free energy is obtained as:

```text
F_target = F_ref + delta_F_stage1 + delta_F_stage2
```

**Modification**

- Added `Ensemble_TI_Superionic` for `ti_superionic_stage1` and `ti_superionic_stage2`.
- Added per-element Einstein spring references, including explicit spring constants and `spring auto` estimation from MSD.
- Added per-element-pair UF references with separate self and cross pair handling.
- Reused the main potential radial neighbor list for UF force and energy evaluation.
- Added CSV output for per-step lambda, dlambda, potential-energy components, and `dHdlambda`.
- Added YAML output for stage metadata, forward/backward work, stage `delta_F`, `F_Einstein`, `F_UF_self`, and `F_ref`.
- Added a shared `uf_reference.cuh` helper so UF reference free-energy tables are shared with `ti_liquid`.
- Added `tools/si_free_energy_sum.py` to combine stage YAML files and report final `F_target` and `G_target`.
- Added tests and minimal GPUMD fixtures for the new stage commands and summary tool.
- Added implementation documentation in `docs/ti_superionic_source_walkthrough.md`.

**Others**

Validation performed:

```text
make clean && make gpumd
pytest tests/gpumd/ti-superionic/test-ti-superionic.py -q
pytest tests/tools/test_si_free_energy_sum.py -q
```

Notes:

- The `press` value is used only for reporting Gibbs free energy through `G = F + PV`; it does not alter the simulation dynamics.
- Cross UF pairs are not included in the analytic reference free energy. They contribute through the Stage 1 and Stage 2 integration values.
- Existing non-superionic workflows are not expected to change. The `ti_liquid` UF reference tables were moved into a shared helper without changing the underlying UF formula.
