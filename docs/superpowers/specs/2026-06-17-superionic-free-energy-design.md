# Superionic Free Energy Design

## Goal

Add a general two-stage thermodynamic integration workflow for superionic systems in GPUMD. The reference state combines an Einstein crystal for selected solid-like species and Uhlenbeck-Ford (UF) fluids for selected mobile species. The feature must support systems such as AlOOH in both H-only diffusion mode, where Al and O are Einstein species and H is a UF fluid, and H+Al diffusion mode, where O is an Einstein species and Al and H are UF fluids.

The implementation should follow GPUMD coding style and add one new CUDA source/header pair for the ensemble logic:

```text
src/integrate/ensemble_ti_superionic.cu
src/integrate/ensemble_ti_superionic.cuh
```

## Commands

Expose two user-facing ensemble commands:

```text
ensemble ti_superionic_stage1 ...
ensemble ti_superionic_stage2 ...
```

Both commands instantiate the same implementation class, `Ensemble_TI_Superionic`, with an internal stage flag. They use the existing GPUMD ensemble rhythm: NVT Langevin integration, `temp`, `tperiod`, `tequil`, `tswitch`, and `press` parameters, and the same smooth forward/backward lambda switching style as `ti_spring` and `ti_liquid`.

The `press` value is used only when reporting Gibbs free energies in post-processing. It must not alter the dynamics.

### Parameter Syntax

The stage commands share this syntax:

```text
ensemble ti_superionic_stage1 temp <T> tperiod <tau> tequil <steps> tswitch <steps> press <P> \
  spring <element> <k> <element> <k> ... \
  uf <element_i> <element_j> <p> <sigma> \
  uf <element_i> <element_j> <p> <sigma> ...

ensemble ti_superionic_stage2 temp <T> tperiod <tau> tequil <steps> tswitch <steps> press <P> \
  spring <element> <k> <element> <k> ... \
  uf <element_i> <element_j> <p> <sigma> \
  uf <element_i> <element_j> <p> <sigma> ...
```

Explicit spring constants use:

```text
spring Al 12 O 12
```

Automatic spring estimation uses:

```text
spring auto Al O
```

When `spring auto` is used, spring constants are estimated during the equilibration period from the mean-squared displacement, following the existing `ti_spring` method:

```text
k = 3 k_B T / <|r_i - r_i0|^2>
```

UF parameters are given per element pair and do not include a cutoff:

```text
uf H H 25 1.0
uf Al H 10 1.0
uf O H 10 1.0
```

The new ensemble reuses the radial neighbor list provided by the main GPUMD potential, matching the existing `ti_liquid` behavior. Users must choose a main potential whose radial neighbor list covers the distances needed by the requested UF interactions.

## Physical Path

Stage 1 represents the reference-to-auxiliary path:

```text
H1(lambda) = H_ref + lambda * U_UF_cross
lambda = 0: reference state
lambda = 1: auxiliary state
```

The reference state is:

```text
H_ref = U_Einstein(spring species) + U_UF_self(fluid species)
```

The auxiliary cross term is the sum of all UF pairs where `element_i != element_j`.

Stage 2 represents the auxiliary-to-target path:

```text
H2(lambda) = (1 - lambda) * H_aux + lambda * H_target
lambda = 0: auxiliary state
lambda = 1: target state
```

The auxiliary state is:

```text
H_aux = U_Einstein + U_UF_self + U_UF_cross
```

The target state is the normal GPUMD potential energy and force computed from the `potential` command. For Stage 1, the main potential is still required so GPUMD can set up normal structure, box, and neighbor-list state, but the Stage 1 ensemble overwrites the main-potential force/energy for the actual dynamics and work calculation.

## Lambda Schedule And Work

Both stages use the same two-direction lambda schedule:

```text
equilibrate at lambda = 0
forward:  lambda 0 -> 1
equilibrate at lambda = 1
backward: lambda 1 -> 0
```

The switch function and derivative should match `ti_spring` and `ti_liquid`.

The stage work values are accumulated with explicit signs:

```text
W_forward  = integral_forward  dHdlambda d_lambda
W_backward = integral_backward dHdlambda d_lambda
delta_F    = 0.5 * (W_forward - W_backward)
```

With this convention:

```text
stage1 delta_F = F_aux - F_ref
stage2 delta_F = F_target - F_aux
F_target       = F_ref + delta_F_stage1 + delta_F_stage2
```

No statistical error estimate is required in the first version.

## Force And Energy Data Flow

The class needs separate GPU buffers for:

```text
U_Einstein per atom
U_UF_self per atom
U_UF_cross per atom
reference/auxiliary force per atom
type masks and UF pair parameter matrices
```

### Stage 1

Stage 1 computes only reference and cross UF forces:

```text
force = F_Einstein + F_UF_self + lambda * F_UF_cross
dHdlambda = U_UF_cross
```

The existing main-potential force from `Force::compute` is ignored for dynamics.

### Stage 2

Stage 2 uses the force already computed by GPUMD as the target force, then mixes it with the auxiliary force:

```text
force = (1 - lambda) * F_aux + lambda * F_target
dHdlambda = U_target - U_aux
```

where:

```text
U_aux = U_Einstein + U_UF_self + U_UF_cross
```

## CSV Output

Each stage writes an independent CSV file.

Stage 1:

```text
ti_superionic_stage1.csv
lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda
```

Stage 2:

```text
ti_superionic_stage2.csv
lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda
```

Energies are reported per atom, matching the style of existing TI CSV outputs.

## YAML Output

Each stage writes an independent YAML file:

```text
ti_superionic_stage1.yaml
ti_superionic_stage2.yaml
```

Each stage YAML includes:

```yaml
stage: <1|2>
T: <temperature>
V: <volume per atom>
P: <pressure converted to GPUMD energy/volume units>
N_total: <number of atoms>
spring_species: [...]
uf_self_pairs: [...]
uf_cross_pairs: [...]
W_forward: <eV/atom>
W_backward: <eV/atom>
delta_F: <eV/atom>
F_Einstein: <eV/atom for spring species contribution divided by total atom count>
F_UF_self: <eV/atom for self UF contribution divided by total atom count>
F_ref: <F_Einstein + F_UF_self>
```

The stage YAML does not try to read the other stage or report the final target free energy. That summary is delegated to a post-processing tool.

## Reference Free Energy

The reference free energy is:

```text
F_ref = F_Einstein(spring species) + sum F_UF(self fluid species)
```

Einstein terms are computed only for atoms whose element appears in `spring`. They are normalized by the total number of atoms, so all stage outputs and post-processing use eV/atom for the full target system.

UF self terms are computed only for pairs where `element_i == element_j`, for example `uf H H 25 1.0` or `uf Al Al 50 1.0`. Cross UF pairs, such as `uf Al H 10 0.4`, are not part of `F_ref`; they only contribute to Stage 1 and Stage 2 work.

The first version reuses the existing `ti_liquid` spline data and restrictions for analytic UF self free energies. Therefore, self UF `p` must be one of:

```text
1, 25, 50, 75, 100
```

Cross UF pairs may use any positive `p` and `sigma` because they are integrated numerically and do not need an analytic reference free energy.

## Post-Processing Tool

Add a lightweight post-processing tool:

```text
tools/si_free_energy_sum.py
```

It reads the two stage YAML files and writes a summary YAML containing:

```yaml
stage1:
  W_forward: ...
  W_backward: ...
  delta_F: ...
stage2:
  W_forward: ...
  W_backward: ...
  delta_F: ...
F_ref: ...
F_target: ...
G_target: ...
T: ...
V: ...
P: ...
```

The summary formulas are:

```text
F_target = F_ref + delta_F_stage1 + delta_F_stage2
G_target = F_target + P * V
```

The tool validates that Stage 1 and Stage 2 use the same temperature, compatible volume/pressure metadata, and matching reference-state definitions.

## GPUMD Integration Points

Required source integration:

```text
src/integrate/integrate.cu
src/integrate/ensemble_ti_superionic.cu
src/integrate/ensemble_ti_superionic.cuh
src/makefile
```

`integrate.cu` must include the new header and add parse branches for:

```text
ti_superionic_stage1
ti_superionic_stage2
```

Both stage types should use the `compute3` path because they need access to `Force&` and its neighbor list, like `ti_liquid`.

## Error Handling

The first version should reject:

- Missing `spring`.
- A `spring` element that does not occur in the structure.
- Missing `uf` definitions.
- Missing UF self pair, because then there is no UF fluid reference contribution.
- Non-positive UF `p` or `sigma`.
- Self UF `p` outside `1, 25, 50, 75, 100`.
- Unknown keywords.
- Mixed explicit and automatic spring syntax in one command.

The first version only documents the main-potential neighbor-list coverage requirement. It does not try to infer or validate whether the existing neighbor list is long enough for all UF pairs.

## Validation Plan

Initial validation should include:

- Build check: `make` succeeds with the new source file.
- Regression check: existing `ti_spring` and `ti_liquid` behavior remains unchanged.
- Parser checks for explicit spring, automatic spring, unknown elements, invalid self UF `p`, and malformed UF entries.
- A small deterministic Stage 1 test where `dHdlambda` equals `U_UF_cross`.
- A Stage 2 smoke test that verifies CSV/YAML files are created and contain finite `W_forward`, `W_backward`, and `delta_F`.
- A post-processing test using two simple YAML fixtures to verify `F_target` and `G_target` formulas.

## Deferred Work

Not included in the first version:

- Error estimates or replica statistics.
- Automatic verification of UF cutoff coverage.
- Reading Stage 1 output from Stage 2.
- Running both stages in one GPUMD command.
- Analytic UF self free energies for unsupported `p` values such as `p = 10`.
