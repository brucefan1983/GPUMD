# NEP model and restart file formats, version 1

This document describes the version-1 `nep.txt` and `nep.restart` file formats, including their fields and parameter layouts.

`nep.txt` contains the model definition and numerical parameters. `nep.restart` contains the SNES optimization state; interpreting its parameter vectors requires the corresponding model definition.

## 1. Scope

Version 1 supports the following model types:

| `model` | Meaning |
|---|---|
| `nep` | Standard energy NEP |
| `nep_spin` | Spin-dependent energy NEP; requires `spin_version` (Section 5.1) |
| `nep_temperature` | Temperature-dependent NEP |
| `qnep1` | qNEP with charge mode 1 |
| `qnep2` | qNEP with charge mode 2 |
| `nep_dipole` | TNEP dipole model |
| `nep_polarizability` | TNEP polarizability model |
| `nep_vdw` | NEP with learned long-range vdW |
| `qnep3_vdw` | qNEP with charge self-energy excluded and learned long-range vdW |

## 2. General syntax of `nep.txt`

A model file starts with an element list, followed by named fields and numerical blocks:

```text
nep 2 Si Ge
format_version 1
model nep
cutoff 5 4 100 80
n_max 6 4
basis_size 8 8
l_max 4 2 1 0 0 0 0
ann 50 0
energy_reference -5.123456789012345 -4.987654321098765
@ parameters 4333
...
@ q_scaler 37
...
```

After the required first line, blank lines are allowed. A `#` starts a comment extending to the end of the line.

The `nep <num_types> <elements...>` record must be the physical first line of the file. No blank line or UTF-8 BOM may precede it.

Field and block names are case-insensitive and unique across both kinds of record. Each field occupies one line.

Except for the first line, records are identified by name rather than by their position in the file.

A block marker has exactly three tokens, `@ <name> <count>`. The following `count` numerical values belong to that block and may be distributed over arbitrary lines.

Unknown fields or blocks are rejected by the model-specific parser.

### 2.1 First line

The first line has the form

```text
nep <num_types> <element_1> <element_2> ...
```

`num_types` must equal the number of listed chemical elements. Element symbols must be valid and must not be repeated. Their order defines the NEP type order.

The first token is always `nep`; the `model` field identifies the model type.

## 3. Common model fields

All models require `format_version`, `model`, `cutoff`, `n_max`, `basis_size`, `l_max`, and `ann`. The requirements for `energy_reference` and `zbl` are described below.

### 3.1 `format_version`

```text
format_version 1
```

The format version is `1`.

### 3.2 `model`

```text
model <model_name>
```

The supported names are listed in Section 1.

### 3.3 `cutoff`

For a model using common radial and angular cutoffs for all elements:

```text
cutoff <rc_radial> <rc_angular> <max_radial_neighbors> <max_angular_neighbors>
```

For example:

```text
cutoff 6 5 274 158
```

A model may instead contain element-dependent cutoffs:

```text
cutoff <r1> <a1> <r2> <a2> ... <rT> <aT> <max_radial_neighbors> <max_angular_neighbors>
```

where `T` is the number of elements.

For types `i` and `j`, the pair cutoffs are `(ri + rj) / 2` and `(ai + aj) / 2`. The final two integers are the largest radial and angular neighbor counts encountered during training. Cutoff distances are in angstroms.

### 3.4 `n_max`

```text
n_max <n_max_radial> <n_max_angular>
```

For example:

```text
n_max 10 8
```

The radial and angular descriptor indices run from zero through their respective `n_max` values. Model-file limits are 15 radially and 12 angularly, subject to the angular descriptor-dimension limit.

### 3.5 `basis_size`

```text
basis_size <basis_size_radial> <basis_size_angular>
```

For example:

```text
basis_size 8 8
```

Basis indices run from zero through the corresponding `basis_size`, inclusive.

### 3.6 `l_max`

The full form is

```text
l_max <L_max> <q_222> <q_1111> <q_112> <q_123> <q_233> <q_134>
```

For example:

```text
l_max 4 2 1 0 0 0 0
```

The last four flags may be omitted when they are zero.

`q_222` is represented by `0` or `2`. The other optional many-body terms are represented by `0` or `1`.

The angular descriptor dimension is

```text
(n_max_angular + 1) *
(L_max + number_of_enabled_optional_many_body_terms)
```

and must not exceed 90.

### 3.7 `ann`

```text
ann <num_neurons_layer1> <num_neurons_layer2>
```

For a one-hidden-layer network:

```text
ann 80 0
```

For a two-hidden-layer network:

```text
ann 80 40
```

A zero second value means one hidden layer. Model-specific restrictions are listed in Section 5.

### 3.8 `energy_reference`

Energy models contain one fixed reference energy per element:

```text
energy_reference <E1> <E2> ... <ET>
```

For example:

```text
energy_reference -5.123456789012345 -4.987654321098765
```

There are exactly `T` float64 values in element order, with no count preceding them. For an atom of type `t`, add `energy_reference[t]` to its NEP atomic energy.

`nep_dipole` and `nep_polarizability` omit this field.

### 3.9 `zbl`

The `zbl` field is optional for energy models and omitted for dipole and polarizability models.

For a universal ZBL potential:

```text
zbl <rc_inner> <rc_outer>
```

For example:

```text
zbl 1 2
```

For a typewise cutoff:

```text
zbl <rc_inner> <rc_outer> <typewise_cutoff_factor>
```

A flexible ZBL model is indicated by

```text
zbl 0 0
```

and additionally requires the `zbl_parameters` block described below.

## 4. Numerical blocks

Every model requires `parameters` and `q_scaler` blocks. Flexible ZBL additionally requires `zbl_parameters`.

Each block starts with `@ <name> <count>`, followed by exactly `count` whitespace-separated values. Line grouping does not affect the array layout.

Block values are float32, written with nine significant digits. 

### 4.1 `parameters`

```text
@ parameters <N>
```

This flattened float32 array contains all trainable parameters, in this order:

```text
ANN set 0
[ANN set 1, if present]
radial descriptor coefficients
angular descriptor coefficients
```

Define

```text
T  = number of elements
nr = n_max_radial
na = n_max_angular
br = basis_size_radial
ba = basis_size_angular
D  = ANN input dimension
N1 = first hidden-layer size
N2 = second hidden-layer size
O  = number of ANN outputs per element
G  = number of global parameters in one ANN set
S  = number of ANN sets
```

The model-dependent values are:

| Model | `O` | `G` | `S` | Extra ANN inputs | Output order |
|---|---:|---:|---:|---:|---|
| `nep` | 1 | 1 | 1 | 0 | energy |
| `nep_spin` | 1 | 1 | 1 | `D_spin` | energy |
| `nep_temperature` | 1 | 1 | 1 | 1 | energy |
| `qnep1` | 2 | 2 | 1 | 0 | energy, charge |
| `qnep2` | 2 | 2 | 1 | 0 | energy, charge |
| `nep_vdw` | 2 | 1 | 1 | 0 | energy, C6 |
| `qnep3_vdw` | 3 | 2 | 1 | 0 | energy, charge, C6 |
| `nep_dipole` | 1 | 1 | 1 | 0 | one TNEP scalar ANN output |
| `nep_polarizability` | 1 | 1 | 2 | 0 | one scalar ANN output per ANN set |

For `nep_polarizability`, ANN set 0 is the tensor-related network and ANN set 1 is the scalar/isotropic network.

Let `M` be the number of nonzero optional flags in `l_max`. The ANN input dimension is

```text
D = (nr + 1) + (na + 1) * (L_max + M) + extra_inputs.
```

The global parameters within one ANN set are ordered as follows:

| Model | Global parameter order |
|---|---|
| `nep`, `nep_spin`, `nep_temperature`, `nep_vdw`, `nep_dipole` | common output bias |
| `nep_polarizability` | common output bias for each ANN set |
| `qnep1`, `qnep2`, `qnep3_vdw` | `sqrt_epsilon_inf`, common energy-output bias |

#### 4.1.1 ANN-set ordering

One ANN set is stored as

```text
parameters for element 0
parameters for element 1
...
parameters for element T-1
G global parameters
```

Each set includes its own global parameters. Descriptor coefficients follow all ANN sets.

#### 4.1.2 One-hidden-layer ANN

For `N2 = 0`, each element stores these arrays in order:

```text
W0    : [N1, D]
b0    : [N1]
Wout  : [O, N1]
```

All multidimensional arrays use C/row-major order: the last index varies fastest.

The number of parameters for one element is

```text
A_element = N1 * D + N1 + O * N1
          = N1 * (D + 1 + O).
```

For the first (energy) output of an energy model,

```text
h[n]  = tanh(sum_d W0[n,d] * q[d] - b0[n])
E_ANN = sum_n Wout[0,n] * h[n] - b_global
```

where `b_global` is the common output bias stored after all element-specific ANN parameters in the ANN set. Additional outputs such as charge and C6 use subsequent rows of `Wout`; they do not have this energy-output bias.

#### 4.1.3 Two-hidden-layer ANN

For `N2 > 0`, there is one ANN output. Each element stores these arrays in order, also in C/row-major order:

```text
W0 : [N1, D]
b0 : [N1]
W1 : [N2, N1]
b1 : [N2]
W2 : [N2]
```

The number of parameters for one element is

```text
A_element = N1 * D + N1 + N2 * N1 + N2 + N2
          = (D + 1) * N1 + (N1 + 2) * N2.
```

The network evaluates

```text
h1 = tanh(W0 @ q  - b0)
h2 = tanh(W1 @ h1 - b1)
y  = W2 @ h2 - b_global
```

#### 4.1.4 Size of the ANN part

For one ANN set,

```text
A_set = T * A_element + G.
```

For `S` ANN sets,

```text
A_total = S * A_set.
```

The descriptor coefficients start at zero-based offset `A_total` in the `parameters` block.

#### 4.1.5 Descriptor coefficient ordering

Define

```text
R = (nr + 1) * (br + 1)
A = (na + 1) * (ba + 1)
P = T * T.
```

There are

```text
C_radial  = R * P
C_angular = A * P
```

descriptor coefficients.

The type pair is ordered: the first type is the center atom and the second type is the neighbor atom.

```text
type_pair = t_center * T + t_neighbor.
```

Radial coefficients are stored in the logical C-order array

```text
[T, T, nr + 1, br + 1]
```

with indices

```text
[t_center, t_neighbor, n, k].
```

The zero-based index within the radial segment is

```text
radial_index
= ((t_center * T + t_neighbor) * (nr + 1) + n) * (br + 1)
  + k.
```

The angular coefficients immediately follow the radial coefficients. They form the logical C-order array

```text
[T, T, na + 1, ba + 1]
```

and the zero-based index within the angular segment is

```text
angular_index
= ((t_center * T + t_neighbor) * (na + 1) + n) * (ba + 1)
  + k.
```

Thus a Python parser can recover the descriptor coefficient arrays directly with

```python
radial = radial_flat.reshape(T, T, nr + 1, br + 1)
angular = angular_flat.reshape(T, T, na + 1, ba + 1)
```

using NumPy's default C-order reshape.

#### 4.1.6 Total parameter count

The number of descriptor coefficients is

```text
C_descriptor
= T^2 * [
    (nr + 1) * (br + 1)
    + (na + 1) * (ba + 1)
  ].
```

The complete parameter count is

```text
N
= S * (T * A_element + G)
  + C_descriptor.
```

`nep_spin` appends magnetic radial and projection coefficients to this array;
its complete count is given in Section 5.1.

This value must exactly match both the count declared in `@ parameters N` and the number of numerical values in that block.

### 4.2 `q_scaler`

```text
@ q_scaler <D>
```

There is one float32 factor per ANN input: radial descriptors first, then angular descriptors, followed by the temperature input for `nep_temperature`. Multiply each input component by its corresponding factor before evaluating the ANN.

Radial components follow increasing `n`. Angular components are grouped by invariant: `l = 1, ..., L_max`, then the enabled terms `q_222`, `q_1111`, `q_112`, `q_123`, `q_233`, and `q_134`, in that order. Within each angular group, `n` runs from zero through `n_max_angular`.

For `nep_spin`, magnetic components follow all structural components; see
Section 5.1 for their order.

The `q_scaler` block is not part of the trainable `parameters` vector and is not included in `N`.

### 4.3 `zbl_parameters`

Flexible ZBL models additionally contain

```text
@ zbl_parameters <N>
```

with

```text
N = 10 * T * (T + 1) / 2.
```

There is one group of ten values for every unordered type pair `i <= j`. The pair order is

```text
(0,0), (0,1), ..., (0,T-1),
(1,1), (1,2), ..., (1,T-1),
...,
(T-1,T-1).
```

For `i <= j`, the zero-based pair index is

```text
pair_index
= i * T - i * (i - 1) / 2 + (j - i).
```

Each pair stores ten contiguous values: `[rc_inner, rc_outer, a1, b1, a2, b2, a3, b3, a4, b4]`. The screening function is `phi(x) = sum_i ai * exp(-bi * x)`.

A Python parser can therefore use

```python
zbl = zbl_flat.reshape(T * (T + 1) // 2, 10)
```

with the rows ordered as above.

## 5. Model-specific requirements

`qnep1`, `qnep2`, `nep_vdw`, and `qnep3_vdw` require one hidden layer. `qnep1` and `qnep2` also require uniform cutoffs.

The model tables in Section 4.1 determine the number of outputs, global parameters, and ANN sets. Use these together with the descriptor and network dimensions to determine all array lengths.

### 5.1 Spin-dependent models: `model nep_spin`

New Spin models use:

```text
nep 2 Fe Ge
format_version 1
model nep_spin
spin_version 1
```

`format_version` describes the shared file syntax. `spin_version` describes the
magnetic descriptor definitions, units, scaler interpretation, and parameter
layout. Version `1` is the current order/compression (Spin3) model, including its
`spin_type` selection of independent spin coordinates.
`spin_order` selects O1/O2/O3 within this contract and is not a version number.

`spin_version` is required and must contain exactly the integer token `1`.
Missing or unknown versions are errors. Changes to descriptor meaning or array
layout that reinterpret existing parameter values require a new supported spin
version. Training-loss settings, implementation optimizations, and bug fixes
that restore the specified behavior do not change the version.

Spin output preserves the model element order; automatic element compaction
is disabled.

The model fields below supplement the common energy-model fields:

| Field | Version-1 meaning |
|---|---|
| `spin_version 1` | Magnetic model contract version |
| `spin_basis_size 8` | Highest magnetic radial basis index; nine basis functions |
| `spin_l_max L` | Magnetic angular rank, `L` in `0..2` |
| `spin_compress C` | Number of compressed radial channels, `C` in `1..9` |
| `spin_order O` | Contraction order, `O` in `1..3` |
| `spin_soc S` | Joint spin/spatial channels: `0` or `1` |
| `spin_cutoff ...` | One positive cutoff, or `T` cutoffs in element order, in angstroms; pair cutoff is their arithmetic mean |
| `spin_projection_size P` | Number of projection coefficients; must equal `4*C*C` |
| `spin_scaler 1` | Per-component scaling is present in `q_scaler` |
| `spin_type <elements...>` | Optional nonempty subset with independent spin degrees of freedom; defaults to all elements |

All element types contribute geometric environments within the cutoffs.
Only `spin_type` elements carry spin degrees of freedom and act as magnetic
descriptor centers. Readers set other elements' effective spin to zero and
ignore their magnetic-force labels, preserving energy and atomic-force labels
without changing the input files. Their public magnetic force is zero, but
atomic forces remain active.
Spins use muB,
energy uses eV, and magnetic force uses eV/muB. For descriptor definitions and
scaling formulas, see [the Spin potential documentation](../../doc/potentials/nep_spin.rst).

Version 1 supports one hidden layer (`ann N1 0`, `1 <= N1 <= 120`) and uniform
structural cutoffs. Structural limits are `nr <= 12`, `na <= 8`, `br <= 16`,
`ba <= 12`, and `L_max <= 8`, with at most 90 structural angular components.
Training supports at most 96 magnetic components, and the combined descriptor
must fit 199 components. The structural and magnetic dimensions are determined
by the enabled contractions, not by free metadata counts.

#### Parameter and scaler layout

Let `D = D_struct + D_spin`. With `O = G = S = 1` in the notation of Section 4.1
(these symbols there describe ANN outputs, global parameters, and ANN sets),
the parameter vector is:

```text
ANN parameters and common energy-output bias
structural radial coefficients   [T, T, nr + 1, br + 1]
structural angular coefficients  [T, T, na + 1, ba + 1]
magnetic radial coefficients    [C, 9, T, T]
magnetic projection coefficients [4, C, C]
```

All arrays use C order. The magnetic radial indices are
`[channel, basis, center_type, neighbor_type]`; their type pair is the fastest
varying index, unlike the structural segments. Projection indices are
`[leg, output_channel, source_channel]`. The total parameter count is

```text
N = T * N1 * (D + 2) + 1
    + T*T * ((nr + 1)*(br + 1) + (na + 1)*(ba + 1))
    + 9*C*T*T + 4*C*C.
```

The scaler contains `D` values, with structural components first as in Section
4.2. Magnetic components follow the version-1 contraction order defined by
`make_spin_polynomial_layout` in [nep_spin3_layout.cuh](../../src/force/nep_spin/nep_spin3_layout.cuh):
local squared moment; order-1 edge terms; order-2 edge/density/correlation and
axial terms; then order-3 gated and coupled terms. Disabled families contribute
no entries. Channels increase within a family; channel-pair families use
`(0,0), (0,1), ..., (0,C-1), (1,1), ..., (C-1,C-1)`.

`energy_reference`, optional ZBL, numerical precision, and the restart syntax
remain the shared conventions. `nep.restart` has no separate spin version: it
must be interpreted with the corresponding `nep.txt`, whose version is checked
before loading the restart. Loss weights such as `lambda_m_parallel` and
`lambda_m_perp` are training inputs, not model-file fields.

#### Training input

Select this model in `nep.in` with `spin_version 1` and the ordinary NEP
architecture selector `version 4`. Omit `spin_version` for ordinary NEP.
Unsupported versions and repeated or conflicting model selectors are errors.
Prediction, scaler import, and restart require a compatible `nep.txt`.
Do not put the model-file fields `model` or `format_version` in `nep.in`.

## 6. `nep.restart`

The restart stores elemental energy references and the SNES mean (`mu`) and standard deviation (`sigma`) for every trainable parameter:

```text
nep_restart
energy_reference <num_references> <E1> <E2> ...
parameters <N>
<mu_1> <sigma_1>
<mu_2> <sigma_2>
...
<mu_N> <sigma_N>
```

Read these records in the order shown. Unlike `nep.txt`, the restart has no `format_version` field or `@` block markers, and its `energy_reference` record includes a count.

`num_references` is `T` for energy models. Dipole and polarizability models use `energy_reference 0`, with no reference values.

Read the remaining values as an `[N, 2]` array: column 0 is `mu`, and column 1 is `sigma`. Parameter index `i` has exactly the same meaning as index `i` in the corresponding model's `parameters` block (Section 4.1). The count `N` must match that model's total parameter count.

Energy references are float64, written with 17 significant digits. `mu` and `sigma` are float32, written with nine significant digits; `sigma` is non-negative.

## 7. Legacy model conversion

The converter below covers the ordinary NEP model families; it does not convert
Spin files. Spin models are written directly by the trainer in the format
described in Section 5.1.

Supported legacy model files can be converted with

```bash
python3 potentials/nep/nep_model_converter.py \
    old_nep.txt new_nep.txt
```

A matching legacy restart can be converted at the same time:

```bash
python3 potentials/nep/nep_model_converter.py \
    old_nep.txt new_nep.txt \
    --restart old_nep.restart \
    --restart-output new_nep.restart
```
