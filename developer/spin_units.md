# Spin units and external magnetic field contract

## Units

- `spin:R:3` is a Cartesian magnetic moment in Bohr magnetons (muB), not a unit direction vector. The same convention applies to training spin aliases.
- `spin_vel:R:3` is in muB/fs.
- `mforce:R:3` is `-dE/dspin`, in eV/muB.
- The torque `spin x mforce` is in eV.
- External magnetic induction `B` is entered in tesla (T), not H in A/m.

The arrays store numerical values in these units. Readers do not normalize spins or rescale existing model parameters. Training data and runtime inputs must use the same convention; declaring these units does not convert a model trained with another spin normalization.

## Zeeman coupling

Let `s` be the stored magnetic moment in muB and let `c_B` be the numerical Bohr-magneton conversion, approximately `5.788e-5` eV/T. Then:

```
f_Z [eV/muB] = c_B * B [T]
E_Z [eV] = -sum_i dot(s_i, f_Z_i)
f_total = f_model + f_Z
E_total = E_model + E_Z
```

No extra g factor, spin magnitude, or gyromagnetic factor belongs in this conversion. For example, a 2 muB moment aligned with 1 T has Zeeman energy approximately -1.1576e-4 eV and magnetic force +5.788e-5 eV/muB along the field.

A spatially uniform field has no explicit mechanical force or virial contribution. A spatially varying field requires the position derivative of its Zeeman energy for conservative coupled lattice dynamics. Time-dependent fields do work, so total system energy alone need not be conserved.

## Integration and implementation boundary

Apply the energy and magnetic-force contributions consistently at initial, SIB midpoint, and endpoint evaluations. Use the actual stored spin at each evaluation, including the unnormalized SIB chord midpoint. Recompute contributions after the base arrays are cleared; never accumulate across repeated evaluations.

SIB operates on numerical spin magnitudes in muB and magnetic forces in eV/muB. Its existing gamma convention is a coefficient for these numerical arrays; a tesla-valued field must first pass through the conversion above. Do not feed tesla directly into the existing magnetic-force slot or silently change gamma defaults.

The `add_mfield` command supports constant fields over a selected GPUMD group or dynamic fractional-coordinate region using three tesla-valued components. See `doc/gpumd/input_parameters/add_mfield.rst` for syntax and boundary semantics.
