.. _kw_add_mfield:

:attr:`add_mfield`
==================

Apply a constant external magnetic induction to an atom group or a fractional-coordinate region::

  add_mfield <group_method> <group_id> <Bx> <By> <Bz>
  add_mfield region <amin> <amax> <bmin> <bmax> <cmin> <cmax> <Bx> <By> <Bz>

Grouping method and group ID select an existing model.xyz group, as in
add_efield. The three finite field components are in tesla (T). There is no
separate magnitude or direction normalization. For a 10 T field along z::

  add_mfield 0 0 0 0 10

Group selection follows atom identities even when atoms move. Region selection
instead follows current fractional coordinates along cell vectors a, b, and c,
using the same coordinate conversion and membership test as heat_lan.
Bounds must be finite, in [0, 1], and satisfy min < max. Intervals are half-open:
[min, max). The region scales and tilts with the current cell, including under
NPT. Coordinates are evaluated after normal periodic boundary wrapping; there
is no extra wrapping in nonperiodic directions. No group labels are needed for
region mode.

For a 10 T field along z in the first half of the cell along a::

  add_mfield region 0 0.5 0 1 0 1 0 0 10

Membership is recomputed on every force evaluation: atoms entering the region
receive the field, and atoms leaving it no longer do. The sharp boundary gives
a discontinuous Zeeman energy. No mechanical boundary impulse is implemented;
energy conservation is not guaranteed for atoms crossing this boundary. This
is a local magnetic drive, not a smooth conservative gradient-field model.

A Spin potential and spin:R:3 are required for a nonzero field. The field acts
on all selected magnetic moments, independently of the model spin_dof_type
response mask. It does not impose an integration constraint.

Each command replaces the previous field and group/region selection. The setting
persists across run commands. To turn it off::

  add_mfield 0 0 0 0 0

Spin is in muB, magnetic force is in eV/muB, and energy is in eV. With
c_B = 5.7883818060e-5, the added magnetic force is c_B times the field vector;
the added per-atom Zeeman energy is minus spin dotted with this force.
Both contributions are recomputed at initial, SIB midpoint, GLSD predictor,
and endpoint states using the actual stored moments, including the
unnormalized SIB chord midpoint, and do not accumulate across evaluations.
Output energies and magnetic forces include these contributions.
No additional g factor is applied. There is no direct mechanical force or
virial contribution. Time-dependent fields and linear gradients are not supported.

The field is magnetic induction B, not H in A/m. The integrators consume
magnetic force in eV/muB, so tesla-valued inputs must undergo this conversion
before integration; the conversion does not change their gamma convention.
For example, a 2 muB moment aligned with 1 T has Zeeman energy approximately
-1.1576e-4 eV and magnetic force +5.7884e-5 eV/muB along the field.
