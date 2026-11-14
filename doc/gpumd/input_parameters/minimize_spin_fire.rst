.. _spin_fire_minimize:

:attr:`minimize` (Spin FIRE)
============================

Optimize magnetic moments, optionally together with atomic positions and cell strain.

Syntax
------

For a Spin potential, use::

  minimize spin_fire <tolerance> <maximal_number_of_steps> <variable_magnitude> [<fixed_atoms> [<box_change> [<hydrostatic_strain>]]]

Parameters
----------

* ``variable_magnitude = 0`` preserves each atom's initial spin magnitude;
  ``1`` optimizes all three Cartesian spin components. Zero spins remain zero
  in mode ``0`` and can grow in mode ``1``.
* ``fixed_atoms`` defaults to ``1``: atomic positions and the cell stay fixed,
  and only spins are optimized. Set it to ``0`` to enable atomic relaxation.
* ``box_change`` defaults to ``0`` (fixed cell). Set it to ``1`` to jointly
  optimize atoms, spins, and the cell; this requires ``fixed_atoms = 0``.
* ``hydrostatic_strain`` defaults to ``0`` (all nine deformation components).
  Set it to ``1`` to apply only isotropic scaling, preserving the cell shape.
  It requires ``box_change = 1``. The two cell flags have the same meanings as
  in ordinary FIRE, and follow the spin-magnitude and fixed-atom flags here.
* Optional flags are positional: ``fixed_atoms``, then ``box_change``, then
  ``hydrostatic_strain``. Supply preceding flags when setting a later one.
* ``tolerance`` must be finite and positive. The maximum per-atom magnetic-force
  **vector norm** must be at most this value in eV/muB. Mode ``0`` tests only the
  component tangent to each constant-magnitude sphere; mode ``1`` tests the full
  magnetic force.
* When atoms are optimized, their largest absolute force **component** must also
  be at most ``tolerance`` in eV/angstrom, as in ordinary FIRE. Thus one numeric
  tolerance is applied separately to atomic and magnetic forces, in their
  respective units.
* When the cell is optimized, the largest absolute virial stress component must
  additionally be at most ``1e-4 GPa``, the same zero-pressure target as ordinary
  FIRE. With ``hydrostatic_strain = 1``, only the mean normal stress is tested;
  shear and deviatoric stresses need not vanish.
* ``maximal_number_of_steps`` is a positive iteration budget. All attempts count,
  including velocity resets and non-finite trial states.

Examples
--------

Optimize only spin directions, keeping magnitudes, atoms, and the cell fixed::

  minimize spin_fire 1.0e-4 10000 0

Optimize only spins, allowing their magnitudes to change::

  minimize spin_fire 1.0e-4 10000 1

Jointly optimize atoms and variable-magnitude spins at fixed cell::

  minimize spin_fire 1.0e-4 20000 1 0

Jointly optimize atoms, variable-magnitude spins, and the cell::

  minimize spin_fire 1.0e-4 20000 1 0 1

Keep spin magnitudes fixed during joint optimization::

  minimize spin_fire 1.0e-4 20000 0 0 1

Allow only isotropic cell scaling during joint optimization::

  minimize spin_fire 1.0e-4 20000 1 0 1 1

Write the optimized structure and magnetic moments using the same zero-time-step
output procedure as ordinary FIRE::

  potential nep.txt
  minimize spin_fire 1.0e-4 20000 1 0 1
  ensemble nve
  time_step 0
  dump_xyz 1 relaxed.xyz precision double force spin mforce virial
  run 1

Notes
-----

* A Spin potential and ``spin:R:3`` in ``model.xyz`` are required. Ordinary
  ``minimize fire`` remains a lattice-only command and is not enabled for Spin
  potentials.
* Cell optimization requires periodic boundaries in all three directions.
  Joint optimization does not support ``fix`` groups and rejects them; in
  spin-only mode, ``fix`` does not select frozen spins.
* Atomic velocities and physical ``spin_vel`` are preserved. Subsequent commands use the optimized configuration.
* External magnetic fields do not participate in Spin FIRE minimization.
  ``add_mfield`` applies only to MD runs.
* Output reports energy, maximum magnetic-force residual, and (for joint runs)
  maximum atomic force and virial stress in GPa (mean normal stress in the
  hydrostatic case). At fixed cell, stress is reported but is not a stopping
  criterion. Only ``converged`` means **all active criteria** have been met.
  ``maximum_steps`` and ``stalled``
  do not indicate convergence.
* Individual FIRE steps can increase the energy. Force convergence identifies
  a stationary point, not necessarily the global minimum or a particular phase.
  Variable-magnitude optimization requires a potential fitted over the relevant
  range of moments, atomic configurations, and cell deformations.

If minimization stalls, inspect the reported residuals before restarting.
Stagnation does not relax the convergence criteria.
The internal FIRE step is an optimization parameter, independent of the MD ``time_step``.
