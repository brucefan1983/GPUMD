.. _kw_minimize:
.. index::
   single: minimize (keyword in run.in)

:attr:`minimize`
================

This keyword is used to minimize the energy of the system.
The fast inertial relaxation engine (FIRE) [Bitzek2006]_ [Guénolé2020]_ and
steepest descent (SD) methods optimize atomic positions. FIRE also supports cell
optimization. For a Spin potential, use ``spin_fire`` to optimize magnetic moments,
optionally together with atomic positions and the cell (see :ref:`spin_fire_minimize`).


Syntax
------

This keyword is used as follows::

  minimize <method> <force_tolerance> <maximal_number_of_steps> <box_change> <hydrostatic_strain>

Here,
:attr:`method` in this lattice-only syntax can be :attr:`sd` (the steepest descent method) or :attr:`fire` (the FIRE method).
:attr:`force_tolerance` is in units of eV/Å.
When the largest absolute force component among the :math:`3N` force components in the system is smaller than :attr:`force_tolerance`, the energy minimization process will stop even though the number of steps (iterations) performed is smaller than :attr:`maximal_number_of_steps`.
:attr:`maximal_number_of_steps` is the maximal number of steps (iterations) to be performed for the energy minimization process.
:attr:`box_change` specifies whether to optimize the box during minimization. It only supports the FIRE method now.
:attr:`hydrostatic_strain` indicates whether hydrostatic strain should be applied to change the box.

Examples
--------

Example 1
^^^^^^^^^
The command::

  minimize sd 1.0e-6 10000

means that one wants to do an energy minimization using the steepest descent method, with a force tolerance of :math:`10^{-6}` eV/Å for up to 10,000 steps.

Example 2
^^^^^^^^^
If you have no idea how small :attr:`force_tolerance` should be, you can simply assign a negative number to it::

  minimize sd -1 10000

In this case, the energy minimization process will definitely run 10,000 steps.

Example 3
^^^^^^^^^
The command::

  minimize fire 1.0e-5 1000

means that one wants to do an energy minimization using the FIRE method, with a force tolerance of :math:`10^{-5}` eV/Å for up to 1,000 steps.

Example 4
^^^^^^^^^^^^^
The command::

  minimize fire 1.0e-5 1000 1 

means that one wants to optimize box during the energy minimization process.

The command::

  minimize fire 1.0e-5 1000 1 1

means that one wants to optimize box using hydrostatic strain during the energy minimization process.

Example 5
^^^^^^^^^
If you want to output the optimized structure, use `dump_xyz` in the `nve` ensemble like the following command::

  minimize fire 1.0e-5 1000 1

  ensemble    nve
  time_step   0
  dump_xyz    1 relaxed.xyz
  run         1

Caveats
-------

* This keyword should occur after the :ref:`potential keyword <kw_potential>`.

.. _spin_fire_minimize:

Spin FIRE
---------

Syntax
^^^^^^

For a Spin potential, use::

  minimize spin_fire <tolerance> <maximal_number_of_steps> <variable_magnitude> [<fixed_atoms> [<box_change> [<hydrostatic_strain>]]]

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
^^^^^^^^

Optimize only spin directions, keeping magnitudes, atoms, and the cell fixed::

  minimize spin_fire 1.0e-4 10000 0

Optimize only spins, allowing their magnitudes to change::

  minimize spin_fire 1.0e-4 10000 1

The same spin-only optimization with atoms explicitly fixed::

  minimize spin_fire 1.0e-4 10000 1 1

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

Caveats
^^^^^^^

* A Spin potential and ``spin:R:3`` in ``model.xyz`` are required. Ordinary
  ``minimize fire`` remains a lattice-only command and is not enabled for Spin
  potentials.
* Cell optimization requires periodic boundaries in all three directions.
  Joint optimization does not support ``fix`` groups and rejects them; in
  spin-only mode, ``fix`` does not select frozen spins.
* Atomic velocities and physical ``spin_vel`` are preserved. FIRE uses separate
  fictitious velocities. After joint optimization, positions, cell inverses,
  periodic images, and endpoint forces are updated for subsequent commands.
* Active ``add_mfield`` contributions enter energy and magnetic force. The
  model's magnetic-response mask retains its usual meaning.
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

Algorithm and stagnation
^^^^^^^^^^^^^^^^^^^^^^^^

The dot product of the generalized force and fictitious velocity controls FIRE's
step size and velocity resets. Energy is diagnostic; a constant energy offset
does not change the optimization. Non-finite trials restore spins, positions,
cell, and endpoint properties before retrying with a smaller step.

Fixed-magnitude spins use spherical updates with parallel transport of fictitious
velocities; variable-magnitude spins use Cartesian updates. Each spin move is
limited to 0.1 radians or 0.1 muB. Joint runs also limit each non-affine atomic
move to 0.1 angstrom and the Frobenius norm of each cell deformation increment to
0.01. Cell coordinates use the initial cube-root volume as a length scale, as
in ordinary cell FIRE. Cell deformation acts on both the cell and atomic
positions; spins stay in laboratory Cartesian coordinates during this affine
part of the update. The full virial transpose supplies the conjugate deformation
force, including nonsymmetric components from spin-orbit interactions.

Stagnation is reported if the internal step drops below 1e-12, if 20 finite moves
produce no resolved update (8 double epsilons relative to the component scale
``max(1, abs(old), abs(new))``), or if 1000 attempts pass without a cumulative 1%
improvement in the largest residual divided by its convergence tolerance. Joint
runs include atomic, magnetic, and active stress residuals. The plateau detector
may stop very slow relaxation; inspect the residuals before restarting. None of
these conditions relaxes the requested convergence criteria. The internal FIRE
step is an optimization parameter, not the MD ``time_step``.
