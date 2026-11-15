.. _spin_fire_minimize:

:attr:`minimize` (Spin FIRE)
============================

Relax magnetic moments using FIRE minimization, optionally together with atomic
positions and the cell. A Spin potential and ``spin:R:3`` in ``model.xyz`` are
required.

Syntax
------

::

    minimize spin_fire <tolerance> <maximal_number_of_steps> <variable_magnitude> [<fixed_atoms> [<box_change> [<hydrostatic_strain>]]]

Parameters
----------

``tolerance``
  Required finite positive force tolerance. The maximum per-atom magnetic-force
  vector norm must not exceed it, in eV/muB. With fixed spin magnitudes, only
  the tangent component is tested; with variable magnitudes, the full magnetic
  force is tested.

``maximal_number_of_steps``
  Required positive integer iteration budget. Attempts that reset velocities
  or reject nonfinite trial states also count toward this budget.

``variable_magnitude``
  Required integer flag, ``0`` or ``1``. ``0`` preserves each initial spin
  magnitude and optimizes directions; zero moments remain zero. ``1`` allows
  both directions and magnitudes to change, including growth from zero.

``fixed_atoms``
  Integer flag, ``0`` or ``1``; default ``1``. ``1`` fixes atoms and the cell.
  ``0`` enables atomic relaxation. The largest absolute atomic-force component
  must then also be at most ``tolerance``, in eV/angstrom. The same numerical
  tolerance is applied separately in the two force units.

``box_change``
  Integer flag, ``0`` or ``1``; default ``0``. ``1`` enables zero-pressure cell
  relaxation and requires ``fixed_atoms = 0`` and periodic boundaries in all
  three directions. The largest absolute active virial-stress component must
  additionally be at most ``1e-4 GPa``.

``hydrostatic_strain``
  Integer flag, ``0`` or ``1``; default ``0``. ``0`` allows general cell
  deformation. ``1`` allows only isotropic scaling, preserving the cell shape,
  and requires ``box_change = 1``. In this mode, the stress convergence test
  uses the mean normal stress; shear and deviatoric stresses need not vanish.

The optional flags are positional. Supply preceding flags when setting a later
one. The three required parameters have no defaults.

Output
------

The log reports progress using ``spin_fire step``, including energy in eV,
``max_mforce`` in eV/muB and the internal FIRE step ``dt``. For joint relaxation,
it also reports ``max_force`` in eV/angstrom and ``max_stress_GPa``. Stress is
reported at fixed cell but is not a convergence criterion there. FIRE's ``dt``
is an optimization parameter and is unrelated to the MD ``time_step``.

The final ``Spin FIRE finished:`` line reports one of these states:

``converged``
  Every active magnetic-force, atomic-force and cell-stress criterion has been
  satisfied. Inspect the final residuals to confirm the requested accuracy.

``maximum_steps``
  The iteration budget was exhausted before convergence. Inspect the residuals
  and, if they are still improving, increase the budget or continue from the
  resulting configuration. This state does not mean convergence.

``stalled``
  The optimizer could no longer make resolved progress. The ``stagnation``
  field identifies a step-size floor, unresolved updates or lack of residual
  improvement. Check the residuals, the initial configuration and the model's
  force precision. Increasing the iteration budget alone may not help; change
  the starting state or revise tolerances only if the intended physical
  accuracy permits it. This state does not mean convergence.

The final line also counts velocity resets (``power_resets``) and rejected
nonfinite trials (``nonfinite_retries``). A nonfinite initial state is an error.
Nonfinite trial states are rejected and retried with a smaller optimization
step.

Examples
--------

Optimize spin directions with fixed magnitudes and fixed atoms::

    minimize spin_fire 1.0e-4 10000 0

Optimize directions and magnitudes with fixed atoms::

    minimize spin_fire 1.0e-4 10000 1

Relax atoms and variable-magnitude spins at fixed cell::

    minimize spin_fire 1.0e-4 20000 1 0

Relax atoms, variable-magnitude spins and the cell::

    minimize spin_fire 1.0e-4 20000 1 0 1

Allow only isotropic cell scaling during joint relaxation::

    minimize spin_fire 1.0e-4 20000 1 0 1 1

Write the optimized configuration and magnetic moments::

    potential nep.txt
    minimize spin_fire 1.0e-4 20000 1 0 1
    ensemble nve
    time_step 0
    dump_xyz 1 relaxed.xyz precision double force spin mforce virial
    run 1

Notes
-----

* Individual steps may increase the energy. Force convergence identifies a
  stationary point, not necessarily the global minimum or a particular phase.
* Variable-magnitude and joint relaxation require a potential fitted over the
  relevant moments, atomic configurations and cell deformations.
* Atomic velocities and physical spin velocities (``spin_vel``) are preserved.
* Joint relaxation does not support ``fix`` groups. In spin-only mode, ``fix``
  does not select frozen spins.
* External fields from ``add_mfield`` apply to MD runs and do not participate
  in Spin FIRE minimization.
* Ordinary ``minimize fire`` is a lattice-only command for non-Spin potentials.
