.. _kw_minimize:
.. index::
   single: minimize (keyword in run.in)

:attr:`minimize`
================

This keyword is used to minimize the energy of the system.
Currently, the fast inertial relaxation engine (FIRE) [Bitzek2006]_ [Guénolé2020]_ method and the steepest descent (SD) method have been implemented.


Syntax
------

This keyword is used as follows::

  minimize <method> <force_tolerance> <maximal_number_of_steps> <box_change> <hydrostatic_strain>

Here,
:attr:`method` can be :attr:`sd` (the steepest descent method) or :attr:`fire` (the FIRE method).
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

Fixed-lattice spin FIRE
----------------------

For a Spin potential, optimize spins while keeping atomic positions and the cell
fixed::

  minimize spin_fire <mforce_tolerance> <maximal_number_of_steps> <variable_magnitude>

For example::

  potential nep.txt
  minimize spin_fire 1.0e-4 10000 0
  ensemble nve
  time_step 0
  dump_xyz 1 relaxed.xyz spin mforce
  run 1

Set ``variable_magnitude`` to ``0`` to preserve each atom's initial spin magnitude,
or ``1`` to optimize all three Cartesian spin components. Zero spins remain zero in fixed
mode; variable mode allows them to grow when a nonzero magnetic force is present.
Both modes require ``spin:R:3`` in ``model.xyz``. Atomic velocities and physical
``spin_vel`` are preserved; FIRE uses separate fictitious velocities. The lattice
``fix`` command does not select frozen spins.

The positive finite tolerance is in eV/muB. Convergence uses the largest
per-atom vector norm of the magnetic force. Fixed mode uses only its component
tangent to the constant-magnitude sphere; variable mode uses the full magnetic
force. This differs from the component-wise lattice force criterion above.
Any active ``add_mfield`` contribution is included in the energy and magnetic
force. The model's magnetic-response mask retains its usual meaning.

Fixed mode uses spherical updates and transports the fictitious velocities;
variable mode uses Cartesian updates. Internal step control limits each move
to 0.1 radians or 0.1 muB, respectively. The magnetic force dotted with
FIRE's fictitious velocity controls the internal step and velocity resets.
Energy is diagnostic: individual steps can increase it, and adding a constant
energy offset does not change the optimization. Non-finite trial states are
rolled back, including all endpoint forces, and retried with a smaller step.
The iteration limit counts attempts, including velocity resets and non-finite
trials. The reported internal step is an optimization parameter, not the MD
``time_step``.

Stagnation is reported if the internal step drops below 1e-12, if 20 finite
moves change no spin component by more than 8 double epsilons relative to
``max(1, abs(old_component), abs(new_component))``, or if 1000 attempts elapse
without a cumulative 1% improvement in the best residual. The last criterion
is a practical plateau detector and may also stop very slow relaxation; inspect
the residual and restart if necessary. None of these conditions counts as force
convergence.

Standard output reports the total potential energy, maximum magnetic-force
residual, and termination reason: ``converged``, ``maximum_steps``, or ``stalled``.
Only ``converged`` means that the requested residual was reached. A finite-precision
model can stall above a very tight tolerance. Force convergence identifies a
stationary point, not necessarily the global energy minimum. Variable-magnitude
optimization also requires a potential fitted over the relevant range of moments.
Neither atomic-position nor cell optimization is supported by ``spin_fire``.
