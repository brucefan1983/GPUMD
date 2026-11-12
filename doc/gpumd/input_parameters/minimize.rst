.. _kw_minimize:
.. index::
   single: minimize (keyword in run.in)

:attr:`minimize`
================

This keyword is used to minimize the energy of the system.
Minimization is performed immediately when this command is read from ``run.in`` and does not require a :ref:`run <kw_run>` command.
Subsequent commands use the resulting atomic positions and simulation cell.
Currently, the fast inertial relaxation engine (FIRE) [Bitzek2006]_ [Guénolé2020]_ method and the steepest descent (SD) method have been implemented.


Syntax
------

This keyword is used as follows::

  minimize sd <force_tolerance> <maximal_number_of_steps>
  minimize fire <force_tolerance> <maximal_number_of_steps> [<box_change> [<hydrostatic_strain>]]

The methods are :attr:`sd` (steepest descent) and :attr:`fire` (FIRE).
:attr:`force_tolerance` is in units of eV/Å.
For a fixed box, when the largest absolute force component among the :math:`3N` force components in the system is smaller than :attr:`force_tolerance`, the energy minimization process will stop even though the number of steps (iterations) performed is smaller than :attr:`maximal_number_of_steps`.
:attr:`maximal_number_of_steps` is the maximal number of steps (iterations) to be performed for the energy minimization process.
:attr:`box_change` is an optional FIRE parameter: ``0`` (default) keeps the box fixed, and ``1`` optimizes the box.
:attr:`hydrostatic_strain` is also optional and defaults to ``0``. With ``box_change 1``, setting it to ``1`` restricts box changes to hydrostatic strain; ``0`` allows general strain.

For box optimization, convergence requires both the force criterion above and a pressure criterion.
With general strain, the largest absolute component of the virial tensor divided by the volume must be below :math:`10^{-4}` GPa.
With hydrostatic strain, the absolute hydrostatic pressure must be below :math:`10^{-4}` GPa.
These minimization pressures contain only the virial contribution.

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
To write the optimized structure, follow minimization with a single NVE step at zero time step::

  minimize fire 1.0e-5 1000 1

  ensemble    nve
  time_step   0
  dump_xyz    1 relaxed.xyz
  run         1

This writes the structure without advancing the atomic positions.
Before a subsequent MD run, restore a suitable positive :ref:`time step <kw_time_step>` and specify the :ref:`ensemble <kw_ensemble>` and output commands for that run.

Caveats
-------

* This keyword should occur after the :ref:`potential keyword <kw_potential>`.

For magnetic moments, optionally coupled to atoms and cell strain, use
:ref:`Spin FIRE <spin_fire_minimize>`.
