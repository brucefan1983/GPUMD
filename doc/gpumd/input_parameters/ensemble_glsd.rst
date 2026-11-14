.. _kw_ensemble_glsd:
.. _kw_nve_glsd:
.. _kw_nvt_glsd:
.. _kw_npt_glsd:
.. index::
   single: nve_glsd (keyword in run.in)
   single: nvt_glsd (keyword in run.in)
   single: npt_glsd (keyword in run.in)

:attr:`ensemble` (GLSD)
=======================

Evolve both the directions and magnitudes of magnetic moments using generalized
Langevin spin dynamics (GLSD). A Spin potential and ``spin:R:3`` magnetic moments
in ``model.xyz`` are required. For fixed-magnitude dynamics, use
:ref:`SIB <kw_ensemble_sib>`.

Syntax
------

::

    ensemble nve_glsd damping <value> [gamma <value>] [stemp <value>] [seed <value>] [midpoint_tol <value>] [midpoint_max_iter <N>]
    ensemble nvt_glsd <T_1> <T_2> <T_coup> damping <value> [lattice on|off] [gamma <value>] [stemp <value>] [seed <value>] [midpoint_tol <value>] [midpoint_max_iter <N>]
    ensemble npt_glsd temp <T_1> <T_2> <pressure_terms> [tperiod <tau_temp>] [pperiod <tau_press>] damping <value> [gamma <value>] [stemp <value>] [seed <value>] [midpoint_tol <value>] [midpoint_max_iter <N>]

Parameters
----------

``nve_glsd`` fixes atomic positions and velocities. ``nvt_glsd`` optionally
couples the spins to moving atoms at fixed cell. ``npt_glsd`` evolves spins,
atoms and the cell.

``T_1``, ``T_2``, ``T_coup``
  NVT initial and final target temperatures in K, and thermostat coupling time
  in time steps, as for :ref:`NHC <kw_ensemble_standard>`. Temperatures must be
  positive and ``T_coup`` must be at least 1. The target changes linearly during
  each run. ``T_coup`` is required even with ``lattice off``.

``pressure_terms``, ``tperiod``, ``pperiod``
  NPT pressure controls and bath periods have the syntax, defaults and ranges
  of :ref:`MTTK <kw_ensemble_mttk>`. Pressures are in GPa and periods are in time
  steps. All GLSD options must follow these MTTK parameters.

``damping``
  Required dimensionless dissipation parameter; finite and non-negative, with
  no default. Zero disables damping and noise. It differs from SIB's ``alpha``.
  The corresponding Cartesian mobility is
  :math:`\lambda=\mathtt{damping}\,\mathtt{gamma}^2\hbar`, in
  :math:`\mu_B^2/(\mathrm{eV}\,\mathrm{ps})`.

``gamma``
  Finite positive numerical precession coefficient in the
  :math:`(\mathrm{eV}\,\mathrm{ps})^{-1}` convention. Default:
  :math:`2/\hbar\simeq3038.54`. Changing it at fixed ``damping`` also changes
  the mobility and noise amplitude.

``stemp``
  Spin-bath temperature: ``-1`` disables noise while retaining damping;
  ``0`` follows the ensemble target; a finite positive value fixes the spin
  temperature in K. Default: ``-1`` for ``nve_glsd``, ``0`` otherwise.
  ``nve_glsd`` does not accept ``stemp 0``.

``lattice``
  Available only for ``nvt_glsd``; default ``on``. ``off`` fixes atomic
  positions and velocities. It does not disable spin dynamics.

``seed``
  Positive integer random seed; default ``12345``. A new run initializes a
  fresh spin-noise stream. See :ref:`spin_restart` for restart behavior.

``midpoint_tol``
  Finite positive relative convergence tolerance; default ``1e-7``. A midpoint
  correction is converged when every component changes by at most
  ``midpoint_tol * (1 + abs(S_mid))``, using numerical spin values in muB.
  Smaller values require tighter self-consistency and may reach the model's
  numerical precision limit.

``midpoint_max_iter``
  Positive integer maximum number of midpoint corrections per step; default
  ``4``. Corrections stop early on convergence. At the limit, the last finite
  result is accepted even if the tolerance has not been reached.

Optional GLSD keywords cannot be repeated. ``alpha`` and ``mobility`` are not
accepted by these commands.

Output
------

The log prints the selected parameters at the start of each run. At the end,
it reports the mean and maximum correction counts and the number of steps
accepted at the iteration limit without convergence, for example::

    GLSD midpoint iterations per step: mean 3.850, max 4.
    GLSD midpoint steps accepted at iteration cap before convergence: 12 of 1000.

Here, 12 steps were accepted without satisfying ``midpoint_tol``. Reaching the
cap is not convergence. If this count is nonzero, reduce :ref:`time_step
<kw_time_step>` or increase ``midpoint_max_iter`` and check that the observables
of interest are insensitive to these choices. If tighter tolerance no longer
improves the result, check the potential's numerical precision. A nonfinite
predictor or correction stops the run with an error; reduce the timestep and
check the model and fields.

Use ``dump_thermo`` for thermodynamic output and ``dump_xyz`` with ``spin`` and
``mforce`` for magnetic moments and magnetic forces.

Examples
--------

Deterministic relaxation at fixed atomic positions::

    ensemble nve_glsd damping 0.01 stemp -1 midpoint_max_iter 20
    time_step 0.1
    run 1000

A 300 K spin bath with fixed atomic positions::

    ensemble nvt_glsd 300 300 100 lattice off damping 0.01 seed 913
    time_step 0.1
    run 1000

Coupled spin and atomic motion at fixed cell::

    ensemble nvt_glsd 300 300 100 lattice on damping 0.01 midpoint_max_iter 20
    time_step 0.1
    dump_thermo 10
    dump_xyz 100 spin.xyz spin mforce
    run 1000

Coupled motion with isotropic zero-pressure control::

    ensemble npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 damping 0.01
    time_step 0.1
    run 1000

These examples assume the Spin potential and initial configuration have already
been supplied. ``damping 0.01`` and the timestep are illustrative values, not
material-independent recommendations.

Notes
-----

* ``nve_glsd`` names the fixed-lattice driver. With positive damping, spin energy
  can dissipate even when noise is disabled; the name does not imply energy
  conservation. With noise enabled, energy exchanges with the spin bath.
* The potential must describe the range of magnetic-moment magnitudes visited.
  Zero moments can evolve when driven by a field or noise.
* Timestep and midpoint convergence must be checked for the chosen model and
  conditions. A finite trajectory alone does not establish accurate sampling.
* Specify an ensemble before every ``run``. Physical-state restarts do not
  continue the previous noise or thermostat trajectory exactly; see
  :ref:`spin_restart`.
