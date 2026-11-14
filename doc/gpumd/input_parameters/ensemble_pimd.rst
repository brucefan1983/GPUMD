.. _kw_ensemble_pimd:

:attr:`ensemble` (PIMD)
=======================

The :attr:`ensemble` keyword is used to set up an integration method (an integrator).
The :attr:`pimd` and :attr:`pimd_scr` integrators incorporate nuclear quantum effects in equilibrium properties using path integral molecular dynamics (:term:`PIMD`).
The :attr:`rpmd` and :attr:`trpmd` integrators provide approximate real-time quantum dynamics [Craig2004]_ [Rossi2014]_.

Syntax
------

::

  ensemble pimd <num_beads> <T_1> <T_2> <T_coup> [<pressure_control_parameters>] [eco <omega_max>]
  ensemble pimd_scr <num_beads> <T_1> <T_2> <T_coup> <pressure_control_parameters> [eco <omega_max>]
  ensemble rpmd <num_beads>
  ensemble trpmd <num_beads>

Parameters
----------

``num_beads`` is the number of beads in the ring polymer. It must be even and between 2 and 128.

``T_1`` and ``T_2`` are the initial and final target temperatures in K and must be positive.
The target changes linearly during the run; use equal values for a constant temperature.
``T_coup`` is the thermostat relaxation time in time steps and must be at least 1.

``pimd`` applies a Langevin thermostat to both the internal and centroid modes [Ceriotti2010]_.
Without pressure parameters, the simulation cell is fixed.
Adding pressure parameters enables a Berendsen barostat.
``pimd_scr`` requires pressure parameters and uses stochastic cell rescaling instead.
Both accept the same three :ref:`pressure-control parameter forms <standard_pressure_parameters>` as the standard NPT integrators: isotropic, independent box lengths, and triclinic.
Pressures and elastic moduli are in GPa; ``p_coup`` is in time steps.

``rpmd`` evolves the ring polymer without a thermostat.
``trpmd`` applies a Langevin thermostat only to the internal modes, leaving the centroid mode unthermostatted.

Appending ``eco <omega_max>`` to ``pimd`` or ``pimd_scr`` selects economised path-integral internal-mode frequencies [Zeng2026]_.
``omega_max`` is a positive maximum physical vibrational wavenumber in cm\ :sup:`-1`.
The dimensionless fitting range is :math:`x_{\max}=hc\omega_{\max}/(k_{\rm B}T)` and follows the target temperature during a ramp.
Without ``eco``, GPUMD uses the original Trotter internal-mode frequencies.
``rpmd`` and ``trpmd`` always use the original Trotter frequencies.

Output
------

None.

Examples
--------

Run PIMD with 32 beads at a fixed target temperature of 300 K::

  ensemble pimd 32 300 300 100

Use isotropic stochastic cell rescaling at 0 GPa, with an estimated modulus of 100 GPa::

  ensemble pimd_scr 32 300 300 100 0 100 1000

Enable Eco frequencies at fixed cell with a maximum wavenumber of 4000 cm\ :sup:`-1`::

  ensemble pimd 32 300 300 100 eco 4000

Notes
-----

Initialize the ring polymer with a ``pimd`` or ``pimd_scr`` run before using ``rpmd`` or ``trpmd``.
The number of beads cannot change between runs.
