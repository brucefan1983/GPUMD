.. _kw_ensemble_pimd:

:attr:`ensemble` (PIMD)
=======================

The :attr:`ensemble` keyword is used to set up an integration method (an integrator).
The integrators described on this page enable one to carry out path integral molecular dynamics (:term:`PIMD`) simulations and thereby to incorporate quantum dynamical effects.


Syntax
------

:attr:`pimd`
^^^^^^^^^^^^
If the first parameter is :attr:`pimd`, it means that the current run will use path-integral molecular dynamics (:term:`PIMD`).

It can be used in the following ways::

    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> 
    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> {<pressure_control_parameters>}
    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> {<pressure_control_parameters>} eco <omega_max>

In all forms, :attr:`num_beads` is the number of beads in the ring polymer, which should be a positive even integer no larger than 128.
The first case is similar to the NVT ensemble with :attr:`nvt_lan` as the Langevin thermostat is used for both the internal and the centroid modes [Ceriotti2010]_. 
The second case is similar to the NPT ensemble with :attr:`npt_ber`, where a Berendsen barostat is added compared to the first case.
Note that :attr:`pimd` or :attr:`pimd_scr` (that is, not :attr:`rpmd` or :attr:`trpmd` described below) must be the first run that requires to set :attr:`num_beads` and one cannot change :attr:`num_beads` from run to run.

Appending ``eco <omega_max>`` selects the economised path-integral internal-mode frequencies [Zeng2026]_.
Here :attr:`omega_max` is the highest physical vibrational wavenumber to be reproduced, in units of cm\ :sup:`-1`.
The dimensionless fitting range is recalculated from the instantaneous target temperature as :math:`x_{\max}=hc\omega_{\max}/(k_{\rm B}T)`, so Eco frequencies also follow a temperature ramp from :attr:`T_1` to :attr:`T_2`.
If the ``eco`` option is omitted, GPUMD uses the original Trotter internal-mode frequencies.
The Eco option is available for :attr:`pimd` and :attr:`pimd_scr` only; :attr:`rpmd` and :attr:`trpmd` use the original Trotter frequencies.

:attr:`pimd_scr`
^^^^^^^^^^^^^^^^^
If the first parameter is :attr:`pimd_scr`, it means that the current run will use path-integral molecular dynamics (:term:`PIMD`) with stochastic cell rescaling (:attr:`npt_scr`) for pressure control.

It can be used in the following ways::

    ensemble pimd_scr <num_beads> <T_1> <T_2> <T_coup> {<pressure_control_parameters>}
    ensemble pimd_scr <num_beads> <T_1> <T_2> <T_coup> {<pressure_control_parameters>} eco <omega_max>

The pressure-control parameters have the same meaning and support the same isotropic, orthogonal, and triclinic forms as in the pressure-controlled :attr:`pimd` case.
The difference is that :attr:`pimd` uses the Berendsen barostat, whereas :attr:`pimd_scr` uses stochastic cell rescaling.
The ``eco`` option has the same meaning as described above for :attr:`pimd`.

:attr:`rpmd`
^^^^^^^^^^^^
If the first parameter is :attr:`rpmd`, it means that the current run will use ring-polymer molecular dynamics (:term:`RPMD`) [Craig2004]_.

It can be used as follows::

    ensemble rpmd <num_beads> 

This can be understood as the NVE version of :term:`PIMD`, where no thermostat is applied.

:attr:`trpmd`
^^^^^^^^^^^^^
If the first parameter is :attr:`trpmd`, it means that the current run will use thermostatted ring-polymer molecular dynamics (:term:`TRPMD`) [Rossi2014]_.

It can be used as follows::

    ensemble trpmd <num_beads> 

This is similar to :term:`RPMD`, but the Langevin thermosat is applied to the internal modes.
