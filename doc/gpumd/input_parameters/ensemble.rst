.. _kw_ensemble:
.. index::
   single: ensemble (keyword in run.in)

:attr:`ensemble` (overview)
===========================

This keyword selects the integration method for the next :ref:`run <kw_run>`.
For conventional classical MD, the main choices are:

* ``nve``: dynamics without a thermostat or barostat.
* ``nvt_*``: temperature control at fixed cell.
* ``npt_*``: temperature and pressure control, allowing the cell to change.
* ``nph_mttk``: pressure control without a thermostat.

The available methods, including specialized integrators, are described on the following pages:

* :ref:`"standard" ensembles <kw_ensemble_standard>`
* :ref:`MTTK integrators <kw_ensemble_mttk>`
* :ref:`TSPIN integrators for Spin potentials <kw_ensemble_tspin>`
* :ref:`SIB integrators for Spin potentials <kw_ensemble_sib>`
* :ref:`GLSD variable-magnitude dynamics <kw_ensemble_glsd>`
* :ref:`local heat-source and heat-sink thermostats <kw_ensemble_heat>`
* :ref:`two-temperature-model integrators <kw_ensemble_ttm>`
* :ref:`quantum thermal bath integrators <kw_ensemble_qtb>`
* :ref:`integrators for path integral molecular dynamics simulations <kw_ensemble_pimd>`
* :ref:`MSST integrator for simulating compressive shock wave <kw_ensemble_msst>`
* :ref:`NPHug integrator for simulating compressive shock wave <kw_ensemble_nphug>`
* :ref:`NEMD for simulating compressive shock wave <kw_ensemble_shock_nemd>`
* Thermodynamic integration and switching: :ref:`TI <kw_ensemble_ti>`, :ref:`TI_Spring <kw_ensemble_ti_spring>`, :ref:`TI_AS <kw_ensemble_ti_as>`, :ref:`TI_RS <kw_ensemble_ti_rs>`, :ref:`TI_Liquid <kw_ensemble_ti_liquid>`, and :ref:`TI_NEP <kw_ensemble_ti_nep>`


.. _choice_of_parameters:

Units and suggested parameters
------------------------------

The units of temperature and pressure for this keyword are K and GPa, respectively.
These are thermostat and barostat targets; they do not set the initial atomic velocities, which are determined by ``model.xyz`` or the :ref:`velocity keyword <kw_velocity>`.
Use equal initial and final targets to keep the target constant during a run.

The coupling parameters below are used by the :ref:`standard integrators <kw_ensemble_standard>` and :ref:`local heat baths <kw_ensemble_heat>` where applicable.
For methods with different parameter names or requirements, follow their individual pages.

The temperature coupling constant :attr:`<T_coup>` means :math:`\tau_T/\Delta t`, where :math:`\tau_T` is the relaxation time of the thermostat and :math:`\Delta t` is the time step for integration.
We require :math:`\tau_T/\Delta t \geq 1` and a good choice is :math:`\tau_T/\Delta t \approx 100`.

When :math:`\tau_T/\Delta t > 100000`, the Berendsen thermostat in :attr:`npt_ber` will be completely ignored, leaving only the Berendsen barostat. In this case, the NPT ensemble reduces to the NPH ensemble that can be useful for melting point calculations using the two-phase method. We have not yet achieved a similar NPH ensemble using the :ref:`stochastic cell rescaling method <stochastic_cell_rescaling>`.

The pressure coupling constant :attr:`<p_coup>` means :math:`\tau_p/\Delta t`, where :math:`\tau_p` is the relaxation time of the barostat and :math:`\Delta t` is the time step for integration.
We require :math:`\tau_p/\Delta t \geq 1` and a good choice is :math:`\tau_p/\Delta t \approx 1000`.

The elastic constants are in units of GPa.


Caveats
-------
Specify exactly one ``ensemble`` command before each :ref:`run <kw_run>`.
The integrator is initialized for that run, and any temperature or pressure ramp is defined over that run.
