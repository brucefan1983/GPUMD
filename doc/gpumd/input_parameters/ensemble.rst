.. _kw_ensemble:
.. index::
   single: ensemble (keyword in run.in)

:attr:`ensemble` (overview)
===========================

This keyword is used to set up an integration method (an integrator).
There are different categories of methods accessible via this keyword, which are described on the following pages:

* :ref:`"standard" ensembles <kw_ensemble_standard>`
* :ref:`MTTK integrators <kw_ensemble_mttk>`
* :ref:`integrators for thermal conductivity simulations <kw_ensemble_heat>`
* :ref:`integrators for path integral molecular dynamics simulations <kw_ensemble_pimd>`
* :ref:`MSST integrator for simulating compressive shock wave <kw_ensemble_msst>`
* :ref:`NPHug integrator for simulating compressive shock wave <kw_ensemble_nphug>`
* :ref:`piston integrator for simulating compressive shock wave <kw_ensemble_piston>`


.. _choice_of_parameters:

Units and suggested parameters
------------------------------

The units of temperature and pressure for this keyword are K and GPa, respectively. 

The temperature coupling constant :attr:`<T_coup>` means :math:`\tau_T/\Delta t`, where :math:`\tau_T` is the relaxation time of the thermostat and :math:`\Delta t` is the time step for integration.
We require :math:`\tau_T/\Delta t \geq 1` and a good choice is :math:`\tau_T/\Delta t \approx 100`.

When :math:`\tau_T/\Delta t > 100000`, the Berendsen thermostat in :attr:`npt_ber` will be completely ignored, leaving only the Berendsen barostat. In this case, the NPT ensemble reduces to the NPH ensemble that can be useful for melting point calculations using the two-phase method. We have not yet achieved a similar NPH ensemble using the :ref:`stochastic cell rescaling method <stochastic_cell_rescaling>`.

The pressure coupling constant :attr:`<p_coup>` means :math:`\tau_p/\Delta t`, where :math:`\tau_p` is the relaxation time of the barostat and :math:`\Delta t` is the time step for integration.
We require :math:`\tau_p/\Delta t \geq 1` and a good choice is :math:`\tau_p/\Delta t \approx 1000`.

The elastic constants are in units of GPa.


Caveats
-------
One should use one and only one instance of this keyword for each :ref:`run keyword <kw_run>`.
