.. _kw_ensemble_ti_spring:

:attr:`ensemble` (TI_Spring)
============================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator. See [Freitas2016]_ for more details.

Syntax
------

::

    ensemble ti_spring temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time> press <pressure> spring <element_name> <spring_constant>

Parameters
----------

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: Thermostat coupling period in time steps; optional, with a default of ``100``.
- :attr:`<equilibrium_time>`: The number of time steps to equilibrate the system.
- :attr:`<switch_time>`: The number of time steps to vary lambda from 0 to 1.
- :attr:`<element_name>` and :attr:`<spring_constant>`: Specify the spring constants of elements.
- :attr:`<pressure>`: Although this is an NVT ensemble, you can assign a pressure value to help GPUMD compute the Gibbs free energy. It does not affect the simulation process.

Please note that the spring constants should be placed at the end of the command. If the ``spring`` keyword is omitted, the spring constants are estimated from the atomic MSD during the initial equilibration, which requires ``tequil`` greater than zero.
Specify both ``tequil`` and ``tswitch``, or omit both.
``tequil`` must be a non-negative integer and ``tswitch`` a positive integer.
When omitted, they are set to ``floor(0.1 * N)`` and ``floor(0.4 * N)``, respectively, where ``N`` is the number of run steps; the resulting ``tswitch`` must still be positive.
The run must contain at least ``2 * (tequil + tswitch)`` steps.

Output
------

This command produces ``ti_spring.csv``. The columns are lambda, dlambda, potential energy and spring energy (eV/atom).

It also produces ``ti_spring.yaml``, which contains the free energies in eV/atom, temperature ``T`` in K, pressure ``P`` in eV/Å³, volume ``V`` in Å³/atom, and reference-system information. The ``press`` input is in GPa; use the pressure corresponding to the equilibrated volume when calculating Gibbs free energy.

Examples
--------

.. code-block:: text

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000 spring Si 6 O 5

Each switching stage lasts 4000 time steps and each equilibration stage lasts 1000 time steps. The spring constant is 6 eV/Å² for Si and 5 eV/Å² for O.

.. code-block:: text

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000

This command calculates spring constants automatically.

Notes
-----

None.
