.. _kw_ensemble_ti_spring:

:attr:`ensemble` (TI_Spring)
============================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator. Please check [Freitas2016]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble ti_spring temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time> press <pressure> spring <element_name> <spring_constant>

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: This parameter is optional, and defaults to ``100``. It determines the period of the thermostat in units of the timestep. It determines how strongly the system is coupled to the thermostat.
- :attr:`<equilibrium_time>`: The number timesteps to equilibrate the system.
- :attr:`<switch_time>`: The number timesteps to vary lambda from 0 to 1.
- :attr:`<element_name>` and :attr:`<spring_constant>`: Specify the spring constants of elements.
- :attr:`<pressure>`: Although this is an NVT ensemble, you can assign a pressure value to help GPUMD compute the Gibbs free energy. It does not affect the simulation process.

Please note that the spring constants should be placed at the end of the command. If the ``spring`` keyword is omitted, the spring constants are estimated from the atomic MSD during the initial equilibration, which requires ``tequil`` greater than zero.
Specify both ``tequil`` and ``tswitch``, or omit both.
``tequil`` must be a non-negative integer and ``tswitch`` a positive integer.
When omitted, they are set to ``floor(0.1 * N)`` and ``floor(0.4 * N)``, respectively, where ``N`` is the number of run steps; the resulting ``tswitch`` must still be positive.
The run must contain at least ``2 * (tequil + tswitch)`` steps.

Example
-------

.. code-block:: rst

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000 spring Si 6 O 5

This command switch lambda for 4000 timesteps, and equilibrate for 1000 timesteps. The spring constant is 6 eV/A^2 for Si and 5 eV/A^2 for O.

.. code-block:: rst

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000

This command calculates spring constants automatically.

Output file
-----------

This command produces ``ti_spring.csv``. The columns are lambda, dlambda, potential energy and spring energy (eV/atom).

It also produces ``ti_spring.yaml``, which contains the free energies in eV/atom, temperature ``T`` in K, pressure ``P`` in eV/Å³, volume ``V`` in Å³/atom, and reference-system information. The ``press`` input is in GPa; use the pressure corresponding to the equilibrated volume when calculating Gibbs free energy.

Additionally, important information will be displayed on the screen during the simulation process, so it is recommended to carefully review and take note of these details.
