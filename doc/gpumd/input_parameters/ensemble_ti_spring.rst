.. _kw_ensemble_ti_spring:

:attr:`ensemble` (TI_Spring)
============================

This keyword is used to set up a nonequilibrium thermodynamic integration integrator. Please check [Freitas2016]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble ti_spring temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time> spring <element_name> <spring_constant>

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: This parameter is optional, and defaults to ``100``. It determines the period of the thermostat in units of the timestep. It determines how strongly the system is coupled to the thermostat.
- :attr:`<equilibrium_time>`: The number timesteps to equilibrate the system.
- :attr:`<switch_time>`: The number timesteps to vary lambda from 0 to 1.
- :attr:`<element_name>` and :attr:`<spring_constant>`: Specify the spring constants of elements.

Please note that the spring constants should be placed at the end of the command.

Example
-------

.. code-block:: rst

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000 spring Si 6 O 5

This command switch lambda for 4000 timesteps, and equilibrate for 1000 timesteps. The spring constant is 6 eV/A^2 for Si and 5 eV/A^2 for O.

Output file
-----------

This command will produce a csv file. The columns are lambda, dlambda, potential energy and spring energy (eV/atom).
