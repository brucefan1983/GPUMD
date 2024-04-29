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
- :attr:`<pressure>`: Although this is an NVT ensemble, you can assign a pressure value to help GPUMD compute the Gibbs free energy. It does not affect the simlation process.

Please note that the spring constants should be placed at the end of the command. If there are no ``spring`` keyword, the spring constants will be computed automatically through the MSD of atoms.
If you do not specify :attr:`<equilibrium_time>` and :attr:`<switch_time>`, they will be automatically set in a 1:4 ratio.

Example
-------

.. code-block:: rst

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000 spring Si 6 O 5

This command switch lambda for 4000 timesteps, and equilibrate for 1000 timesteps. The spring constant is 6 eV/A^2 for Si and 5 eV/A^2 for O.

.. code-block:: rst

    ensemble ti_spring temp 300 tequil 1000 tswitch 4000

This command calculate spring constants automatically.

Output file
-----------

This command will produce a csv file. The columns are lambda, dlambda, potential energy and spring energy (eV/atom).

This command will also produce a yaml file, which contains Gibbs free energy and other information.

Additionally, important information will be displayed on the screen during the simulation process, so it is recommended to carefully review and take note of these details.