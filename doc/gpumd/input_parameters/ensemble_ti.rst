.. _kw_ensemble_ti:

:attr:`ensemble` (TI)
=====================

This keyword is used to set up a equilibrium thermodynamic integration integrator. It is for testing purpose and its only differece from :ref:`ti_spring keyword <kw_ensemble_ti_spring>` is that the lambda value is fixed instead of changing.

Syntax
------

The parameters can be specified as follows::

    ensemble ti lambda <lambda> temp <temperature> tperiod <tau_temperature> spring <element_name> <spring_constant>

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: This parameter is optional, and defaults to ``100``. It determines the period of the thermostat in units of the timestep. It determines how strongly the system is coupled to the thermostat.
- :attr:`<element_name>` and :attr:`<spring_constant>`: Specify the spring constants of elements.
- :attr:`<lambda>`: The lambda value of the simulation.

Example
-------

.. code-block:: rst

    ensemble ti_spring temp 300 lambda 0.3 spring Si 6 O 5

This command uses lambda value 0.3 (30% spring force and 70% original force field). The spring constant is 6 eV/A^2 for Si and 5 eV/A^2 for O.