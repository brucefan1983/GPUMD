.. _kw_ensemble_ti:

:attr:`ensemble` (TI)
=====================

Run thermodynamic integration at a fixed coupling parameter ``lambda``.
The reference system is the same as for :ref:`ti_spring <kw_ensemble_ti_spring>`, but the coupling does not change during the run.

Syntax
------

::

    ensemble ti lambda <lambda> temp <temperature> tperiod <tau_temperature> spring <element_name> <spring_constant>

Parameters
----------

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: Thermostat coupling period in time steps; optional, with a default of ``100``.
- :attr:`<element_name>` and :attr:`<spring_constant>`: Specify the spring constants of elements.
- :attr:`<lambda>`: The lambda value of the simulation.

Examples
--------

.. code-block:: text

    ensemble ti lambda 0.3 temp 300 spring Si 6 O 5

This command uses lambda value 0.3 (30% spring force and 70% original force field). The spring constant is 6 eV/Å² for Si and 5 eV/Å² for O.
