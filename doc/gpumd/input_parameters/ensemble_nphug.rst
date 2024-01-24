.. _nphug:
.. _kw_ensemble_nphug:
.. index::
   single: nphug (keyword in run.in)
   single: NPHug integrator

:attr:`ensemble` (NPHug)
========================

This keyword sets up a Hugoniot Thermostat (:term:`NPHug`) integrator.

This integrator lets you specify a target stress, and adjust temperature to make the system converge to Hugoniot.

In this implementation, we use the barostat and thermostat of MTTK integrator, so it is very similiar to :ref:`mttk ensemble <kw_ensemble_mttk>`.

Please check [Ravelo2004]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble nphug <direction> <p_1> <p_2> tperiod <tau_temp> pperiod <tau_press>

The :attr:`<direction>` parameter can assume one or more of the following values: ``iso``, ``aniso``, ``tri``, ``x``, ``y``, ``z``.
Here, ``iso``, ``aniso``, and ``tri`` use hydrostatic pressure as the target pressure.
``iso`` updates the simulation cell isotropically.
``aniso`` updates the dimensions of the simulation cell along :math:`x`, :math:`y`, and :math:`z` independently.
``tri`` updatess all six degrees of freedom of the simulation cell.
Using ``x``, ``y``, ``z`` allows one to specify each stress component independently.

The parameters :attr:`<p_1>` and :attr:`<p_2>` specify the target pressure, and they should be equal.
Finally, the optional parameter :attr:`<tau_press>`, which defaults to ``1000``, determines the period of the barostat in units of the timestep.
It determines how strongly the system is coupled to the barostat.

Example
--------

.. code-block:: rst

    ensemble nphug x 300 300

This command performs an :term:`NPHug` simulation with a 300 GPa Hugoniot pressure in the x direction.
