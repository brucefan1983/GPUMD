.. _nphug:
.. _kw_ensemble_nphug:
.. index::
   single: nphug (keyword in run.in)
   single: NPHug integrator

``ensemble`` (NPHug)
========================

This keyword sets up a Hugoniot thermostat integrator.

This integrator lets you specify a target stress, and adjust temperature to make the system converge to Hugoniot.

In this implementation, we use the barostat and thermostat of :term:`MTTK` integrator, so it is very similar to :ref:`mttk ensemble <kw_ensemble_mttk>`.

Please check [Ravelo2004]_ for more details.

Syntax
------

The parameters can be specified as follows::

    ensemble nphug <direction> <p_1> <p_2> tperiod <tau_temp> pperiod <tau_press> p0 <p0> v0 <v0> e0 <e0>

Parameters
----------

The ``<direction>`` parameter can assume one or more of the following values: ``iso``, ``aniso``, ``tri``, ``x``, ``y``, ``z``.
Here, ``iso``, ``aniso``, and ``tri`` use hydrostatic pressure as the target pressure.
``iso`` updates the simulation cell isotropically.
``aniso`` updates the dimensions of the simulation cell along :math:`x`, :math:`y`, and :math:`z` independently.
``tri`` updates all six degrees of freedom of the simulation cell.
Using ``x``, ``y``, ``z`` allows one to specify each stress component independently.

The parameters ``<p_1>`` and ``<p_2>`` specify the target pressure, and they should be equal.
Finally, the optional parameter ``<tau_press>``, which defaults to ``1000``, determines the period of the barostat in units of the timestep.
It determines how strongly the system is coupled to the barostat.

If keywords ``<p0>``, ``<v0>`` or ``<e0>`` are not supplied, these quantities will be calculated on the first step.
In most cases, you don't need to specify these quantities.

Output
------

None.

Examples
--------

.. code-block:: rst

    ensemble nphug x 300 300

This command performs a simulation using a Hugoniot thermostat with a 300 GPa Hugoniot pressure in the :math:`x` direction.

Notes
-----

None.
