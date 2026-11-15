.. _kw_ensemble_heat:

:attr:`ensemble` (thermal transport)
====================================

Heat a source region and cool a sink region for :ref:`nonequilibrium thermal transport <nemd>`.
For coupling to an electron temperature grid, see :ref:`TTM integrators <kw_ensemble_ttm>`.

Syntax
------

::

    ensemble heat_nhc <T> <T_coup> <delta_T> <label_source> <label_sink>
    ensemble heat_bdp <T> <T_coup> <delta_T> <label_source> <label_sink>
    ensemble heat_lan <T> <T_coup> <delta_T> <label_source> <label_sink>
    ensemble heat_lan <T> <T_coup> <delta_T> {source_bounds} {sink_bounds}

Parameters
----------

* ``T`` is the mean of the two target temperatures, in K.
  The source and sink targets are ``T + delta_T`` and ``T - delta_T``, so their difference is ``2 * delta_T``.
* ``T_coup`` is the thermostat coupling time in time steps; see :ref:`units and suggested parameters <choice_of_parameters>`.
* ``label_source`` and ``label_sink`` select groups in grouping method 0.
* ``heat_nhc``, ``heat_bdp``, and ``heat_lan`` use :ref:`Nosé-Hoover chain <nose_hoover_chain_thermostat>`, :ref:`Bussi-Donadio-Parrinello <bdp_thermostat>`, and :ref:`Langevin <langevin_thermostat>` thermostats, respectively.

The last form of ``heat_lan`` selects spatial regions instead of fixed atom groups.
Each brace-enclosed placeholder represents six numbers, entered without braces:
``a_min a_max b_min b_max c_min c_max``.
They are fractional-coordinate bounds along the three cell vectors.
Each pair must satisfy :math:`0 \leq s_\mathrm{min} < s_\mathrm{max} \leq 1`, and the source and sink regions must not overlap.
The lower bounds are included and the upper bounds are excluded.
Region membership is determined from the current atomic positions during integration, so atoms can enter and leave the heat baths.

Output
------

None.

Examples
--------

Apply source and sink targets of 310 K and 290 K to groups 0 and 1::

    ensemble heat_lan 300 100 10 0 1

Apply Langevin heat baths to two spatial slabs::

  ensemble heat_lan 300 100 10 0 0.1 0 1 0 1 0.5 0.6 0 1 0 1

This places the 310 K source at :math:`0 \leq s_a < 0.1` and the 290 K sink at :math:`0.5 \leq s_a < 0.6`, spanning the full cell along the other two fractional coordinates.

Notes
-----

None.
