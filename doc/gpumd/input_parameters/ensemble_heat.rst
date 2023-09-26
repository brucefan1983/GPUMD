.. _kw_ensemble_heat:

:attr:`ensemble` (thermal transport)
====================================

The :attr:`ensemble` keyword is used to set up an integration method (an integrator).
The integrators described on this page provide methods for studying thermal transport via classical :term:`MD` simulations.
Background information pertaining to the methods referred to on this page can be found :ref:`here <green_kubo_method>`.


Syntax
------

:attr:`heat_nhc`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_nhc`, it means heating a source region and simultaneously cooling a sink region using local :ref:`Nose-Hoover chain thermostats <nose_hoover_chain_thermostat>`.
The full command is::

  ensemble heat_nhc <T> <T_coup> <delta_T> <label_source> <label_sink>

The target temperatures in the source region with label :attr:`<label_source>` and the sink region with label :attr:`<label_sink>` are :attr:`<T>+<delta_T>` and :attr:`<T>-<delta_T>`, respectively.
Therefore, the temperature difference between the two regions is two times :attr:`<delta_T>`.
In the command above, the parameter :attr:`<T_coup>` has the same meaning as in the case of :attr:`nvt_nhc`.
Both :attr:`<label_source>` and :attr:`<label_sink>` refer to the 0-th grouping method.

:attr:`heat_bdp`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_bdp`, it is similar to the case of :attr:`heat_nhc`, but using the :ref:`Bussi-Donadio-Parrinello method <bdp_thermostat>`.

:attr:`heat_lan`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_lan`, it is similar to the case of :attr:`heat_nhc`, but using the :ref:`Langevin method <langevin_thermostat>`.
