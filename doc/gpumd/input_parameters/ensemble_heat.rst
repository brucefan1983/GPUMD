.. _kw_ensemble_heat:

:attr:`ensemble` (thermal transport)
====================================

The :attr:`ensemble` keyword is used to set up an integration method (an integrator).
The local source and sink thermostats described on this page are used for :ref:`nonequilibrium thermal transport <nemd>`.
For equilibrium and homogeneous nonequilibrium calculations, see :ref:`compute_hac <kw_compute_hac>` and :ref:`compute_hnemd <kw_compute_hnemd>`, respectively.
The two-temperature-model variants are described in :ref:`TTM integrators <kw_ensemble_ttm>`.


Syntax
------

:attr:`heat_nhc`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_nhc`, it means heating a source region and simultaneously cooling a sink region using local :ref:`Nose-Hoover chain thermostats <nose_hoover_chain_thermostat>`.
The full command is::

  ensemble heat_nhc <T> <T_coup> <delta_T> <label_source> <label_sink>

The target temperatures in the source region with label :attr:`<label_source>` and the sink region with label :attr:`<label_sink>` are :code:`<T> + <delta_T>` and :code:`<T> - <delta_T>`, respectively.
Therefore, the temperature difference between the two regions is two times :attr:`<delta_T>`.
In the command above, the parameter :attr:`<T_coup>` has the same meaning as in the case of :attr:`nvt_nhc`.
Both :attr:`<label_source>` and :attr:`<label_sink>` refer to the 0-th grouping method.

:attr:`heat_bdp`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_bdp`, it is similar to the case of :attr:`heat_nhc`, but using the :ref:`Bussi-Donadio-Parrinello method <bdp_thermostat>`.

:attr:`heat_lan`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_lan`, it is similar to the case of :attr:`heat_nhc`, but using the :ref:`Langevin method <langevin_thermostat>`.
The source and sink can be specified using static groups::

  ensemble heat_lan <T> <T_coup> <delta_T> <label_source> <label_sink>

Alternatively, they can be specified using spatial regions::

  ensemble heat_lan <T> <T_coup> <delta_T> {source_bounds} {sink_bounds}

Each brace-enclosed placeholder represents six numbers, entered without braces:
``a_min a_max b_min b_max c_min c_max``.
They are fractional-coordinate bounds along the three cell vectors.
Each pair must satisfy :math:`0 \leq s_\mathrm{min} < s_\mathrm{max} \leq 1`, and the source and sink regions must not overlap.
The lower bounds are included and the upper bounds are excluded.
Region membership is determined from the current atomic positions during integration, so atoms can enter and leave the heat baths.

The temperatures are :code:`<T> + <delta_T>` and :code:`<T> - <delta_T>` for the source and sink, respectively.
:attr:`<T_coup>` is the temperature coupling time in units of the time step.
For example::

  ensemble heat_lan 300 100 10 0 0.1 0 1 0 1 0.5 0.6 0 1 0 1

This places the 310 K source at :math:`0 \leq s_a < 0.1` and the 290 K sink at :math:`0.5 \leq s_a < 0.6`, spanning the full cell along the other two fractional coordinates.

:attr:`ttm`
^^^^^^^^^^^
If the first parameter is :attr:`ttm`, the run uses a two-temperature-model (TTM) integrator.
The selected atom group is coupled to an electron temperature grid.

The full command is::

  ensemble ttm <grouping_method> <group_id> <Ce> <rho_e> <kappa_e> <gamma_p> <gamma_s> <v_0> <nx> <ny> <nz> <T_e_init> [{optional_args}]

The meanings of the parameters and optional arguments are described in
:ref:`TTM integrators <kw_ensemble_ttm>`.

:attr:`heat_ttm`
^^^^^^^^^^^^^^^^
If the first parameter is :attr:`heat_ttm`, the run uses local source/sink Langevin thermostats together with the TTM electron grid.

The full command is::

  ensemble heat_ttm <T> <T_coup> <delta_T> <label_source> <label_sink> <grouping_method> <group_id> <Ce> <rho_e> <kappa_e> <gamma_p> <gamma_s> <v_0> <nx> <ny> <nz> <T_e_init> [{optional_args}]

The first five parameters,
:attr:`<T>`, :attr:`<T_coup>`, :attr:`<delta_T>`, :attr:`<label_source>`, and :attr:`<label_sink>`,
have the same meanings as in :attr:`heat_lan`.
The remaining parameters and optional arguments are described in
:ref:`TTM integrators <kw_ensemble_ttm>`.
