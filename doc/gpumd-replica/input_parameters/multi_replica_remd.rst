.. _replica_remd:
.. index::
   single: multi_replica remd (keyword in gpumd_replica run.in)

:attr:`!multi_replica remd`
===========================

This command selects temperature replica-exchange molecular dynamics (REMD) [Sugita1999]_.

Syntax
------

::

  multi_replica remd replicas <number> [<per_gpu>] exchange <interval> temp <T_min> <T_max> [spacing <type>] [equilibrate <steps>] [resume <prefix>]

* ``number``: at least two replicas; see :ref:`replica_kw_multi_replica` for GPU allocation.
* ``exchange``: positive number of MD steps between exchange rounds.
* ``temp``: positive ladder endpoints in K, with ``T_min < T_max``.
  Alternatively, use ``temp <filename>`` to read a :ref:`temperature file <replica_temperatures>`.
* ``spacing``: ``geometric`` (default) or ``linear`` for generated ladders.
* ``equilibrate``: independent NVT steps before exchanges begin; default 0.
  These steps do not count toward ``run``.
* ``resume``: :ref:`restart prefix <replica_restart>`; cannot be combined with positive ``equilibrate``.

Example
-------

::

  multi_replica remd replicas 8 exchange 1000 temp 300 450 equilibrate 200000

A complete input is shown in :ref:`replica_run_in`.

Caveats
-------

* Exchanges alternate between adjacent temperature-label pairs, starting with ``(0, 1), (2, 3), ...``.
  For two replicas, the only pair is attempted every round.
* Accepted exchanges swap temperature assignments and rescale velocities; coordinates remain with their replica.
  Collect fixed-temperature statistics by temperature label, not replica index.
