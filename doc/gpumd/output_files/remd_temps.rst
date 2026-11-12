.. _replica_remd_temps:
.. index::
   single: gpumd output files; remd_temps.in

``remd_temps.in``
=================

Records the temperature ladder in increasing order, with one temperature in K per line.
Written at the start of each run, including resumed runs.
It can be reused as a :ref:`temperature input file <replica_temperatures>`.
