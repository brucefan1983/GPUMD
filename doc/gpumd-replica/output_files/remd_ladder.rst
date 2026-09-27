.. _replica_remd_ladder:
.. index::
   single: gpumd_replica output files; remd_ladder

``remd_ladder.out``
===================

Written at completion, with one row per adjacent temperature pair::

  temp_i temp_j temperature_i temperature_j attempts accepted acceptance_ratio average_probability meets_target

``temp_i`` and ``temp_j`` are zero-based labels; temperatures are in K.
``acceptance_ratio`` is accepted exchanges divided by attempts, and ``average_probability`` is the mean acceptance probability.
``meets_target`` is 1 for an attempted pair with acceptance ratio at least 0.2, otherwise 0.
This diagnostic does not adapt the ladder or establish convergence.
