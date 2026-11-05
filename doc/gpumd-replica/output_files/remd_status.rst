.. _replica_remd_status:
.. index::
   single: gpumd_replica output files; remd_status

``remd_temperature_status.out``
===============================

Each row contains the production step followed by one temperature label per replica, in replica-index order::

  step label_of_replica_0 label_of_replica_1 ...

Labels start at zero. The mapping is recorded after any exchange at that step.
Rows are written initially, at exchange and trajectory-output steps, and at the final step.
