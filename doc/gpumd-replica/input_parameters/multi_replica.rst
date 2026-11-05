.. _replica_kw_multi_replica:
.. index::
   single: multi_replica (keyword in gpumd_replica run.in)

:attr:`!multi_replica`
======================

Syntax
------

::

  multi_replica <mode> replicas <number> [<per_gpu>] <options>

Select :ref:`remd <replica_remd>` or :ref:`prd <replica_prd>` and supply the corresponding options.
This command runs independent copies of the system; it does not create a spatial supercell.

``number`` is the total replica count.
The optional ``per_gpu`` is the capacity per GPU, defaulting to the replica count divided by the number of visible GPUs, rounded up.
Total capacity must accommodate all replicas.
Replicas are assigned round-robin and remain resident in GPU memory.

Example
-------

With four visible GPUs, this places two replicas on each GPU::

  multi_replica remd replicas 8 2 exchange 1000 temp 300 450
