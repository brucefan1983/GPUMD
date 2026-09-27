.. _replica_temperatures:
.. index::
   single: gpumd_replica input files; temperatures

Temperature file
================

REMD can read a temperature ladder using ``temp <filename>`` in :ref:`replica_remd`.
The file must contain exactly one positive temperature in K per replica, in strictly increasing order.
Values are separated by whitespace; ``#`` introduces a comment.
No header or initial temperature count is required.

Example
-------

For four replicas::

  # Temperatures in K
  300
  340
  390
  450

Use ``ensemble nvt_bdp 300 450 100`` with this ladder.
