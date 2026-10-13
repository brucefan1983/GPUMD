.. _replica_run_in:
.. index::
   single: gpumd_replica input files; run.in

Simulation protocol (``run.in``)
================================

This file defines the replica simulation using one command per line.
Blank lines and text following ``#`` are ignored.
Only the commands listed under :ref:`replica_input_parameters` are supported.
A single ``run`` command must appear last.

Example
-------

An eight-replica REMD input is::

  potential nep.txt
  time_step 0.5
  ensemble nvt_bdp 300 450 100
  multi_replica remd replicas 8 exchange 1000 temp 300 450 equilibrate 200000
  dump_xyz -1 0 5000 trajectory.xyz velocity
  run 2000000

This uses a geometric temperature ladder from 300 to 450 K, with 200000 equilibration steps followed by 2000000 production steps per replica.
