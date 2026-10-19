.. _replica_output_files:
.. index::
   single: gpumd_replica output files

Output files
============

Diagnostics are written automatically; trajectories require :ref:`replica_kw_dump_xyz`.
Fresh runs overwrite diagnostics and trajectories; resumed runs append.
Temperature ladders, ladder reports, and restart files are overwritten.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - File
     - Description
   * - :ref:`remd_temps.in <replica_remd_temps>`
     - Temperature ladder
   * - :ref:`remd_exchange.out <replica_remd_exchange>`
     - Exchange attempts
   * - :ref:`remd_temperature_status.out <replica_remd_status>`
     - Replica-to-temperature mapping
   * - :ref:`remd_thermo.out <replica_remd_thermo>`
     - Target temperatures, energies, and volumes
   * - :ref:`remd_ladder.out <replica_remd_ladder>`
     - Acceptance statistics
   * - :ref:`prd_events.out <replica_prd_events>`
     - Transitions and physical clock
   * - :ref:`Trajectory files <replica_trajectory>`
     - Atomic configurations
   * - :ref:`Restart files <replica_restart>`
     - State for continuing a run

.. toctree::
   :maxdepth: 1
   :caption: Contents

   remd_temps
   remd_exchange
   remd_status
   remd_thermo
   remd_ladder
   prd_events
   trajectory
   restart
