.. _replica_remd_tp_output:

REMD-TP output files
====================

:ref:`multi_replica remd_tp <replica_remd_tp>` writes the following files.
Condition labels follow ``label = p_index * num_temperatures + t_index``.
Collect equilibrium statistics by condition label, because each replica moves between conditions.

* ``remd_tp_conditions.out``: label, temperature index, pressure index, temperature (K), pressure (GPa).
* ``remd_tp_exchange.out``: step, exchange direction (``T`` or ``P``), replica indices,
  old labels, temperatures, pressures, potential energies (eV), volumes (Angstrom cubed),
  log acceptance ratio, acceptance probability, and acceptance flag.
  Values describe the states before the exchange.
* ``remd_tp_status.out``: step followed by the current condition label of every replica.
* ``remd_tp_thermo.out``: step followed by label, potential energy, and volume for every replica.
  Rows describe the states after any exchange at that step and are written at advancement boundaries.
* ``remd_tp_pairs.out``: attempts, accepts, mean probability, and observed acceptance rate for each neighboring condition pair.

Exchange, status, and thermo files append on resume. Conditions and pair reports are overwritten.
Trajectory frames and :ref:`restart files <replica_restart>` include each replica's current box.
