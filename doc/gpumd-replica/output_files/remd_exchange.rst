.. _replica_remd_exchange:
.. index::
   single: gpumd_replica output files; remd_exchange

``remd_exchange.out``
=====================

One row is written per attempted pair, with columns::

  step replica_i replica_j temperature_label_i temperature_label_j target_temperature_i target_temperature_j potential_energy_i potential_energy_j log_acceptance_ratio acceptance_probability accepted

Indices start at zero. Temperatures are in K and total potential energies in eV.
Labels and energies describe the state before the exchange.
``accepted`` is 1 for acceptance and 0 for rejection; multiple pairs may share a step.
