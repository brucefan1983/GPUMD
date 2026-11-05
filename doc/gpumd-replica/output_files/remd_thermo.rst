.. _replica_remd_thermo:
.. index::
   single: gpumd_replica output files; remd_thermo

``remd_thermo.out``
===================

Each row contains the step and three columns per replica::

  step target_temperature_0 potential_energy_0 volume_0 target_temperature_1 potential_energy_1 volume_1 ...

Temperatures are thermostat targets in K, not measured kinetic temperatures.
Potential energies are totals in eV and volumes are in cubic Angstrom.
Values are recorded after any exchange at that step.

Rows are written initially and at exchange, trajectory-output, progress-report, and final boundaries.
Use the step column: the sampling interval can be nonuniform.
